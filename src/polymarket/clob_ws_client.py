from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import websockets

try:
    import orjson as _orjson

    def _json_loads(data: str | bytes) -> Any:
        return _orjson.loads(data)

    def _json_dumps(obj: Any) -> str:
        return _orjson.dumps(obj).decode()

except ImportError:
    import json as _stdlib_json  # type: ignore[no-redef]

    def _json_loads(data: str | bytes) -> Any:  # type: ignore[misc]
        return _stdlib_json.loads(data)

    def _json_dumps(obj: Any) -> str:  # type: ignore[misc]
        return _stdlib_json.dumps(obj)

from src import profiling
from src.polymarket.models import OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)


@dataclass
class WebSocketOrderBookState:
    book: OrderBook | None = None
    updated_at_monotonic: float = 0.0
    last_message_received_at: datetime | None = None
    last_latency_ms: float | None = None
    last_event_type: str | None = None


class ClobWebSocketClient:
    """Public market-data WebSocket cache — dual connections, dedicated threads.

    Architecture
    ------------
    Each connection slot runs in its own OS thread with its own asyncio event
    loop (``asyncio.new_event_loop()``).  This separates WebSocket I/O and JSON
    parsing from the main event loop that runs the scan loop and REST calls.

    Benefits
    ~~~~~~~~
    * **GIL release during JSON parsing**: orjson releases the GIL; stdlib json
      does not, but executing in a thread still keeps the main loop unblocked.
    * **Backpressure isolation**: if ``_scan_once`` is busy (REST fetch, numpy),
      the WS thread drains the socket buffer continuously, preventing TCP
      receive-window fill and message loss.
    * **Hot redundancy**: two connections — if one drops, the other keeps books
      fresh during the 5-second reconnect window.
    * **Effective latency = min(conn_0, conn_1)** on active markets.

    Thread safety
    ~~~~~~~~~~~~~
    * ``_states`` (dict): written by WS threads, read by main loop.  Dict
      ``__setitem__`` / ``__getitem__`` are atomic in CPython under the GIL.
    * ``_update_queue`` (asyncio.Queue): threads call
      ``main_loop.call_soon_threadsafe(queue.put_nowait, token_set)``
      which schedules ``put_nowait`` on the main event loop — safe.
    * Counters (``connected``, ``reconnect_count``, etc.): simple int/bool
      assignments; atomic in CPython.

    This client only consumes public order-book updates.  No authentication,
    no order placement.  Callers should fall back to REST for stale books.
    """

    _NUM_CONNECTIONS = 2  # slots; increase to 3 for extra redundancy

    def __init__(
        self,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        *,
        reconnect_delay_seconds: float = 5.0,
        stale_book_ms: int = 2000,
        audit: Any | None = None,
    ) -> None:
        self.ws_url = ws_url
        self.audit = audit
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.stale_book_ms = stale_book_ms
        # Public health indicators (written from threads, read from main loop)
        self.connected = False          # True when ≥1 slot is connected
        self.reconnect_count = 0
        self.token_update_count = 0
        self.event_triggered_recomputes = 0
        # Subscription state
        self._desired_tokens: set[str] = set()
        # Shared book state — written by threads, read by main loop
        self._states: dict[str, WebSocketOrderBookState] = {}
        # Main-loop asyncio queue; threads push via call_soon_threadsafe.
        # Each item is (updated_tokens, frame_receipt_perf_ns) so the main loop
        # can measure how long the update waited before it was drained.
        self._update_queue: asyncio.Queue[tuple[set[str], int]] = asyncio.Queue()
        # ── Per-slot thread state ──────────────────────────────────────────
        self._threads: list[threading.Thread | None] = [None] * self._NUM_CONNECTIONS
        self._thread_loops: list[asyncio.AbstractEventLoop | None] = [None] * self._NUM_CONNECTIONS
        self._ws_slots: list[Any] = [None] * self._NUM_CONNECTIONS
        self._slot_connected: list[bool] = [False] * self._NUM_CONNECTIONS
        # Bool flags written by main loop, read by WS threads (atomic in CPython)
        self._stop_requested = False
        self._resub_requested: list[bool] = [False] * self._NUM_CONNECTIONS
        # ── Main-loop-only state ───────────────────────────────────────────
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    # ── Public API (called from main event loop) ──────────────────────────────

    async def start(self, token_ids: list[str]) -> None:
        async with self._lock:
            self._desired_tokens = {t for t in token_ids if t}
            self._stop_requested = False
            self._main_loop = asyncio.get_running_loop()
            for slot in range(self._NUM_CONNECTIONS):
                t = self._threads[slot]
                if t is None or not t.is_alive():
                    new_t = threading.Thread(
                        target=self._run_thread,
                        args=(slot,),
                        daemon=True,
                        name=f"clob-ws-{slot}",
                    )
                    new_t.start()
                    self._threads[slot] = new_t

    async def close(self) -> None:
        self._stop_requested = True
        # Ask each thread to close its WS connection so recv loops unblock
        for slot in range(self._NUM_CONNECTIONS):
            thread_loop = self._thread_loops[slot]
            ws = self._ws_slots[slot]
            if ws is not None and thread_loop is not None and not thread_loop.is_closed():
                # Fire-and-forget: thread will see the close and exit recv loop
                asyncio.run_coroutine_threadsafe(ws.close(), thread_loop)
        # Join threads without blocking the main event loop
        loop = asyncio.get_running_loop()
        live = [t for t in self._threads if t is not None and t.is_alive()]
        if live:
            await asyncio.gather(
                *(loop.run_in_executor(None, t.join, 3.0) for t in live),
                return_exceptions=True,
            )

    async def update_subscriptions(self, token_ids: list[str]) -> None:
        async with self._lock:
            new_tokens = {t for t in token_ids if t}
            if new_tokens == self._desired_tokens:
                return
            self._desired_tokens = new_tokens
            # Signal each WS thread to re-subscribe on its next recv iteration
            for slot in range(self._NUM_CONNECTIONS):
                self._resub_requested[slot] = True

    async def wait_for_updates(self, timeout: float | None = None) -> set[str]:
        try:
            first = await asyncio.wait_for(self._update_queue.get(), timeout=timeout)
        except TimeoutError:
            return set()
        first_tokens, first_receipt_ns = first
        updated = set(first_tokens)
        # Drain any additional updates already in the queue
        while True:
            try:
                tokens, _ = self._update_queue.get_nowait()
                updated.update(tokens)
            except asyncio.QueueEmpty:
                break
        if updated:
            self.event_triggered_recomputes += 1
            # queue_wait = frame receipt (WS thread) → main-loop drain.  Uses the
            # oldest (first) drained item, which is the longest any update waited.
            if profiling.ENABLED and (rec := profiling.RECORDER) is not None:
                rec.record_series("queue_wait", time.perf_counter_ns() - first_receipt_ns)
        return updated

    def seed_books(self, books: dict[str, OrderBook]) -> None:
        now = time.monotonic()
        received_at = datetime.now(UTC)
        for token_id, book in books.items():
            self._states[token_id] = WebSocketOrderBookState(
                book=book,
                updated_at_monotonic=now,
                last_message_received_at=received_at,
                last_latency_ms=0.0,
                last_event_type="seed",
            )

    def get_book(self, token_id: str) -> OrderBook | None:
        state = self._states.get(token_id)
        return state.book if state else None

    def get_book_age_ms(self, token_id: str) -> float | None:
        state = self._states.get(token_id)
        if state is None or state.updated_at_monotonic <= 0:
            return None
        return max(0.0, (time.monotonic() - state.updated_at_monotonic) * 1000)

    def is_stale(self, token_id: str) -> bool:
        age = self.get_book_age_ms(token_id)
        return age is None or age > self.stale_book_ms

    def get_update_latency_ms(self, token_id: str) -> float | None:
        state = self._states.get(token_id)
        return state.last_latency_ms if state else None

    def save_snapshot(self, path: "Path") -> None:
        """Synchronously persist current book state to *path* (best-effort).

        Safe to call from the main event loop — performs a shallow copy of
        ``_states`` under the GIL, then delegates atomic JSON write to
        ``book_snapshot.save_snapshot``.  Never raises.
        """
        try:
            from src.book_snapshot import save_snapshot as _save
            _save(dict(self._states), path)
        except Exception as exc:
            logger.warning("ClobWebSocketClient.save_snapshot failed: %s", exc)

    async def stream(self, token_ids: list[str]) -> AsyncIterator[set[str]]:
        await self.start(token_ids)
        while not self._stop_requested:
            yield await self.wait_for_updates(timeout=None)

    # ── Thread entry point ────────────────────────────────────────────────────

    def _run_thread(self, slot: int) -> None:
        """OS thread entry point: creates a private asyncio event loop and
        runs the WS connection coroutine until stop is requested."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._thread_loops[slot] = loop
        try:
            loop.run_until_complete(self._ws_loop_async(slot))
        except Exception as exc:  # noqa: BLE001
            logger.debug("WS thread[%d] exited with error: %s", slot, exc)
        finally:
            self._thread_loops[slot] = None
            self._slot_connected[slot] = False
            self._ws_slots[slot] = None
            self.connected = any(self._slot_connected)
            loop.close()

    # ── WS loop (runs inside thread's event loop) ─────────────────────────────

    async def _ws_loop_async(self, slot: int) -> None:
        """Full WebSocket lifecycle for one connection slot.

        Runs in the slot's private event loop (thread context).
        Reconnects automatically with staggered delay.
        Notifies the main event loop via call_soon_threadsafe.
        """
        # Stagger initial connect so both slots don't hammer the server together
        if slot > 0:
            await asyncio.sleep(slot * 1.0)

        main_loop = self._main_loop

        while not self._stop_requested:
            try:
                async with websockets.connect(
                    self.ws_url, ping_interval=8, ping_timeout=10
                ) as ws:
                    self._ws_slots[slot] = ws
                    self._slot_connected[slot] = True
                    self.connected = True
                    self._resub_requested[slot] = False
                    await self._send_subscription(ws)

                    async for raw_message in ws:
                        if self._stop_requested:
                            break

                        # Check for subscription update request
                        if self._resub_requested[slot]:
                            self._resub_requested[slot] = False
                            await self._send_subscription(ws)

                        # Timestamps captured before CPU-bound JSON parsing
                        receipt_wall_ns = time.time_ns()
                        receipt_perf_ns = time.perf_counter_ns()

                        # JSON parse + state mutation — happens in thread,
                        # off the main event loop.
                        for message in self._coerce_messages(raw_message):
                            updated = self._handle_message(
                                message,
                                raw_message=raw_message,
                                local_receipt_ts_ns=receipt_wall_ns,
                                receipt_perf_ns=receipt_perf_ns,
                            )
                            if updated and main_loop is not None:
                                # Thread-safe delivery to the main event loop.
                                # call_soon_threadsafe is the ONLY asyncio call
                                # that is thread-safe without a running loop.
                                # Carry the frame receipt time so the main loop
                                # can measure queue_wait when it drains this item.
                                main_loop.call_soon_threadsafe(
                                    self._update_queue.put_nowait,
                                    (updated, receipt_perf_ns),
                                )

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self._stop_requested:
                    break
                self._slot_connected[slot] = False
                self._ws_slots[slot] = None
                self.connected = any(self._slot_connected)
                self.reconnect_count += 1
                if self.audit is not None and getattr(self.audit, "enabled", False):
                    now_wall_ns = time.time_ns()
                    now_perf_ns = time.perf_counter_ns()
                    self.audit.record_network(
                        method="WS",
                        endpoint=self.ws_url,
                        status_code=None,
                        wall_start_ns=now_wall_ns,
                        wall_end_ns=now_wall_ns,
                        perf_start_ns=now_perf_ns,
                        perf_end_ns=now_perf_ns,
                        exception=f"websocket[{slot}] reconnect: {exc}",
                    )
                logger.info("WebSocket[%d] reconnect after error: %s", slot, exc)
                await asyncio.sleep(self.reconnect_delay_seconds)
            finally:
                self._slot_connected[slot] = False
                self._ws_slots[slot] = None
                self.connected = any(self._slot_connected)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _send_subscription(self, ws: Any) -> None:
        await ws.send(
            _json_dumps(
                {
                    "type": "market",
                    "assets_ids": sorted(self._desired_tokens),
                    "custom_feature_enabled": True,
                }
            )
        )

    def _coerce_messages(self, raw_message: str | bytes) -> list[dict[str, Any]]:
        try:
            payload = _json_loads(raw_message)
        except Exception:  # noqa: BLE001
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

    def _handle_message(
        self,
        payload: dict[str, Any],
        *,
        raw_message: str | bytes | None = None,
        local_receipt_ts_ns: int | None = None,
        receipt_perf_ns: int | None = None,
    ) -> set[str]:
        received_at = datetime.now(UTC)
        received_monotonic = time.monotonic()
        receipt_wall_ns = local_receipt_ts_ns or time.time_ns()
        receipt_perf_ns = receipt_perf_ns or time.perf_counter_ns()
        processing_wall_ns = time.time_ns()
        processing_perf_ns = time.perf_counter_ns()
        audit_event_id = None
        if self.audit is not None and getattr(self.audit, "enabled", False):
            audit_event_id = self.audit.record_ws_event(
                payload=payload,
                raw_message=raw_message,
                local_receipt_ts_ns=receipt_wall_ns,
                receipt_perf_ns=receipt_perf_ns,
                processing_ts_ns=processing_wall_ns,
                processing_perf_ns=processing_perf_ns,
            )
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        if event_type == "book":
            token_id = str(payload.get("asset_id") or payload.get("assetId") or "")
            if not token_id:
                return set()
            book = OrderBook.from_api(payload, token_id=token_id)
            book.raw_json["_audit_source"] = "websocket"
            book.raw_json["_audit_event_id"] = audit_event_id
            self._store_book(
                token_id, book, event_type, received_at, received_monotonic,
                _message_latency_ms(payload, received_at),
            )
            return {token_id}
        if event_type == "price_change":
            updated: set[str] = set()
            for change in payload.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                token_id = str(change.get("asset_id") or change.get("assetId") or "")
                if not token_id:
                    continue
                self._apply_price_change(
                    token_id, change, payload, received_at, received_monotonic, audit_event_id
                )
                updated.add(token_id)
            return updated
        if event_type == "best_bid_ask":
            token_id = str(payload.get("asset_id") or payload.get("assetId") or "")
            if token_id and self._apply_best_bid_ask(
                token_id, payload, received_at, received_monotonic, audit_event_id
            ):
                return {token_id}
        return set()

    def _store_book(
        self,
        token_id: str,
        book: OrderBook,
        event_type: str,
        received_at: datetime,
        received_monotonic: float,
        latency_ms: float | None,
    ) -> None:
        self._states[token_id] = WebSocketOrderBookState(
            book=book,
            updated_at_monotonic=received_monotonic,
            last_message_received_at=received_at,
            last_latency_ms=latency_ms,
            last_event_type=event_type,
        )
        self.token_update_count += 1

    def _apply_price_change(
        self,
        token_id: str,
        change: dict[str, Any],
        envelope: dict[str, Any],
        received_at: datetime,
        received_monotonic: float,
        audit_event_id: str | None = None,
    ) -> None:
        state = self._states.get(token_id) or WebSocketOrderBookState(
            book=OrderBook(asset_id=token_id)
        )
        book = state.book or OrderBook(asset_id=token_id)
        side = str(change.get("side") or "").upper()
        try:
            price = Decimal(str(change.get("price") or "0"))
            size = Decimal(str(change.get("size") or "0"))
        except Exception:
            return
        if side == "BUY":
            book.apply_bid_delta(price, size)
        else:
            book.apply_ask_delta(price, size)
        book.timestamp = _coerce_timestamp(envelope.get("timestamp"))
        book.raw_json["_audit_source"] = "websocket"
        book.raw_json["_audit_event_id"] = audit_event_id
        book.raw_json["_audit_event_type"] = "price_change"
        state.book = book
        state.updated_at_monotonic = received_monotonic
        state.last_message_received_at = received_at
        state.last_latency_ms = _message_latency_ms(envelope, received_at)
        state.last_event_type = "price_change"
        self._states[token_id] = state
        self.token_update_count += 1

    def _apply_best_bid_ask(
        self,
        token_id: str,
        payload: dict[str, Any],
        received_at: datetime,
        received_monotonic: float,
        audit_event_id: str | None = None,
    ) -> bool:
        best_bid = _decimal_or_none(
            payload.get("best_bid") or payload.get("bestBid") or payload.get("bid")
        )
        best_ask = _decimal_or_none(
            payload.get("best_ask") or payload.get("bestAsk") or payload.get("ask")
        )
        if best_bid is None and best_ask is None:
            return False

        state = self._states.get(token_id) or WebSocketOrderBookState(
            book=OrderBook(asset_id=token_id)
        )
        book = state.book or OrderBook(asset_id=token_id)
        if best_bid is not None:
            book.replace_best_bid(best_bid)
        if best_ask is not None:
            book.replace_best_ask(best_ask)
        book.timestamp = _coerce_timestamp(payload.get("timestamp")) or book.timestamp
        has_depth = any(
            payload.get(key) not in (None, "")
            for key in [
                "best_bid_size", "bestBidSize", "bid_size",
                "best_ask_size", "bestAskSize", "ask_size",
            ]
        )
        book.raw_json["best_bid_ask_depth_untrusted"] = not has_depth
        book.raw_json["best_bid_ask_updated_at"] = received_at.isoformat()
        book.raw_json["_audit_source"] = "websocket"
        book.raw_json["_audit_event_id"] = audit_event_id
        book.raw_json["_audit_event_type"] = "best_bid_ask"
        state.book = book
        state.updated_at_monotonic = received_monotonic
        state.last_message_received_at = received_at
        state.last_latency_ms = _message_latency_ms(payload, received_at)
        state.last_event_type = "best_bid_ask"
        self._states[token_id] = state
        self.token_update_count += 1
        return True


# ── Module-level helpers ──────────────────────────────────────────────────────

def _coerce_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        numeric = int(str(value))
    except (TypeError, ValueError):
        return None
    seconds = numeric / 1000 if numeric > 10_000_000_000 else numeric
    return datetime.fromtimestamp(seconds, tz=UTC)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value))
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _message_latency_ms(payload: dict[str, Any], received_at: datetime) -> float | None:
    timestamp = payload.get("timestamp")
    if timestamp in (None, ""):
        return None
    try:
        numeric = int(str(timestamp))
    except (TypeError, ValueError):
        return None
    event_ms = numeric if numeric > 10_000_000_000 else numeric * 1000
    event_time = datetime.fromtimestamp(event_ms / 1000, tz=UTC)
    latency = (received_at - event_time).total_seconds() * 1000
    if latency < -1000 or latency > 60_000:
        return None
    return max(0.0, latency)
