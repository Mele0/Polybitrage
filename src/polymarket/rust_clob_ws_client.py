"""Python adapter that wraps the Rust ``polymarket_rs.RustWsClient`` extension
and exposes the same interface as ``ClobWebSocketClient``.

The Rust client runs two tokio async tasks in a dedicated multi-threaded
runtime — one per WS connection slot.  JSON parsing, byte-to-float
conversion, and BTreeMap book updates all happen in Rust without holding the
GIL, cutting WS→book latency from ~500 μs (Python) to ~10 μs (Rust).

Key interface differences from ``ClobWebSocketClient``
------------------------------------------------------
* No asyncio internal state: there is no ``asyncio.Queue``.  Instead,
  ``wait_for_updates`` calls ``loop.run_in_executor(None, rust.poll_updates,
  timeout_ms)`` which blocks an OS thread (not the event loop) until data
  arrives.
* Audit integration: the Rust client does not emit audit events.  If the
  forensic audit is needed, keep using the Python WS client.
* ``save_snapshot`` delegates to ``book_snapshot.save_snapshot`` via a thin
  adapter.

Import guard
------------
This module imports ``polymarket_rs`` at the top level.  If the wheel has not
been built (``cd rust_ws_client && maturin build --release``) the import will
raise ``ImportError``.  ``_make_provider`` in simulator.py catches that and
falls back to the Python WS client automatically.
"""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polymarket_rs  # noqa: F401 — raises ImportError if wheel not built

from src import profiling
from src.polymarket.models import OrderBook

if TYPE_CHECKING:
    pass


class RustClobWsClient:
    """Drop-in replacement for ``ClobWebSocketClient`` powered by a Rust backend.

    Implements the full interface expected by ``WebSocketOrderBookProvider``:

        start / close / update_subscriptions / wait_for_updates /
        get_book / seed_books / save_snapshot /
        connected / reconnect_count / token_update_count / event_triggered_recomputes /
        get_book_age_ms / is_stale / get_update_latency_ms
    """

    def __init__(
        self,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        *,
        reconnect_delay_seconds: float = 5.0,
        stale_book_ms: int = 2000,
        audit: Any | None = None,  # not used — Rust client has no audit path
    ) -> None:
        self._inner = polymarket_rs.RustWsClient(
            ws_url, reconnect_delay_s=reconnect_delay_seconds
        )
        self.stale_book_ms = stale_book_ms
        # audit is accepted but ignored; keep signature compatible.
        self._desired_tokens: set[str] = set()
        self._lock = asyncio.Lock()
        self._stop_requested = False
        self.event_triggered_recomputes = 0

    # ── Public health indicators (read by WebSocketOrderBookProvider) ─────────

    @property
    def connected(self) -> bool:
        return self._inner.is_connected()

    @property
    def reconnect_count(self) -> int:
        return self._inner.reconnect_count()

    @property
    def token_update_count(self) -> int:
        return self._inner.token_update_count()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self, token_ids: list[str]) -> None:
        async with self._lock:
            self._desired_tokens = {t for t in token_ids if t}
            self._stop_requested = False
            self._inner.start(sorted(self._desired_tokens))

    async def close(self) -> None:
        self._stop_requested = True
        self._inner.stop()

    async def update_subscriptions(self, token_ids: list[str]) -> None:
        async with self._lock:
            new_tokens = {t for t in token_ids if t}
            if new_tokens == self._desired_tokens:
                return
            self._desired_tokens = new_tokens
            self._inner.update_subscriptions(sorted(self._desired_tokens))

    # ── Hot path: wait for WS updates ────────────────────────────────────────

    async def wait_for_updates(self, timeout: float | None = None) -> set[str]:
        """Await book updates without blocking the asyncio event loop.

        Runs ``RustWsClient.poll_updates`` in a thread-pool executor so the
        Rust condvar-wait releases the GIL and the event loop remains free.

        When profiling is on, also records two pipeline latencies that are
        invisible from inside the scan:

          * ``queue_wait``    — how long the oldest drained update sat in the
            Rust queue before this poll returned (rises when we are falling
            behind the WS firehose).  Read from ``last_queue_wait_ms``.
          * ``wakeup_latency`` — executor → event-loop handoff: from the instant
            ``poll_updates`` returned on the worker thread to the instant this
            coroutine resumes on the main loop (OS/scheduler thread latency).
        """
        timeout_ms = int((timeout or 5.0) * 1000)
        loop = asyncio.get_running_loop()

        if not profiling.ENABLED:
            updated_ids: list[str] = await loop.run_in_executor(
                None, self._inner.poll_updates, timeout_ms
            )
            if updated_ids:
                self.event_triggered_recomputes += 1
            return set(updated_ids)

        # Profiled path: stamp the worker thread the moment data is ready, then
        # measure how long until the loop actually resumes us.  perf_counter_ns
        # is process-wide monotonic, so the cross-thread delta is valid.
        def _poll() -> tuple[list[str], int]:
            ids = self._inner.poll_updates(timeout_ms)
            return ids, time.perf_counter_ns()

        updated_ids, ready_ns = await loop.run_in_executor(None, _poll)
        if updated_ids:
            self.event_triggered_recomputes += 1
            rec = profiling.RECORDER
            if rec is not None:
                rec.record_series("wakeup_latency", time.perf_counter_ns() - ready_ns)
                qw = self._inner.last_queue_wait_ms()
                if qw is not None:
                    rec.record_series("queue_wait", qw * 1e6)
        return set(updated_ids)

    # ── Book access ──────────────────────────────────────────────────────────

    def get_book(self, token_id: str) -> OrderBook | None:
        """Reconstruct a Python ``OrderBook`` from the Rust book state.

        Uses ``OrderBook.from_float_data`` to bypass all Decimal round-trips:
        Rust already stores prices/sizes as f64, so we go float → SortedDict
        directly without the expensive Decimal(str(float)) conversion that the
        general ``__init__`` path requires.
        """
        # Profiling: split the Rust→Python boundary into the Rust snapshot call
        # (s2a) and the Python OrderBook reconstruction (s2b).  Gated so the
        # non-profiled path below is untouched (one bool check).
        if profiling.ENABLED and profiling.RECORDER is not None:
            _t0 = time.perf_counter_ns()
            snap = self._inner.get_book_snapshot(token_id)
            _t1 = time.perf_counter_ns()
            if snap is None:
                profiling.RECORDER.record("s2a_rust_snapshot", _t1 - _t0)
                return None
            bids_raw, asks_raw, _ = snap
            book = OrderBook.from_float_data(token_id, bids_raw, asks_raw)
            _t2 = time.perf_counter_ns()
            profiling.RECORDER.record("s2a_rust_snapshot", _t1 - _t0)
            profiling.RECORDER.record("s2b_python_rebuild", _t2 - _t1)
            return book
        snap = self._inner.get_book_snapshot(token_id)
        if snap is None:
            return None
        bids_raw, asks_raw, _ = snap
        return OrderBook.from_float_data(token_id, bids_raw, asks_raw)

    def seed_books(self, books: dict[str, OrderBook]) -> None:
        """Pre-warm Rust book state from REST-fetched ``OrderBook`` objects."""
        for token_id, book in books.items():
            bids = [(float(lvl.price), float(lvl.size)) for lvl in book.bids]
            asks = [(float(lvl.price), float(lvl.size)) for lvl in book.asks]
            self._inner.seed_book(token_id, bids, asks)

    def get_book_age_ms(self, token_id: str) -> float | None:
        return self._inner.get_book_age_ms(token_id)

    def is_stale(self, token_id: str) -> bool:
        age = self.get_book_age_ms(token_id)
        return age is None or age > self.stale_book_ms

    def get_update_latency_ms(self, token_id: str) -> float | None:
        # Wire latency (exchange event → local receipt) for the last message that
        # touched this token, now tracked in Rust from the WS `timestamp` field.
        return self._inner.get_update_latency_ms(token_id)

    def ws_stats(self) -> dict[str, float]:
        """Deep WS telemetry from the Rust backend (read on the /metrics scrape).

        All cheap relaxed-atomic reads: per-type message counts, frame/byte
        throughput, parse errors, connected-slot count, feed staleness, and the
        last drain size (tick backpressure depth).
        """
        inner = self._inner
        return {
            "book_msgs": float(inner.book_msg_count()),
            "price_change_msgs": float(inner.price_change_msg_count()),
            "bba_msgs": float(inner.bba_msg_count()),
            "frames_recv": float(inner.frames_received()),
            "bytes_recv": float(inner.bytes_received()),
            "parse_errors": float(inner.parse_error_count()),
            "connected_slots": float(inner.connected_slot_count()),
            "ms_since_last_message": round(float(inner.ms_since_last_message()), 1),
            "last_drain_size": float(inner.last_drain_size()),
        }

    # ── Snapshot support (Tier-3 book persistence) ───────────────────────────

    def save_snapshot(self, path: "Path") -> None:
        """Persist current Rust book state to ``path`` via book_snapshot.

        Builds a lightweight adapter dict compatible with
        ``book_snapshot.save_snapshot`` and delegates atomically.
        Never raises — failures are logged at WARNING level.
        """
        try:
            from src.book_snapshot import save_snapshot as _save  # noqa: PLC0415

            token_ids = self._inner.all_token_ids()
            states: dict[str, Any] = {}
            now = time.monotonic()
            received_at = datetime.now(UTC)
            for tid in token_ids:
                book = self.get_book(tid)
                if book is not None:
                    states[tid] = _RustBookStateAdapter(
                        book, updated_at_monotonic=now,
                        last_message_received_at=received_at,
                    )
            _save(states, path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "RustClobWsClient.save_snapshot failed: %s", exc
            )

    # ── stream helper (mirrors ClobWebSocketClient.stream) ───────────────────

    async def stream(self, token_ids: list[str]):  # type: ignore[override]
        await self.start(token_ids)
        while not self._stop_requested:
            yield await self.wait_for_updates(timeout=None)


class _RustBookStateAdapter:
    """Minimal adapter so ``book_snapshot.save_snapshot`` can read Rust books.

    ``book_snapshot._book_to_dict`` only accesses ``state.book._bids_sd``
    and ``state.book._asks_sd``.  We pass through a real ``OrderBook`` so
    those attributes are present.
    """

    __slots__ = ("book", "updated_at_monotonic", "last_message_received_at")

    def __init__(
        self,
        book: OrderBook,
        *,
        updated_at_monotonic: float,
        last_message_received_at: datetime,
    ) -> None:
        self.book = book
        self.updated_at_monotonic = updated_at_monotonic
        self.last_message_received_at = last_message_received_at
