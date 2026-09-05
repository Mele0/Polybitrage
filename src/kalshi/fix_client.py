"""Kalshi FIX 5.0 SP2 market data client (port 8233).

Implements a real-time order book feed that replaces REST polling for Kalshi,
dropping Kalshi book-fetch latency from ~100–200 ms to <5 ms once subscribed.

Provides the same interface as ClobWebSocketClient so it works as a drop-in
with WebSocketOrderBookProvider.

Requirements
------------
- A Kalshi FIX API key (separate from REST key). Request via the Kalshi portal.
- Configure via env vars (or pass directly to the constructor):
    KALSHI_FIX_API_KEY    — FIX SenderCompID / username
    KALSHI_FIX_API_SECRET — FIX password

FIX session details (from Kalshi connectivity docs)
----------------------------------------------------
Host  : mm.fix.elections.kalshi.com
Port  : 8233  no-retransmission (fastest), 8234 with-retransmission (reliable)
Proto : FIXT.1.1 / FIX50SP2 over TLS 1.2+
Keys  : one FIX connection per API key; use separate keys for MD and order-entry

Usage
-----
    from src.kalshi.fix_client import KalshiFixMarketDataClient

    client = KalshiFixMarketDataClient(
        api_key=os.environ["KALSHI_FIX_API_KEY"],
        api_secret=os.environ["KALSHI_FIX_API_SECRET"],
    )
    # Pass to WebSocketOrderBookProvider in place of ClobWebSocketClient
"""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.kalshi.client import kalshi_token_id, parse_kalshi_token_id
from src.polymarket.models import OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)

SOH = b"\x01"
ZERO = Decimal("0")
ONE = Decimal("1")

# ── FIX message type constants ─────────────────────────────────────────────────
_HEARTBEAT = b"0"
_TEST_REQUEST = b"1"
_LOGOUT = b"5"
_LOGON = b"A"
_MD_REQUEST = b"V"
_MD_SNAPSHOT = b"W"
_MD_INCREMENTAL = b"X"

# ── FIX tag constants ──────────────────────────────────────────────────────────
_T = {
    "BeginString": 8,
    "BodyLength": 9,
    "MsgType": 35,
    "SenderCompID": 49,
    "TargetCompID": 56,
    "MsgSeqNum": 34,
    "SendingTime": 52,
    "CheckSum": 10,
    "EncryptMethod": 98,
    "HeartBtInt": 108,
    "TestReqID": 112,
    "Username": 553,
    "Password": 554,
    "DefaultApplVerID": 1137,
    "MDReqID": 262,
    "SubscriptionRequestType": 263,
    "MarketDepth": 264,
    "MDUpdateType": 265,
    "NoMDEntryTypes": 267,
    "MDEntryType": 269,
    "MDEntryPx": 270,
    "MDEntrySize": 271,
    "MDUpdateAction": 279,
    "NoMDEntries": 268,
    "NoRelatedSym": 146,
    "Symbol": 55,
    "Text": 58,
}

# MDEntryType
_BID_TYPE = b"0"
_OFFER_TYPE = b"1"

# MDUpdateAction
_NEW = b"0"
_CHANGE = b"1"
_DELETE = b"2"


# ── Low-level FIX message building & parsing ───────────────────────────────────


def _fix_now() -> bytes:
    return datetime.now(UTC).strftime("%Y%m%d-%H:%M:%S.%f")[:23].encode()


def _fix_checksum(data: bytes) -> bytes:
    return f"{sum(data) % 256:03d}".encode()


def _fix_build(msg_type: bytes, body_fields: list[tuple[int, bytes | str]], sender: bytes, target: bytes, seq: int) -> bytes:
    """Build a complete FIX message with header, body fields, and checksum."""
    body = b""
    body += b"35=" + msg_type + SOH
    body += b"49=" + sender + SOH
    body += b"56=" + target + SOH
    body += b"34=" + str(seq).encode() + SOH
    body += b"52=" + _fix_now() + SOH
    for tag, val in body_fields:
        bval = val if isinstance(val, bytes) else str(val).encode()
        body += str(tag).encode() + b"=" + bval + SOH
    header = b"8=FIXT.1.1" + SOH + b"9=" + str(len(body)).encode() + SOH
    full = header + body
    return full + b"10=" + _fix_checksum(full) + SOH


def _fix_parse(raw: bytes) -> dict[int, bytes]:
    """Parse a raw FIX message into {tag: value} dict. Handles repeated tags via list accumulation."""
    fields: dict[int, bytes] = {}
    for pair in raw.split(SOH):
        if b"=" not in pair:
            continue
        tag_b, val = pair.split(b"=", 1)
        if not tag_b.isdigit():
            continue
        tag = int(tag_b)
        fields[tag] = val
    return fields


def _fix_parse_repeated(raw: bytes) -> list[dict[int, bytes]]:
    """Parse a FIX message with repeating groups into a list of field dicts.

    Returns one dict per repeating group entry (split on tag 279 for incremental
    or tag 269 for snapshot entry-type changes in sequential groups).
    """
    all_fields: list[tuple[int, bytes]] = []
    for pair in raw.split(SOH):
        if b"=" not in pair:
            continue
        tag_b, val = pair.split(b"=", 1)
        if tag_b.isdigit():
            all_fields.append((int(tag_b), val))
    return all_fields


# ── WebSocketOrderBookState compat (imported by clob_ws_client) ───────────────


@dataclass
class _BookState:
    book: OrderBook | None = None
    updated_at_monotonic: float = 0.0
    last_message_received_at: datetime | None = None
    last_latency_ms: float | None = None
    last_event_type: str | None = None


# ── KalshiFixMarketDataClient ─────────────────────────────────────────────────


class KalshiFixMarketDataClient:
    """Real-time Kalshi order book feed via FIX 5.0 SP2 (port 8233).

    Drop-in replacement for ClobWebSocketClient — exposes the same interface
    so it works transparently with WebSocketOrderBookProvider.

    Token ID format used here is the same as in KalshiClient:
    ``kalshi:<TICKER>:yes`` and ``kalshi:<TICKER>:no``.

    The FIX feed subscribes per ticker and derives both YES and NO books
    from the bid/offer prices using the complement relationship:
    NO ask = 1 - YES bid,  NO bid = 1 - YES ask.
    """

    HOST = "mm.fix.elections.kalshi.com"
    PORT_FAST = 8233
    PORT_RELIABLE = 8234
    TARGET_COMP_ID = b"KALSHI"
    DEFAULT_APP_VER = b"9"  # FIX50SP2

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        host: str = HOST,
        port: int = PORT_FAST,
        heartbeat_interval: int = 30,
        stale_book_ms: int = 2000,
        reconnect_delay_seconds: float = 5.0,
    ) -> None:
        self.api_key = (api_key or os.environ.get("KALSHI_FIX_API_KEY", "")).strip()
        self.api_secret = (api_secret or os.environ.get("KALSHI_FIX_API_SECRET", "")).strip()
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Kalshi FIX requires KALSHI_FIX_API_KEY and KALSHI_FIX_API_SECRET. "
                "Set via env vars or pass to the constructor."
            )
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.stale_book_ms = stale_book_ms
        self.reconnect_delay_seconds = reconnect_delay_seconds

        self._sender = self.api_key.encode()
        self._seq = 1

        self.connected = False
        self.reconnect_count = 0
        self.token_update_count = 0
        self.event_triggered_recomputes = 0

        self._desired_tokens: set[str] = set()
        self._states: dict[str, _BookState] = {}
        self._update_queue: asyncio.Queue[set[str]] = asyncio.Queue()
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self._req_id_counter = 0

    # ── Public interface (mirrors ClobWebSocketClient) ────────────────────────

    async def start(self, token_ids: list[str]) -> None:
        async with self._lock:
            self._desired_tokens = {tid for tid in token_ids if tid}
            self._stop_requested = False
            if self._runner_task is None or self._runner_task.done():
                self._runner_task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self._stop_requested = True
        if self._writer is not None:
            try:
                await self._send_logout("client shutdown")
            except Exception:
                pass
            self._writer.close()
        if self._runner_task is not None:
            try:
                await asyncio.wait_for(self._runner_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError):
                self._runner_task.cancel()

    async def update_subscriptions(self, token_ids: list[str]) -> None:
        async with self._lock:
            new_tokens = {tid for tid in token_ids if tid}
            if new_tokens == self._desired_tokens:
                return
            added = new_tokens - self._desired_tokens
            self._desired_tokens = new_tokens
            if self.connected and self._writer is not None and added:
                for ticker in self._unique_tickers(added):
                    await self._send_md_request(ticker)

    def seed_books(self, books: dict[str, OrderBook]) -> None:
        """Seed initial books from a REST fallback (called by WebSocketOrderBookProvider)."""
        now = time.monotonic()
        received_at = datetime.now(UTC)
        for token_id, book in books.items():
            self._states[token_id] = _BookState(
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

    async def wait_for_updates(self, timeout: float | None = None) -> set[str]:
        try:
            first = await asyncio.wait_for(self._update_queue.get(), timeout=timeout)
        except TimeoutError:
            return set()
        updated = set(first)
        while True:
            try:
                updated.update(self._update_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if updated:
            self.event_triggered_recomputes += 1
        return updated

    # ── FIX session loop ──────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stop_requested:
            try:
                ssl_ctx = ssl.create_default_context()
                reader, writer = await asyncio.open_connection(self.host, self.port, ssl=ssl_ctx)
                self._writer = writer
                self._seq = 1
                await self._send_logon()
                self.connected = True
                logger.info("Kalshi FIX connected to %s:%d", self.host, self.port)
                # Subscribe to all currently desired tickers
                for ticker in self._unique_tickers(self._desired_tokens):
                    await self._send_md_request(ticker)
                # Start heartbeat sender
                hb_task = asyncio.create_task(self._heartbeat_loop())
                try:
                    await self._read_loop(reader)
                finally:
                    hb_task.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self._stop_requested:
                    break
                self.connected = False
                self.reconnect_count += 1
                logger.warning("Kalshi FIX disconnected: %s. Reconnecting in %.1fs.", exc, self.reconnect_delay_seconds)
                await asyncio.sleep(self.reconnect_delay_seconds)
            finally:
                self.connected = False
                self._writer = None

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        """Read raw FIX bytes, frame complete messages, dispatch to handlers."""
        buf = b""
        while not self._stop_requested:
            chunk = await reader.read(65536)
            if not chunk:
                raise ConnectionResetError("FIX feed closed by server")
            buf += chunk
            # Extract complete FIX messages (end with 10=xxx\x01)
            while True:
                end = buf.find(b"10=")
                if end == -1:
                    break
                eom = buf.find(SOH, end)
                if eom == -1:
                    break
                msg = buf[: eom + 1]
                buf = buf[eom + 1 :]
                self._dispatch(msg)

    def _dispatch(self, raw: bytes) -> None:
        fields = _fix_parse(raw)
        msg_type = fields.get(_T["MsgType"], b"")
        if msg_type == _HEARTBEAT:
            return
        if msg_type == _TEST_REQUEST:
            test_req_id = fields.get(_T["TestReqID"], b"")
            asyncio.create_task(self._send_heartbeat(test_req_id))
        elif msg_type == _LOGOUT:
            text = fields.get(_T["Text"], b"").decode(errors="replace")
            logger.warning("Kalshi FIX LOGOUT received: %s", text)
            raise ConnectionResetError(f"Server sent LOGOUT: {text}")
        elif msg_type == _MD_SNAPSHOT:
            self._handle_snapshot(raw, fields)
        elif msg_type == _MD_INCREMENTAL:
            self._handle_incremental(raw)

    # ── FIX market data handlers ──────────────────────────────────────────────

    def _handle_snapshot(self, raw: bytes, header: dict[int, bytes]) -> None:
        """MarketDataSnapshotFullRefresh (W): rebuild full book for a ticker."""
        ticker = header.get(_T["Symbol"], b"").decode(errors="replace").upper()
        if not ticker:
            return
        all_fields = _fix_parse_repeated(raw)
        yes_bids: list[OrderBookLevel] = []
        yes_asks: list[OrderBookLevel] = []
        i = 0
        while i < len(all_fields):
            tag, val = all_fields[i]
            if tag == _T["MDEntryType"]:
                entry_type = val
                price_dec: Decimal | None = None
                size_dec: Decimal | None = None
                j = i + 1
                while j < len(all_fields) and all_fields[j][0] != _T["MDEntryType"]:
                    t2, v2 = all_fields[j]
                    if t2 == _T["MDEntryPx"]:
                        try:
                            price_dec = Decimal(v2.decode())
                        except Exception:
                            pass
                    elif t2 == _T["MDEntrySize"]:
                        try:
                            size_dec = Decimal(v2.decode())
                        except Exception:
                            pass
                    j += 1
                if price_dec is not None and size_dec is not None and size_dec > ZERO:
                    level = OrderBookLevel(price=price_dec, size=size_dec)
                    if entry_type == _BID_TYPE:
                        yes_bids.append(level)
                    elif entry_type == _OFFER_TYPE:
                        yes_asks.append(level)
                i = j
            else:
                i += 1

        updated = self._update_books_from_yes_side(ticker, yes_bids, yes_asks, event="snapshot")
        if updated:
            self._update_queue.put_nowait(updated)

    def _handle_incremental(self, raw: bytes) -> None:
        """MarketDataIncrementalRefresh (X): apply per-level deltas."""
        all_fields = _fix_parse_repeated(raw)
        updates_by_ticker: dict[str, tuple[list[tuple[str, Decimal, Decimal]], list[tuple[str, Decimal, Decimal]]]] = {}
        i = 0
        while i < len(all_fields):
            tag, val = all_fields[i]
            if tag == _T["MDUpdateAction"]:
                action = val  # b"0"=New, b"1"=Change, b"2"=Delete
                entry_type: bytes | None = None
                ticker = ""
                price_dec: Decimal | None = None
                size_dec = ZERO
                j = i + 1
                while j < len(all_fields) and all_fields[j][0] != _T["MDUpdateAction"]:
                    t2, v2 = all_fields[j]
                    if t2 == _T["MDEntryType"]:
                        entry_type = v2
                    elif t2 == _T["Symbol"]:
                        ticker = v2.decode(errors="replace").upper()
                    elif t2 == _T["MDEntryPx"]:
                        try:
                            price_dec = Decimal(v2.decode())
                        except Exception:
                            pass
                    elif t2 == _T["MDEntrySize"]:
                        try:
                            size_dec = Decimal(v2.decode())
                        except Exception:
                            pass
                    j += 1
                if ticker and price_dec is not None and entry_type in (_BID_TYPE, _OFFER_TYPE):
                    effective_size = ZERO if action == _DELETE else size_dec
                    side = "bid" if entry_type == _BID_TYPE else "ask"
                    if ticker not in updates_by_ticker:
                        updates_by_ticker[ticker] = ([], [])
                    bid_deltas, ask_deltas = updates_by_ticker[ticker]
                    if side == "bid":
                        bid_deltas.append((action.decode(), price_dec, effective_size))
                    else:
                        ask_deltas.append((action.decode(), price_dec, effective_size))
                i = j
            else:
                i += 1

        updated: set[str] = set()
        now = time.monotonic()
        received_at = datetime.now(UTC)
        for ticker, (bid_deltas, ask_deltas) in updates_by_ticker.items():
            affected = self._apply_incremental_deltas(ticker, bid_deltas, ask_deltas, now, received_at)
            updated.update(affected)
        if updated:
            self._update_queue.put_nowait(updated)

    def _update_books_from_yes_side(
        self,
        ticker: str,
        yes_bids: list[OrderBookLevel],
        yes_asks: list[OrderBookLevel],
        event: str,
    ) -> set[str]:
        """Construct YES and NO OrderBook objects from the YES bid/ask prices."""
        # NO bids = 1 - YES asks;  NO asks = 1 - YES bids
        no_bids = [
            OrderBookLevel(price=ONE - level.price, size=level.size)
            for level in yes_asks
            if ZERO < level.price < ONE
        ]
        no_asks = [
            OrderBookLevel(price=ONE - level.price, size=level.size)
            for level in yes_bids
            if ZERO < level.price < ONE
        ]
        yes_token = kalshi_token_id(ticker, "yes")
        no_token = kalshi_token_id(ticker, "no")
        now = time.monotonic()
        received_at = datetime.now(UTC)
        updated: set[str] = set()
        for token_id, bids, asks in [
            (yes_token, yes_bids, yes_asks),
            (no_token, no_bids, no_asks),
        ]:
            if not bids and not asks:
                continue
            book = OrderBook(
                asset_id=token_id,
                bids=bids,
                asks=asks,
                timestamp=received_at,
                raw_json={"_audit_source": "kalshi_fix", "_event": event, "_ticker": ticker},
            )
            self._states[token_id] = _BookState(
                book=book,
                updated_at_monotonic=now,
                last_message_received_at=received_at,
                last_latency_ms=0.0,
                last_event_type=event,
            )
            updated.add(token_id)
            self.token_update_count += 1
        return updated

    def _apply_incremental_deltas(
        self,
        ticker: str,
        bid_deltas: list[tuple[str, Decimal, Decimal]],
        ask_deltas: list[tuple[str, Decimal, Decimal]],
        now: float,
        received_at: datetime,
    ) -> set[str]:
        """Apply FIX incremental deltas to the YES and NO books."""
        yes_token = kalshi_token_id(ticker, "yes")
        no_token = kalshi_token_id(ticker, "no")
        yes_state = self._states.get(yes_token)
        no_state = self._states.get(no_token)
        yes_book = (yes_state.book if yes_state else None) or OrderBook(asset_id=yes_token)
        no_book = (no_state.book if no_state else None) or OrderBook(asset_id=no_token)

        updated: set[str] = set()

        # YES bid delta → YES bid, also implies NO ask delta
        for _action, price, size in bid_deltas:
            yes_book.apply_bid_delta(price, size)
            implied_no_ask = ONE - price
            if ZERO < implied_no_ask < ONE:
                no_book.apply_ask_delta(implied_no_ask, size)

        # YES ask delta → YES ask, also implies NO bid delta
        for _action, price, size in ask_deltas:
            yes_book.apply_ask_delta(price, size)
            implied_no_bid = ONE - price
            if ZERO < implied_no_bid < ONE:
                no_book.apply_bid_delta(implied_no_bid, size)

        for token_id, book, state in [
            (yes_token, yes_book, yes_state),
            (no_token, no_book, no_state),
        ]:
            book.raw_json["_audit_source"] = "kalshi_fix"
            book.raw_json["_event"] = "incremental"
            new_state = _BookState(
                book=book,
                updated_at_monotonic=now,
                last_message_received_at=received_at,
                last_latency_ms=0.0,
                last_event_type="incremental",
            )
            self._states[token_id] = new_state
            updated.add(token_id)
            self.token_update_count += 1
        return updated

    # ── FIX message senders ───────────────────────────────────────────────────

    async def _send(self, msg: bytes) -> None:
        if self._writer is not None and not self._writer.is_closing():
            self._writer.write(msg)
            await self._writer.drain()
        self._seq += 1

    async def _send_logon(self) -> None:
        msg = _fix_build(
            _LOGON,
            [
                (_T["EncryptMethod"], b"0"),
                (_T["HeartBtInt"], str(self.heartbeat_interval).encode()),
                (_T["Username"], self._sender),
                (_T["Password"], self.api_secret.encode()),
                (_T["DefaultApplVerID"], self.DEFAULT_APP_VER),
            ],
            self._sender,
            self.TARGET_COMP_ID,
            self._seq,
        )
        await self._send(msg)
        logger.debug("Kalshi FIX LOGON sent (seq=%d)", self._seq - 1)

    async def _send_logout(self, reason: str = "") -> None:
        fields = [(_T["Text"], reason.encode())] if reason else []
        msg = _fix_build(_LOGOUT, fields, self._sender, self.TARGET_COMP_ID, self._seq)
        await self._send(msg)

    async def _send_heartbeat(self, test_req_id: bytes | None = None) -> None:
        fields = [(_T["TestReqID"], test_req_id)] if test_req_id else []
        msg = _fix_build(_HEARTBEAT, fields, self._sender, self.TARGET_COMP_ID, self._seq)
        await self._send(msg)

    async def _send_md_request(self, ticker: str) -> None:
        self._req_id_counter += 1
        req_id = f"MDR{self._req_id_counter}".encode()
        msg = _fix_build(
            _MD_REQUEST,
            [
                (_T["MDReqID"], req_id),
                (_T["SubscriptionRequestType"], b"1"),   # Snapshot + Updates
                (_T["MarketDepth"], b"0"),               # Full book
                (_T["MDUpdateType"], b"1"),              # Incremental refresh
                (_T["NoMDEntryTypes"], b"2"),
                (_T["MDEntryType"], _BID_TYPE),
                (_T["MDEntryType"], _OFFER_TYPE),
                (_T["NoRelatedSym"], b"1"),
                (_T["Symbol"], ticker.encode()),
            ],
            self._sender,
            self.TARGET_COMP_ID,
            self._seq,
        )
        await self._send(msg)
        logger.debug("Kalshi FIX subscribed to %s (req=%s)", ticker, req_id)

    async def _heartbeat_loop(self) -> None:
        while not self._stop_requested and self.connected:
            await asyncio.sleep(self.heartbeat_interval)
            if self.connected and self._writer is not None:
                try:
                    await self._send_heartbeat()
                except Exception as exc:
                    logger.debug("Heartbeat send failed: %s", exc)
                    break

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _unique_tickers(self, token_ids: set[str]) -> list[str]:
        tickers: dict[str, None] = {}
        for tid in token_ids:
            parsed = parse_kalshi_token_id(tid)
            if parsed is not None:
                tickers[parsed.ticker] = None
        return list(tickers)
