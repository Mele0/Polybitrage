"""Kalshi FIX Order Entry client (port 8228).

Submits Fill-Or-Kill limit orders via the Kalshi FIX 5.0 SP2 (FIXT.1.1) Order Entry
session and receives ExecutionReports. This is 10–50× faster than the REST order
endpoint once the TCP+TLS session is established.

Ports
-----
8228  no-retransmission  (fastest — use this for latency-critical live trading)
8229  with-retransmission (safer for low-volume or unreliable links)

Requirements
------------
A Kalshi FIX Order Entry API key — distinct from both the REST key and the
market-data FIX key. Request via the Kalshi portal.

Configure via env vars (or pass directly):
  KALSHI_FIX_OE_API_KEY    — SenderCompID / username
  KALSHI_FIX_OE_API_SECRET — password

Usage
-----
    client = KalshiFixOrderEntryClient(
        api_key=os.environ["KALSHI_FIX_OE_API_KEY"],
        api_secret=os.environ["KALSHI_FIX_OE_API_SECRET"],
    )
    async with client:
        result = await client.submit_fok_order("KXBTCD-25DEC31-T50000", "buy", 0.52, 10.0)
        if result.success:
            print(f"Filled at avg {result.avg_price:.4f}, qty {result.filled_qty}")
"""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.kalshi.fix_client import (
    SOH,
    _HEARTBEAT,
    _LOGON,
    _LOGOUT,
    _TEST_REQUEST,
    _T,
    _fix_build,
    _fix_now,
    _fix_parse,
)

logger = logging.getLogger(__name__)

# ── FIX Order Entry message types ─────────────────────────────────────────────
_NEW_ORDER_SINGLE = b"D"
_ORDER_CANCEL_REQUEST = b"F"
_EXECUTION_REPORT = b"8"

# ── FIX field tags for order entry ────────────────────────────────────────────
_OE_TAGS = {
    "ClOrdID": 11,
    "HandlInst": 21,
    "OrderQty": 38,
    "OrdType": 40,
    "Price": 44,
    "Side": 54,
    "Symbol": 55,
    "TimeInForce": 59,
    "TransactTime": 60,
    "OrderID": 37,
    "OrdStatus": 39,
    "ExecType": 150,
    "CumQty": 14,
    "AvgPx": 6,
    "LeavesQty": 151,
    "Text": 58,
    "ExecID": 17,
    "LastPx": 31,
    "LastQty": 32,
}

# OrdStatus / ExecType values
_STATUS_NEW = b"0"
_STATUS_PARTIAL = b"1"
_STATUS_FILLED = b"2"
_STATUS_CANCELLED = b"4"
_STATUS_REJECTED = b"8"

# Side
_SIDE_BUY = b"1"
_SIDE_SELL = b"2"


@dataclass
class FixOrderResult:
    success: bool
    filled_qty: float = 0.0
    avg_price: float = 0.0
    order_id: str = ""
    error: str | None = None
    submit_ns: int | None = None
    ack_ns: int | None = None

    @property
    def round_trip_ms(self) -> float | None:
        if self.submit_ns is None or self.ack_ns is None:
            return None
        return (self.ack_ns - self.submit_ns) / 1_000_000


class KalshiFixOrderEntryClient:
    """Kalshi FIX 5.0 SP2 Order Entry session (port 8228).

    Maintains a persistent FIX session and correlates order responses via
    asyncio.Future objects keyed by ClOrdID.  Submit an order; the coroutine
    suspends until the ExecutionReport arrives or the timeout fires.
    """

    HOST = "mm.fix.elections.kalshi.com"
    PORT_FAST = 8228
    PORT_RELIABLE = 8229
    TARGET_COMP_ID = b"KALSHI"
    DEFAULT_APP_VER = b"9"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        host: str = HOST,
        port: int = PORT_FAST,
        heartbeat_interval: int = 30,
        reconnect_delay_seconds: float = 5.0,
    ) -> None:
        self.api_key = (api_key or os.environ.get("KALSHI_FIX_OE_API_KEY", "")).strip()
        self.api_secret = (api_secret or os.environ.get("KALSHI_FIX_OE_API_SECRET", "")).strip()
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Kalshi FIX Order Entry requires KALSHI_FIX_OE_API_KEY and "
                "KALSHI_FIX_OE_API_SECRET. Set via env vars or pass to the constructor."
            )
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_delay_seconds = reconnect_delay_seconds

        self._sender = self.api_key.encode()
        self._seq = 1
        self._clord_counter = 0

        self.connected = False
        self.reconnect_count = 0

        self._writer: asyncio.StreamWriter | None = None
        self._runner_task: asyncio.Task | None = None
        self._stop_requested = False

        # ClOrdID → asyncio.Future[FixOrderResult]
        self._pending: dict[str, asyncio.Future[FixOrderResult]] = {}

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "KalshiFixOrderEntryClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def connect(self) -> None:
        self._stop_requested = False
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self._run())
        # Wait for LOGON to complete (up to 10s)
        deadline = time.monotonic() + 10.0
        while not self.connected and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if not self.connected:
            raise ConnectionError("Kalshi FIX Order Entry: LOGON not confirmed within 10s")

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
        # Cancel all pending order futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(FixOrderResult(success=False, error="client closed"))
        self._pending.clear()

    # ── Order submission ──────────────────────────────────────────────────────

    async def submit_fok_order(
        self,
        ticker: str,
        side: str,
        price: float,
        qty: float,
        *,
        timeout: float = 5.0,
    ) -> FixOrderResult:
        """Submit a Fill-Or-Kill limit order and await the ExecutionReport.

        Parameters
        ----------
        ticker : str
            Kalshi market ticker, e.g. ``KXBTCD-25DEC31-T50000``.
        side : str
            ``"buy"`` or ``"sell"``.
        price : float
            Limit price (0 < price < 1).
        qty : float
            Number of contracts.
        timeout : float
            Seconds to wait for ExecutionReport before treating as timeout error.
        """
        if not self.connected:
            return FixOrderResult(success=False, error="not connected to FIX order entry session")

        self._clord_counter += 1
        clord_id = f"C{int(time.time_ns() // 1_000_000)}{self._clord_counter}"
        fix_side = _SIDE_BUY if side.lower() == "buy" else _SIDE_SELL

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[FixOrderResult] = loop.create_future()
        self._pending[clord_id] = fut

        submit_ns = time.perf_counter_ns()
        try:
            await self._send_new_order(clord_id, ticker.encode(), fix_side, price, qty)
            result = await asyncio.wait_for(fut, timeout=timeout)
            result.submit_ns = submit_ns
            return result
        except TimeoutError:
            return FixOrderResult(success=False, error=f"order timeout after {timeout}s", submit_ns=submit_ns)
        finally:
            self._pending.pop(clord_id, None)

    # ── FIX session loop ──────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stop_requested:
            try:
                ssl_ctx = ssl.create_default_context()
                reader, writer = await asyncio.open_connection(self.host, self.port, ssl=ssl_ctx)
                self._writer = writer
                self._seq = 1
                await self._send_logon()
                hb_task = asyncio.create_task(self._heartbeat_loop())
                try:
                    await self._read_loop(reader)
                finally:
                    hb_task.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop_requested:
                    break
                self.connected = False
                self.reconnect_count += 1
                logger.warning("Kalshi FIX OE disconnected: %s. Reconnecting in %.1fs.", exc, self.reconnect_delay_seconds)
                await asyncio.sleep(self.reconnect_delay_seconds)
            finally:
                self.connected = False
                self._writer = None

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        buf = b""
        while not self._stop_requested:
            chunk = await reader.read(65536)
            if not chunk:
                raise ConnectionResetError("FIX OE session closed by server")
            buf += chunk
            while True:
                end = buf.find(b"10=")
                if end == -1:
                    break
                eom = buf.find(SOH, end)
                if eom == -1:
                    break
                self._dispatch(buf[: eom + 1])
                buf = buf[eom + 1 :]

    def _dispatch(self, raw: bytes) -> None:
        fields = _fix_parse(raw)
        msg_type = fields.get(_T["MsgType"], b"")
        if msg_type == _HEARTBEAT:
            return
        if msg_type == _TEST_REQUEST:
            asyncio.create_task(self._send_heartbeat(fields.get(_T["TestReqID"], b"")))
        elif msg_type == _LOGON:
            self.connected = True
            logger.info("Kalshi FIX OE session established (seq=%d)", self._seq)
        elif msg_type == _LOGOUT:
            text = fields.get(_T["Text"], b"").decode(errors="replace")
            self.connected = False
            raise ConnectionResetError(f"FIX OE LOGOUT: {text}")
        elif msg_type == _EXECUTION_REPORT:
            self._handle_execution_report(fields)

    def _handle_execution_report(self, fields: dict[int, bytes]) -> None:
        clord_id = fields.get(_OE_TAGS["ClOrdID"], b"").decode(errors="replace")
        fut = self._pending.get(clord_id)
        if fut is None or fut.done():
            return

        ack_ns = time.perf_counter_ns()
        ord_status = fields.get(_OE_TAGS["OrdStatus"], b"")
        exec_type = fields.get(_OE_TAGS["ExecType"], b"")
        text = fields.get(_OE_TAGS["Text"], b"").decode(errors="replace")
        order_id = fields.get(_OE_TAGS["OrderID"], b"").decode(errors="replace")

        try:
            filled_qty = float(fields.get(_OE_TAGS["CumQty"], b"0"))
            avg_price = float(fields.get(_OE_TAGS["AvgPx"], b"0"))
        except (ValueError, TypeError):
            filled_qty = 0.0
            avg_price = 0.0

        if exec_type in (_STATUS_FILLED, _STATUS_PARTIAL) or ord_status == _STATUS_FILLED:
            result = FixOrderResult(
                success=True,
                filled_qty=filled_qty,
                avg_price=avg_price,
                order_id=order_id,
                ack_ns=ack_ns,
            )
        elif ord_status in (_STATUS_REJECTED, _STATUS_CANCELLED):
            result = FixOrderResult(
                success=False,
                filled_qty=filled_qty,
                avg_price=avg_price,
                order_id=order_id,
                error=text or f"OrdStatus={ord_status.decode()}",
                ack_ns=ack_ns,
            )
        else:
            # Intermediate status (new, pending) — keep waiting
            return

        fut.set_result(result)

    # ── FIX senders ───────────────────────────────────────────────────────────

    async def _send(self, msg: bytes) -> None:
        if self._writer is not None and not self._writer.is_closing():
            self._writer.write(msg)
            await self._writer.drain()
        self._seq += 1

    async def _send_logon(self) -> None:
        await self._send(_fix_build(
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
        ))

    async def _send_logout(self, reason: str = "") -> None:
        fields = [(_T["Text"], reason.encode())] if reason else []
        await self._send(_fix_build(_LOGOUT, fields, self._sender, self.TARGET_COMP_ID, self._seq))

    async def _send_heartbeat(self, test_req_id: bytes | None = None) -> None:
        fields = [(_T["TestReqID"], test_req_id)] if test_req_id else []
        await self._send(_fix_build(_HEARTBEAT, fields, self._sender, self.TARGET_COMP_ID, self._seq))

    async def _send_new_order(
        self,
        clord_id: str,
        symbol: bytes,
        side: bytes,
        price: float,
        qty: float,
    ) -> None:
        await self._send(_fix_build(
            _NEW_ORDER_SINGLE,
            [
                (_OE_TAGS["ClOrdID"], clord_id.encode()),
                (_OE_TAGS["HandlInst"], b"1"),
                (_OE_TAGS["Symbol"], symbol),
                (_OE_TAGS["Side"], side),
                (_OE_TAGS["TransactTime"], _fix_now()),
                (_OE_TAGS["OrderQty"], f"{qty:.2f}".encode()),
                (_OE_TAGS["OrdType"], b"2"),          # Limit
                (_OE_TAGS["Price"], f"{price:.4f}".encode()),
                (_OE_TAGS["TimeInForce"], b"4"),       # Fill Or Kill
            ],
            self._sender,
            self.TARGET_COMP_ID,
            self._seq,
        ))

    async def _heartbeat_loop(self) -> None:
        while not self._stop_requested and self.connected:
            await asyncio.sleep(self.heartbeat_interval)
            if self.connected and self._writer is not None:
                try:
                    await self._send_heartbeat()
                except Exception:
                    break
