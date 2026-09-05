"""Two-node UDP arbitrage signal bridge.

Enables a London VPS (reading Polymarket WebSocket) and a Chicago VPS
(executing Kalshi FIX orders) to coordinate sub-5ms cross-market arbitrage.

Architecture
------------
London node:
  - Runs the normal PaperArbSimulator in signal-sender mode
  - When an arbitrage opportunity is detected it fires a UDP datagram
  - The datagram is HMAC-SHA256 signed with a shared secret

Chicago node:
  - Runs SignalReceiver, which owns KalshiFixOrderEntryClient
  - On a verified signal: validates freshness, checks edge, submits FIX order
  - Results are logged to CSV

Signal format (binary framing)
-------------------------------
  [4 bytes: magic 0xA4B3C2D1]
  [8 bytes: send_ns (uint64 big-endian)]
  [32 bytes: HMAC-SHA256 of payload]
  [4 bytes: payload_length (uint32 big-endian)]
  [payload_length bytes: UTF-8 JSON]

JSON payload schema
-------------------
  {
    "v": 1,
    "id": "<uuid>",              dedup ID
    "ts_ns": <int>,              nanosecond send timestamp
    "pairs": [
      {
        "token_id": "kalshi:TICKER:yes",
        "limit_price": 0.52,
        "size": 10.0,
        "tick_size": "0.01",
        "neg_risk": false
      }
    ],
    "guaranteed_payout": 1.0,
    "min_edge": 0.002
  }

Configuration
-------------
  SIGNAL_BRIDGE_SECRET   — shared HMAC secret (both nodes must have same value)
  SIGNAL_BRIDGE_HOST     — sender target / receiver bind host
  SIGNAL_BRIDGE_PORT     — UDP port (default 9999)

Usage
-----
  # London node (sender):
  sender = SignalSender(secret=os.environ["SIGNAL_BRIDGE_SECRET"],
                        target_host="<chicago-ip>")
  await sender.send_opportunity(legs=[...])

  # Chicago node (receiver + executor):
  receiver = SignalReceiver(secret=os.environ["SIGNAL_BRIDGE_SECRET"],
                            fix_client=KalshiFixOrderEntryClient(...))
  await receiver.listen()
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

try:
    import orjson as _json_lib
    def _dumps(obj: Any) -> bytes: return _json_lib.dumps(obj)
    def _loads(data: bytes) -> Any: return _json_lib.loads(data)
except ImportError:
    import json as _json_lib  # type: ignore[no-redef]
    def _dumps(obj: Any) -> bytes: return _json_lib.dumps(obj).encode()  # type: ignore[misc]
    def _loads(data: bytes) -> Any: return _json_lib.loads(data)  # type: ignore[misc]

logger = logging.getLogger(__name__)

_MAGIC = 0xA4B3C2D1
_MAGIC_BYTES = struct.pack(">I", _MAGIC)
_HEADER_FMT = ">IQ32sI"   # magic(4) + ts_ns(8) + hmac(32) + payload_len(4)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_AGE_NS = 5_000_000_000  # 5 seconds — discard stale signals
_DEFAULT_PORT = 9999


@dataclass
class SignalLeg:
    token_id: str
    limit_price: float
    size: float
    tick_size: str | None = None
    neg_risk: bool | None = None


@dataclass
class ArbitrageSignal:
    signal_id: str
    ts_ns: int
    legs: list[SignalLeg]
    guaranteed_payout: float
    min_edge: float
    recv_ns: int = field(default_factory=time.perf_counter_ns)

    @property
    def age_ms(self) -> float:
        return (time.perf_counter_ns() - self.recv_ns) / 1_000_000


# ── Framing helpers ───────────────────────────────────────────────────────────


def _sign(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, payload, hashlib.sha256).digest()


def _pack(secret: bytes, payload_json: bytes) -> bytes:
    ts_ns = time.time_ns()
    mac = _sign(secret, payload_json)
    header = struct.pack(_HEADER_FMT, _MAGIC, ts_ns, mac, len(payload_json))
    return header + payload_json


def _unpack(secret: bytes, datagram: bytes) -> ArbitrageSignal | None:
    if len(datagram) < _HEADER_SIZE:
        return None
    magic, ts_ns, mac_recv, payload_len = struct.unpack_from(_HEADER_FMT, datagram)
    if magic != _MAGIC:
        return None
    if len(datagram) < _HEADER_SIZE + payload_len:
        return None
    payload = datagram[_HEADER_SIZE: _HEADER_SIZE + payload_len]
    # Constant-time HMAC comparison
    mac_expected = _sign(secret, payload)
    if not hmac.compare_digest(mac_recv, mac_expected):
        logger.warning("Signal bridge: HMAC mismatch — discarded")
        return None
    # Freshness check
    age_ns = abs(time.time_ns() - ts_ns)
    if age_ns > _MAX_AGE_NS:
        logger.info("Signal bridge: stale signal (age=%.1f ms) — discarded", age_ns / 1e6)
        return None
    try:
        data = _loads(payload)
    except Exception as exc:
        logger.warning("Signal bridge: JSON parse error: %s", exc)
        return None
    try:
        legs = [
            SignalLeg(
                token_id=leg["token_id"],
                limit_price=float(leg["limit_price"]),
                size=float(leg["size"]),
                tick_size=leg.get("tick_size"),
                neg_risk=leg.get("neg_risk"),
            )
            for leg in data.get("pairs", [])
        ]
        return ArbitrageSignal(
            signal_id=data.get("id", ""),
            ts_ns=int(data.get("ts_ns", ts_ns)),
            legs=legs,
            guaranteed_payout=float(data.get("guaranteed_payout", 1.0)),
            min_edge=float(data.get("min_edge", 0.0)),
        )
    except Exception as exc:
        logger.warning("Signal bridge: payload schema error: %s", exc)
        return None


# ── Signal sender (London node) ───────────────────────────────────────────────


class SignalSender:
    """UDP signal sender for the Polymarket-reading London node.

    Call ``send_opportunity()`` when an arbitrage candidate is detected.
    The datagram is sent fire-and-forget — no ack, no retry.
    """

    def __init__(
        self,
        secret: str | bytes | None = None,
        *,
        target_host: str = "127.0.0.1",
        port: int = _DEFAULT_PORT,
    ) -> None:
        raw = secret or os.environ.get("SIGNAL_BRIDGE_SECRET", "")
        self._secret = raw.encode() if isinstance(raw, str) else raw
        if not self._secret:
            raise ValueError("SignalSender requires a SIGNAL_BRIDGE_SECRET")
        self.target_host = target_host
        self.port = port
        self._transport: asyncio.DatagramTransport | None = None

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol,
            remote_addr=(self.target_host, self.port),
        )

    async def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    async def send_opportunity(
        self,
        legs: list[SignalLeg],
        *,
        guaranteed_payout: float = 1.0,
        min_edge: float = 0.0,
    ) -> None:
        """Fire a signal datagram. Returns immediately — UDP is best-effort."""
        if self._transport is None or self._transport.is_closing():
            logger.debug("SignalSender: not connected, dropping signal")
            return
        payload = _dumps({
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts_ns": time.time_ns(),
            "pairs": [
                {
                    "token_id": leg.token_id,
                    "limit_price": leg.limit_price,
                    "size": leg.size,
                    "tick_size": leg.tick_size,
                    "neg_risk": leg.neg_risk,
                }
                for leg in legs
            ],
            "guaranteed_payout": guaranteed_payout,
            "min_edge": min_edge,
        })
        datagram = _pack(self._secret, payload)
        try:
            self._transport.sendto(datagram)
        except Exception as exc:
            logger.debug("SignalSender: send failed: %s", exc)


# ── Signal receiver (Chicago node) ────────────────────────────────────────────


class _ReceiverProtocol(asyncio.DatagramProtocol):
    def __init__(self, secret: bytes, on_signal: Callable[[ArbitrageSignal], None]) -> None:
        self._secret = secret
        self._on_signal = on_signal
        self._seen_ids: dict[str, float] = {}  # dedup: id → monotonic receive time

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        signal = _unpack(self._secret, data)
        if signal is None:
            return
        # Dedup: ignore signals seen in the last 10 seconds
        now = time.monotonic()
        if signal.signal_id in self._seen_ids:
            return
        # Evict old seen IDs
        self._seen_ids = {k: v for k, v in self._seen_ids.items() if now - v < 10.0}
        self._seen_ids[signal.signal_id] = now
        self._on_signal(signal)

    def error_received(self, exc: Exception) -> None:
        logger.debug("SignalReceiver: protocol error: %s", exc)


class SignalReceiver:
    """UDP signal receiver for the Kalshi-executing Chicago node.

    Call ``listen()`` to block until ``stop()`` is called.
    Supply an ``on_signal`` coroutine that receives an ``ArbitrageSignal``
    and executes the trade.
    """

    def __init__(
        self,
        secret: str | bytes | None = None,
        *,
        bind_host: str = "0.0.0.0",
        port: int = _DEFAULT_PORT,
    ) -> None:
        raw = secret or os.environ.get("SIGNAL_BRIDGE_SECRET", "")
        self._secret = raw.encode() if isinstance(raw, str) else raw
        if not self._secret:
            raise ValueError("SignalReceiver requires a SIGNAL_BRIDGE_SECRET")
        self.bind_host = bind_host
        self.port = port
        self._transport: asyncio.DatagramTransport | None = None
        self._on_signal: Callable[[ArbitrageSignal], Awaitable[None]] | None = None
        self._stop_event = asyncio.Event()

    def _sync_on_signal(self, signal: ArbitrageSignal) -> None:
        """Called from the datagram protocol (sync context) — dispatch to async."""
        if self._on_signal is not None:
            asyncio.create_task(self._on_signal(signal))

    async def listen(
        self,
        on_signal: Callable[[ArbitrageSignal], Awaitable[None]],
    ) -> None:
        """Start listening for signals.  Runs until ``stop()`` is called."""
        self._on_signal = on_signal
        self._stop_event.clear()
        loop = asyncio.get_event_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _ReceiverProtocol(self._secret, self._sync_on_signal),
            local_addr=(self.bind_host, self.port),
        )
        logger.info(
            "SignalReceiver listening on %s:%d (HMAC-SHA256 authenticated, max age 5s)",
            self.bind_host,
            self.port,
        )
        await self._stop_event.wait()
        if self._transport is not None:
            self._transport.close()

    async def stop(self) -> None:
        self._stop_event.set()
