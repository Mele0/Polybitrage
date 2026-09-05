"""Active network + clock-skew profiler for the arb bot (background, off-path).

Passive ``wire_latency`` (exchange timestamp → receipt) can't tell you *why* it's
high — network, or a skewed local clock, or stale market timestamps.  This module
adds the layered active probing HFT setups rely on, on a slow background cadence:

  * **TCP connect RTT** to the CLOB / Gamma hosts (transport-level).
  * **HTTPS GET RTT** to a cheap endpoint (application-level).
  * **Clock skew** vs the exchange: ``GET clob.polymarket.com/time`` returns the
    server's epoch seconds; comparing to the local clock (corrected for RTT/2)
    tells you whether ``wire_latency`` is real network delay or just a clock that
    drifted.  This directly disambiguates the observed multi-second wire latency.

Runs every ``interval_s`` (default 15s) on a background task and caches the
result — the ``/metrics`` scrape just reads the cached dict.
"""
from __future__ import annotations

import asyncio
import ssl
import time
from typing import Any

# (label, host, port, path-for-GET)
_TARGETS: tuple[tuple[str, str, int, str], ...] = (
    ("clob", "clob.polymarket.com", 443, "/time"),
    ("gamma", "gamma-api.polymarket.com", 443, "/markets?limit=1"),
)

_LATEST: dict[str, float] = {}
_SSL = ssl.create_default_context()


async def _tcp_connect_ms(host: str, port: int) -> float:
    t0 = time.perf_counter()
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=_SSL), timeout=5.0)
        dt = (time.perf_counter() - t0) * 1000
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
        except Exception:  # noqa: BLE001
            pass
        return dt
    except Exception:  # noqa: BLE001
        return float("nan")


async def _https_get(host: str, port: int, path: str) -> tuple[float, bytes]:
    t0 = time.perf_counter()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=_SSL), timeout=5.0)
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
               "User-Agent: arb-net-probe/1.0\r\nConnection: close\r\n\r\n").encode()
        writer.write(req)
        await writer.drain()
        data = b""
        while len(data) < 65536:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            if not chunk:
                break
            data += chunk
        dt = (time.perf_counter() - t0) * 1000
        writer.close()
        return dt, data
    except Exception:  # noqa: BLE001
        return float("nan"), b""


def _parse_clock_skew(body: bytes, rtt_ms: float, local_mid_epoch: float) -> float | None:
    """clob /time returns server epoch seconds (plain text after headers).

    Returns local-minus-server skew in ms (positive ⇒ local clock ahead).
    """
    try:
        sep = body.find(b"\r\n\r\n")
        payload = body[sep + 4:] if sep != -1 else body
        text = payload.decode("ascii", "ignore").strip().strip('"')
        server_epoch = float(text.split()[0]) if text else None
        if server_epoch is None or server_epoch <= 0:
            return None
        if server_epoch > 1e12:  # ms not s
            server_epoch /= 1000.0
        return round((local_mid_epoch - server_epoch) * 1000.0, 1)
    except Exception:  # noqa: BLE001
        return None


async def _probe_once() -> dict[str, float]:
    out: dict[str, float] = {}
    for label, host, port, path in _TARGETS:
        tcp = await _tcp_connect_ms(host, port)
        if tcp == tcp:  # not NaN
            out[f"{label}_tcp_connect_ms"] = round(tcp, 2)
        t0 = time.time()
        lat, body = await _https_get(host, port, path)
        t1 = time.time()
        if lat == lat:
            out[f"{label}_https_get_ms"] = round(lat, 2)
        out[f"{label}_reachable"] = 1.0 if lat == lat else 0.0
        if label == "clob" and body:
            skew = _parse_clock_skew(body, lat, (t0 + t1) / 2.0)
            if skew is not None:
                out["clock_skew_ms"] = skew
    return out


async def prober(interval_s: float = 15.0) -> None:
    """Background task: refresh the cached network/clock snapshot.

    A handful of short-lived TLS connections every ``interval_s`` — negligible and
    fully off the scan path.  Cancelled on shutdown.
    """
    global _LATEST
    while True:
        try:
            _LATEST = await _probe_once()
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(max(1.0, interval_s))


def snapshot() -> dict[str, float]:
    return _LATEST
