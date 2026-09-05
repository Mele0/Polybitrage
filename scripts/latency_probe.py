#!/usr/bin/env python3
"""Network latency probe — measure RTT to Polymarket and Kalshi from this machine.

Run this before and after changing VPS region to quantify co-location gains.

Usage
-----
  python scripts/latency_probe.py                  # full probe, 20 samples each
  python scripts/latency_probe.py --samples 50     # more samples for p99
  python scripts/latency_probe.py --target kalshi  # only probe Kalshi

Output
------
  Endpoint                       p50      p95      p99    min    max   unit
  ─────────────────────────────────────────────────────────────────────────
  Polymarket CLOB (TCP)         6.2 ms   8.1 ms  11.3ms  5.8   14.2   ms
  Polymarket CLOB (HTTPS GET)  62.3 ms  71.0 ms  89.0ms  58.1  94.3   ms
  Kalshi REST (TCP)            28.1 ms  31.4 ms  38.7ms  27.2  45.0   ms
  Kalshi REST (HTTPS GET)      89.4 ms 101.3 ms 120.0ms  85.0 135.0   ms
  Kalshi FIX host (TCP)        29.0 ms  33.0 ms  40.0ms  27.8  47.0   ms
"""
from __future__ import annotations

import argparse
import asyncio
import ssl
import statistics
import sys
import time
from dataclasses import dataclass


@dataclass
class ProbeTarget:
    label: str
    host: str
    port: int
    path: str | None = None  # if set, also measure HTTP GET latency


TARGETS = [
    ProbeTarget("Polymarket CLOB", "clob.polymarket.com", 443, "/time"),
    ProbeTarget("Kalshi REST", "external-api.kalshi.com", 443, "/trade-api/v2/exchange/status"),
    ProbeTarget("Kalshi FIX Market Data", "mm.fix.elections.kalshi.com", 8233),
    ProbeTarget("Kalshi FIX Order Entry", "mm.fix.elections.kalshi.com", 8228),
]


async def _tcp_connect_ms(host: str, port: int, ssl_ctx: ssl.SSLContext | None = None) -> float:
    t0 = time.perf_counter()
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=5.0,
        )
        t1 = time.perf_counter()
        writer.close()
        return (t1 - t0) * 1000
    except Exception:
        return float("nan")


async def _http_get_ms(host: str, port: int, path: str, ssl_ctx: ssl.SSLContext) -> float:
    t0 = time.perf_counter()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=5.0,
        )
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "User-Agent: latency-probe/1.0\r\n"
            "Connection: close\r\n\r\n"
        ).encode()
        writer.write(request)
        await writer.drain()
        # Read until headers end
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            if not chunk:
                break
            data += chunk
        t1 = time.perf_counter()
        writer.close()
        return (t1 - t0) * 1000
    except Exception:
        return float("nan")


def _percentile(data: list[float], pct: float) -> float:
    clean = [x for x in data if not (x != x)]  # remove NaN
    if not clean:
        return float("nan")
    clean.sort()
    idx = (len(clean) - 1) * pct / 100
    lo = int(idx)
    hi = min(lo + 1, len(clean) - 1)
    return clean[lo] + (idx - lo) * (clean[hi] - clean[lo])


def _fmt(ms: float) -> str:
    return f"{ms:6.1f}" if ms == ms else "   N/A"  # noqa: PLR0124 — NaN check


async def probe_target(target: ProbeTarget, samples: int, verbose: bool) -> None:
    ssl_ctx = ssl.create_default_context()
    label_w = 38

    # TCP connection latency
    tcp_samples = []
    for i in range(samples):
        ms = await _tcp_connect_ms(target.host, target.port, ssl_ctx if target.port == 443 or target.port == 8228 or target.port == 8233 else None)
        tcp_samples.append(ms)
        if verbose:
            sys.stdout.write(f"\r  TCP {i+1}/{samples}  {ms:.1f}ms   ")
            sys.stdout.flush()
    if verbose:
        print()

    label = f"{target.label} (TCP)"
    p50 = _percentile(tcp_samples, 50)
    p95 = _percentile(tcp_samples, 95)
    p99 = _percentile(tcp_samples, 99)
    mn = min((x for x in tcp_samples if x == x), default=float("nan"))
    mx = max((x for x in tcp_samples if x == x), default=float("nan"))
    print(f"  {label:<{label_w}} p50={_fmt(p50)}  p95={_fmt(p95)}  p99={_fmt(p99)}  min={_fmt(mn)}  max={_fmt(mx)} ms")

    # HTTP GET latency (if path specified)
    if target.path:
        http_samples = []
        for i in range(samples):
            ms = await _http_get_ms(target.host, target.port, target.path, ssl_ctx)
            http_samples.append(ms)
            if verbose:
                sys.stdout.write(f"\r  HTTP {i+1}/{samples}  {ms:.1f}ms   ")
                sys.stdout.flush()
            await asyncio.sleep(0.05)  # rate-limit HTTP probes
        if verbose:
            print()

        label = f"{target.label} (HTTPS GET)"
        p50 = _percentile(http_samples, 50)
        p95 = _percentile(http_samples, 95)
        p99 = _percentile(http_samples, 99)
        mn = min((x for x in http_samples if x == x), default=float("nan"))
        mx = max((x for x in http_samples if x == x), default=float("nan"))
        print(f"  {label:<{label_w}} p50={_fmt(p50)}  p95={_fmt(p95)}  p99={_fmt(p99)}  min={_fmt(mn)}  max={_fmt(mx)} ms")


async def main(args: argparse.Namespace) -> None:
    targets = TARGETS
    if args.target:
        t_lc = args.target.lower()
        targets = [t for t in targets if t_lc in t.label.lower()]
        if not targets:
            print(f"No target matching {args.target!r}. Available: {[t.label for t in TARGETS]}")
            return

    print(f"\nLatency probe — {args.samples} samples per endpoint\n")
    print(f"  {'Endpoint':<38} {'p50':>8} {'p95':>8} {'p99':>8} {'min':>7} {'max':>7}")
    print(f"  {'─'*38} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
    for target in targets:
        await probe_target(target, args.samples, args.verbose)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure network latency to trading endpoints.")
    parser.add_argument("--samples", type=int, default=20, help="Number of round-trips per endpoint.")
    parser.add_argument("--target", default="", help="Filter endpoints by name substring.")
    parser.add_argument("--verbose", action="store_true", help="Print each sample.")
    asyncio.run(main(parser.parse_args()))
