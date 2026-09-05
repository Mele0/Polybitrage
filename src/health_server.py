"""Lightweight HTTP health-check endpoint for the arb bot process.

Opens a TCP server on ``host:port`` (default ``127.0.0.1:8765``) that handles
one HTTP/1.1 route:

    GET /health   → 200 JSON with process vitals
    GET /ready    → 200 if WS connected, 503 otherwise (for liveness probes)
    *             → 404

No external dependencies — uses only ``asyncio.start_server``.

Usage (from the simulator main loop)::

    server = HealthServer(simulator)
    await server.start()
    ...
    await server.stop()

The ``HealthServer`` queries the ``PaperArbSimulator`` instance for vitals at
request time, so the JSON is always fresh.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.simulator import PaperArbSimulator

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765


class HealthServer:
    """Minimal asyncio HTTP server exposing bot vitals at ``/health``.

    Parameters
    ----------
    simulator:
        Live ``PaperArbSimulator`` instance to query for vitals.
    host / port:
        Bind address.  Defaults to localhost-only (127.0.0.1:8765).
        Change ``host`` to ``"0.0.0.0"`` to expose to the network (only do
        this behind a firewall / private VPC).
    """

    def __init__(
        self,
        simulator: "PaperArbSimulator",
        *,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self._sim = simulator
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._started_at = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bind the TCP server and begin accepting connections."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )
        addr = self._server.sockets[0].getsockname() if self._server.sockets else (self._host, self._port)
        import logging
        logging.getLogger(__name__).info(
            "Health server listening on http://%s:%s/health", addr[0], addr[1]
        )

    async def stop(self) -> None:
        """Close the server gracefully."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ── Request handling ──────────────────────────────────────────────────────

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.read(512), timeout=2.0)
        except (asyncio.TimeoutError, ConnectionError):
            writer.close()
            return

        path = _parse_request_path(raw)
        content_type = "application/json"
        if path == "/metrics":
            # Prometheus scrape target.  Cheap: reads the latest ScanRow + the
            # profiler's cached snapshot, no numpy on the event loop.
            from src.metrics import render_prometheus  # noqa: PLC0415
            try:
                body = render_prometheus(self._sim, self._started_at).encode()
            except Exception as exc:  # never let a scrape crash the loop
                body = f"# render error: {exc}\n".encode()
            content_type = "text/plain; version=0.0.4; charset=utf-8"
            status = "200 OK"
        elif path == "/health":
            body = json.dumps(self._vitals(), separators=(",", ":")).encode()
            status = "200 OK"
        elif path == "/ready":
            vitals = self._vitals()
            if vitals.get("ws_connected"):
                body = b'{"ready":true}'
                status = "200 OK"
            else:
                body = b'{"ready":false,"reason":"ws_not_connected"}'
                status = "503 Service Unavailable"
        else:
            body = b'{"error":"not found"}'
            status = "404 Not Found"

        response = (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body

        try:
            writer.write(response)
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()

    def _vitals(self) -> dict[str, Any]:
        sim = self._sim
        provider = getattr(sim, "_provider", None)
        ws_client = getattr(provider, "ws_client", None)

        ws_connected = bool(getattr(ws_client, "connected", False))
        reconnect_count = int(getattr(ws_client, "reconnect_count", 0))
        token_update_count = int(getattr(ws_client, "token_update_count", 0))

        # Cash / portfolio
        cash = float(getattr(sim, "cash", 0.0))
        positions = getattr(sim, "positions", [])
        open_positions = sum(1 for p in positions if not getattr(p, "closed", True))

        # Last trade time
        last_trade_times = getattr(sim, "last_trade_time_by_pair", {})
        last_trade_ts: float | None = max(last_trade_times.values()) if last_trade_times else None

        uptime_s = time.time() - self._started_at

        return {
            "ok": True,
            "uptime_s": round(uptime_s, 1),
            "ws_connected": ws_connected,
            "ws_reconnect_count": reconnect_count,
            "ws_token_updates": token_update_count,
            "cash_usd": round(cash, 4),
            "open_positions": open_positions,
            "last_trade_epoch_s": last_trade_ts,
            "ts": time.time(),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_request_path(raw: bytes) -> str:
    """Extract the request path from the first line of an HTTP request."""
    try:
        first_line = raw.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
        parts = first_line.split()
        if len(parts) >= 2:
            # Strip query string
            return parts[1].split("?", 1)[0]
    except Exception:
        pass
    return "/"
