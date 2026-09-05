from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src import profiling

try:
    import orjson as _orjson

    def _resp_json(response: httpx.Response) -> Any:
        return _orjson.loads(response.content)

except ImportError:
    import json as _stdlib_json

    def _resp_json(response: httpx.Response) -> Any:  # type: ignore[misc]
        return _stdlib_json.loads(response.content)

from src.polymarket.models import OrderBook

logger = logging.getLogger(__name__)
BOOKS_BATCH_SIZE = 50


@dataclass
class OrderBookBatchResult:
    books: dict[str, OrderBook] = field(default_factory=dict)
    unavailable_token_ids: list[str] = field(default_factory=list)
    error_by_token_id: dict[str, str] = field(default_factory=dict)


class ClobClient:
    """Small public CLOB client.

    This clean repo intentionally exposes only public data endpoints. It contains no
    private-key handling and no order-placement methods.
    """

    def __init__(
        self,
        base_url: str = "https://clob.polymarket.com",
        timeout_seconds: float = 10.0,
        *,
        audit: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.audit = audit
        self._fee_rate_cache: dict[str, float] = {}
        try:
            import h2  # noqa: F401
            _http2 = True
        except ImportError:
            _http2 = False
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            http2=_http2,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=20,
                keepalive_expiry=30.0,
            ),
            timeout=timeout_seconds,
            headers={"User-Agent": "polybitrage/0.1 research"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ClobClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def warmup(self) -> None:
        """Pre-establish the TLS connection and HTTP/2 session.

        Call once after construction to eliminate cold-start handshake latency
        from the first real request. Sends a lightweight HEAD to /time.
        """
        try:
            await self._client.head("/time")
        except Exception:
            pass

    async def get_order_book(self, token_id: str) -> OrderBook | None:
        try:
            response = await self._request("GET", "/book", params={"token_id": token_id})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            book = OrderBook.from_api(_resp_json(response), token_id=token_id)
            book.raw_json["_audit_source"] = "rest_book"
            return book
        except Exception as exc:  # noqa: BLE001 - a missing book should not kill a live paper scan.
            logger.info("CLOB book unavailable for token %s: %s", token_id, exc)
            return None

    async def get_order_books(
        self,
        token_ids: list[str],
        *,
        max_concurrent_requests: int = 10,
    ) -> OrderBookBatchResult:
        requested = list(dict.fromkeys(str(token_id) for token_id in token_ids if token_id))
        if not requested:
            return OrderBookBatchResult()

        # Time the whole REST book fetch (batch + any fallback) as rest_call_latency.
        # This is the REST leg of every polling fetch, speculative pre-fetch, and
        # entry REST recheck — the slowest controllable hop after the network.
        _prof = profiling.ENABLED and profiling.RECORDER is not None
        _t0 = time.perf_counter_ns() if _prof else 0
        try:
            # Try batch first because it is faster and closer to the old working scanner.
            try:
                books = await self._fetch_book_batches(
                    requested,
                    max_concurrent_requests=max_concurrent_requests,
                )
                missing = [token_id for token_id in requested if token_id not in books]
                if not missing:
                    return OrderBookBatchResult(books=books)
                fallback = await self._fetch_concurrently(missing, max_concurrent_requests=max_concurrent_requests)
                books.update(fallback.books)
                fallback.books = books
                return fallback
            except Exception as exc:  # noqa: BLE001
                logger.info("Batch CLOB book request failed; falling back to per-token fetch: %s", exc)

            return await self._fetch_concurrently(requested, max_concurrent_requests=max_concurrent_requests)
        finally:
            if _prof:
                profiling.RECORDER.record_series("rest_call_latency", time.perf_counter_ns() - _t0)

    async def _fetch_book_batches(self, token_ids: list[str], *, max_concurrent_requests: int) -> dict[str, OrderBook]:
        chunks = [token_ids[index : index + BOOKS_BATCH_SIZE] for index in range(0, len(token_ids), BOOKS_BATCH_SIZE)]
        semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))

        async def fetch_chunk(chunk: list[str]) -> dict[str, OrderBook]:
            wait_started = time.perf_counter_ns()
            await semaphore.acquire()
            semaphore_wait_ms = (time.perf_counter_ns() - wait_started) / 1_000_000
            try:
                response = await self._request(
                    "POST",
                    "/books",
                    json=[{"token_id": token_id} for token_id in chunk],
                    audit_semaphore_wait_ms=semaphore_wait_ms,
                )
                if response.status_code == 404:
                    return {}
                response.raise_for_status()
                payload = _resp_json(response)
                items = payload.get("books") or payload.get("data") if isinstance(payload, dict) else payload
                books: dict[str, OrderBook] = {}
                for item in items or []:
                    if isinstance(item, dict):
                        book = OrderBook.from_api(item)
                        if book.asset_id:
                            book.raw_json["_audit_source"] = "rest_books"
                            books[book.asset_id] = book
                return books
            finally:
                semaphore.release()

        merged: dict[str, OrderBook] = {}
        for batch_books in await asyncio.gather(*(fetch_chunk(chunk) for chunk in chunks)):
            merged.update(batch_books)
        return merged

    async def _fetch_concurrently(self, token_ids: list[str], *, max_concurrent_requests: int) -> OrderBookBatchResult:
        semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))
        result = OrderBookBatchResult()

        async def fetch(token_id: str) -> tuple[str, OrderBook | None]:
            async with semaphore:
                return token_id, await self.get_order_book(token_id)

        for token_id, book in await asyncio.gather(*(fetch(token_id) for token_id in token_ids)):
            if book is None:
                result.unavailable_token_ids.append(token_id)
                result.error_by_token_id[token_id] = "order book unavailable"
            else:
                result.books[token_id] = book
        return result

    async def get_fee_rate(self, token_id: str) -> float | None:
        """Return the token fee rate as a decimal, e.g. 720 bps -> 0.072.

        Polymarket's public fee-rate endpoint returns `base_fee` in basis points.
        This is read-only market metadata and does not require credentials.
        """
        token_id = str(token_id or "")
        if not token_id:
            return None
        if token_id in self._fee_rate_cache:
            return self._fee_rate_cache[token_id]
        try:
            response = await self._request("GET", "/fee-rate", params={"token_id": token_id})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = _resp_json(response)
            base_fee = payload.get("base_fee") if isinstance(payload, dict) else None
            if base_fee is None:
                return None
            rate = float(base_fee) / 10_000.0
            self._fee_rate_cache[token_id] = rate
            return rate
        except Exception as exc:  # noqa: BLE001 - fee metadata should fall back conservatively.
            logger.info("CLOB fee rate unavailable for token %s: %s", token_id, exc)
            return None

    async def get_fee_rates(
        self,
        token_ids: list[str],
        *,
        max_concurrent_requests: int = 10,
    ) -> dict[str, float | None]:
        requested = list(dict.fromkeys(str(token_id) for token_id in token_ids if token_id))
        if not requested:
            return {}
        semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))

        async def fetch(token_id: str) -> tuple[str, float | None]:
            async with semaphore:
                return token_id, await self.get_fee_rate(token_id)

        return dict(await asyncio.gather(*(fetch(token_id) for token_id in requested)))

    async def get_server_time(self, *, audit_source: str = "manual") -> int | None:
        wall_send_ns = time.time_ns()
        perf_send_ns = time.perf_counter_ns()
        response = await self._request(
            "GET",
            "/time",
            audit_wall_start_ns=wall_send_ns,
            audit_perf_start_ns=perf_send_ns,
        )
        wall_recv_ns = time.time_ns()
        perf_recv_ns = time.perf_counter_ns()
        response.raise_for_status()
        payload = _resp_json(response)
        server_time = int(payload)
        if self.audit is not None and getattr(self.audit, "enabled", False):
            self.audit.record_clock_sync(
                wall_send_ns=wall_send_ns,
                wall_recv_ns=wall_recv_ns,
                perf_send_ns=perf_send_ns,
                perf_recv_ns=perf_recv_ns,
                server_time_seconds=server_time,
                source=audit_source,
            )
        return server_time

    async def _request(
        self,
        method: str,
        path: str,
        *,
        audit_wall_start_ns: int | None = None,
        audit_perf_start_ns: int | None = None,
        audit_queue_wait_ms: float | None = None,
        audit_semaphore_wait_ms: float | None = None,
        audit_backoff_ms: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        wall_start_ns = audit_wall_start_ns or time.time_ns()
        perf_start_ns = audit_perf_start_ns or time.perf_counter_ns()
        response: httpx.Response | None = None
        exception: str | None = None
        try:
            response = await self._client.request(method, path, **kwargs)
            return response
        except Exception as exc:
            exception = str(exc)
            raise
        finally:
            wall_end_ns = time.time_ns()
            perf_end_ns = time.perf_counter_ns()
            if self.audit is not None and getattr(self.audit, "enabled", False):
                self.audit.record_network(
                    method=method,
                    endpoint=path,
                    status_code=response.status_code if response is not None else None,
                    wall_start_ns=wall_start_ns,
                    wall_end_ns=wall_end_ns,
                    perf_start_ns=perf_start_ns,
                    perf_end_ns=perf_end_ns,
                    request_bytes=_request_size(kwargs),
                    response_bytes=len(response.content) if response is not None else None,
                    queue_wait_ms=audit_queue_wait_ms,
                    semaphore_wait_ms=audit_semaphore_wait_ms,
                    backoff_ms=audit_backoff_ms,
                    exception=exception,
                )


def _request_size(kwargs: dict[str, Any]) -> int:
    total = 0
    for key in ["params", "json", "data", "content"]:
        value = kwargs.get(key)
        if value is None:
            continue
        if isinstance(value, bytes):
            total += len(value)
        else:
            total += len(str(value).encode("utf-8"))
    return total
