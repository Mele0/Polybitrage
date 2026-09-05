from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

try:
    import orjson as _orjson

    def _resp_json(response: httpx.Response) -> Any:
        return _orjson.loads(response.content)

except ImportError:
    def _resp_json(response: httpx.Response) -> Any:  # type: ignore[misc]
        return _resp_json(response)

from src.polymarket.models import OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)

KALSHI_TOKEN_PREFIX = "kalshi"
KALSHI_ORDERBOOK_BATCH_SIZE = 100
ONE = Decimal("1")
ZERO = Decimal("0")


@dataclass
class KalshiOrderBookBatchResult:
    books: dict[str, OrderBook] = field(default_factory=dict)
    unavailable_token_ids: list[str] = field(default_factory=list)
    error_by_token_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KalshiToken:
    ticker: str
    side: str

    @property
    def token_id(self) -> str:
        return kalshi_token_id(self.ticker, self.side)


class KalshiClient:
    """Small public Kalshi market-data client.

    The simulator uses Kalshi REST orderbooks for paper/audit scanning. Kalshi returns
    YES and NO bids; asks are implied from the complementary side.
    """

    def __init__(
        self,
        base_url: str = "https://external-api.kalshi.com/trade-api/v2",
        timeout_seconds: float = 10.0,
        *,
        audit: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.audit = audit
        try:
            import h2  # noqa: F401
            _http2 = True
        except ImportError:
            _http2 = False
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            http2=_http2,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
            timeout=timeout_seconds,
            headers={"User-Agent": "polybitrage/0.1 kalshi-research"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "KalshiClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def warmup(self) -> None:
        """Pre-establish TLS session to eliminate cold-start latency."""
        try:
            await self._client.head("/exchange/status")
        except Exception:
            pass

    async def get_order_book(self, token_id: str) -> OrderBook | None:
        parsed = parse_kalshi_token_id(token_id)
        if parsed is None:
            return None
        books = await self._get_market_books(parsed.ticker)
        return books.get(parsed.side)

    async def get_market_books(self, ticker: str) -> dict[str, OrderBook]:
        return await self._get_market_books(ticker)

    async def get_markets(
        self,
        *,
        limit: int = 100,
        max_pages: int | None = None,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = "open",
        tickers: list[str] | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return markets from Kalshi's paginated public market list endpoint."""
        markets: list[dict[str, Any]] = []
        cursor = ""
        page = 0
        while True:
            page += 1
            params: dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
            if cursor:
                params["cursor"] = cursor
            if event_ticker:
                params["event_ticker"] = event_ticker
            if series_ticker:
                params["series_ticker"] = series_ticker
            if status:
                params["status"] = status
            if tickers:
                params["tickers"] = ",".join(tickers)
            if min_close_ts is not None:
                params["min_close_ts"] = int(min_close_ts)
            if max_close_ts is not None:
                params["max_close_ts"] = int(max_close_ts)
            response = await self._request("GET", "/markets", params=params)
            response.raise_for_status()
            payload = _resp_json(response)
            markets.extend([item for item in payload.get("markets") or [] if isinstance(item, dict)])
            cursor = str(payload.get("cursor") or "")
            if not cursor or (max_pages is not None and page >= max_pages):
                break
        return markets

    async def get_order_books(
        self,
        token_ids: list[str],
        *,
        max_concurrent_requests: int = 10,
    ) -> KalshiOrderBookBatchResult:
        requested = list(dict.fromkeys(str(token_id) for token_id in token_ids if token_id))
        if not requested:
            return KalshiOrderBookBatchResult()
        parsed_by_token = {token_id: parse_kalshi_token_id(token_id) for token_id in requested}
        result = KalshiOrderBookBatchResult()
        invalid = [token_id for token_id, parsed in parsed_by_token.items() if parsed is None]
        for token_id in invalid:
            result.unavailable_token_ids.append(token_id)
            result.error_by_token_id[token_id] = "invalid Kalshi token id; expected kalshi:<ticker>:yes|no"

        tickers = sorted({parsed.ticker for parsed in parsed_by_token.values() if parsed is not None})
        semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))

        async def fetch(ticker: str) -> tuple[str, dict[str, OrderBook], str | None]:
            async with semaphore:
                try:
                    return ticker, await self._get_market_books(ticker), None
                except Exception as exc:  # noqa: BLE001 - one unavailable market should not kill a scan.
                    logger.info("Kalshi orderbook unavailable for %s: %s", ticker, exc)
                    return ticker, {}, str(exc)

        market_books: dict[str, dict[str, OrderBook]] = {}
        errors_by_ticker: dict[str, str] = {}
        for ticker, books, error in await asyncio.gather(*(fetch(ticker) for ticker in tickers)):
            market_books[ticker] = books
            if error:
                errors_by_ticker[ticker] = error

        for token_id, parsed in parsed_by_token.items():
            if parsed is None:
                continue
            book = market_books.get(parsed.ticker, {}).get(parsed.side)
            if book is None:
                result.unavailable_token_ids.append(token_id)
                result.error_by_token_id[token_id] = errors_by_ticker.get(parsed.ticker, "order book unavailable")
            else:
                result.books[token_id] = book
        return result

    async def get_fee_rate(self, token_id: str) -> float | None:
        return None

    async def get_fee_rates(
        self,
        token_ids: list[str],
        *,
        max_concurrent_requests: int = 10,
    ) -> dict[str, float | None]:
        return {str(token_id): None for token_id in dict.fromkeys(token_ids) if token_id}

    async def get_server_time(self, *, audit_source: str = "manual") -> int | None:
        # Kalshi does not expose the same public /time endpoint used by Polymarket.
        # Probe a lightweight public endpoint so audit network timing still records
        # Kalshi connectivity, then anchor clock samples to local wall time.
        wall_send_ns = time.time_ns()
        perf_send_ns = time.perf_counter_ns()
        try:
            response = await self._request(
                "GET",
                "/exchange/status",
                audit_wall_start_ns=wall_send_ns,
                audit_perf_start_ns=perf_send_ns,
            )
            response.raise_for_status()
        except Exception:
            # Some Kalshi environments may not expose /exchange/status publicly.
            # The background probe is diagnostic only; do not kill the run.
            pass
        wall_recv_ns = time.time_ns()
        perf_recv_ns = time.perf_counter_ns()
        server_time = int(wall_recv_ns / 1_000_000_000)
        if self.audit is not None and getattr(self.audit, "enabled", False):
            self.audit.record_clock_sync(
                wall_send_ns=wall_send_ns,
                wall_recv_ns=wall_recv_ns,
                perf_send_ns=perf_send_ns,
                perf_recv_ns=perf_recv_ns,
                server_time_seconds=server_time,
                source=f"kalshi_{audit_source}",
            )
        return server_time

    async def _get_market_books(self, ticker: str) -> dict[str, OrderBook]:
        response = await self._request("GET", f"/markets/{ticker}/orderbook")
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return kalshi_books_from_api(ticker, _resp_json(response))

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


def kalshi_token_id(ticker: str, side: str) -> str:
    clean_ticker = str(ticker or "").strip().upper()
    clean_side = str(side or "").strip().lower()
    return f"{KALSHI_TOKEN_PREFIX}:{clean_ticker}:{clean_side}"


def parse_kalshi_token_id(token_id: str) -> KalshiToken | None:
    raw = str(token_id or "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    if len(parts) == 3 and parts[0].lower() == KALSHI_TOKEN_PREFIX:
        ticker, side = parts[1].strip().upper(), parts[2].strip().lower()
    elif len(parts) == 2:
        ticker, side = parts[0].strip().upper(), parts[1].strip().lower()
    elif "#" in raw:
        ticker, side = raw.rsplit("#", 1)
        ticker, side = ticker.strip().upper(), side.strip().lower()
    else:
        return None
    if not ticker or side not in {"yes", "no"}:
        return None
    return KalshiToken(ticker=ticker, side=side)


def kalshi_books_from_api(ticker: str, payload: dict[str, Any]) -> dict[str, OrderBook]:
    raw_book = payload.get("orderbook_fp") or payload.get("orderbook") or payload
    yes_bids = _parse_levels(raw_book.get("yes_dollars") or raw_book.get("yes_dollars_fp") or raw_book.get("yes") or [])
    no_bids = _parse_levels(raw_book.get("no_dollars") or raw_book.get("no_dollars_fp") or raw_book.get("no") or [])
    timestamp = _coerce_kalshi_timestamp(payload)
    yes_book = _book_from_kalshi_sides(
        ticker=ticker,
        side="yes",
        direct_bids=yes_bids,
        opposite_bids=no_bids,
        timestamp=timestamp,
        raw_json=payload,
    )
    no_book = _book_from_kalshi_sides(
        ticker=ticker,
        side="no",
        direct_bids=no_bids,
        opposite_bids=yes_bids,
        timestamp=timestamp,
        raw_json=payload,
    )
    return {"yes": yes_book, "no": no_book}


def _book_from_kalshi_sides(
    *,
    ticker: str,
    side: str,
    direct_bids: list[OrderBookLevel],
    opposite_bids: list[OrderBookLevel],
    timestamp: datetime | None,
    raw_json: dict[str, Any],
) -> OrderBook:
    asks = [
        OrderBookLevel(price=ONE - level.price, size=level.size)
        for level in opposite_bids
        if ZERO < level.price < ONE
    ]
    book = OrderBook(
        asset_id=kalshi_token_id(ticker, side),
        bids=sorted(direct_bids, key=lambda level: level.price, reverse=True),
        asks=sorted(asks, key=lambda level: level.price),
        timestamp=timestamp,
        raw_json=dict(raw_json),
    )
    book.raw_json["_audit_source"] = "kalshi_rest_book"
    book.raw_json["_kalshi_ticker"] = str(ticker or "").upper()
    book.raw_json["_kalshi_side"] = side
    return book


def _parse_levels(values: Any) -> list[OrderBookLevel]:
    levels: list[OrderBookLevel] = []
    for item in values or []:
        if isinstance(item, dict):
            price = item.get("price_dollars") or item.get("price") or item.get("price_cents")
            size = item.get("count_fp") or item.get("quantity") or item.get("size") or item.get("count")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price, size = item[0], item[1]
        else:
            continue
        try:
            decimal_price = _coerce_price(price)
            decimal_size = Decimal(str(size))
        except Exception:
            continue
        if ZERO <= decimal_price <= ONE and decimal_size > ZERO:
            levels.append(OrderBookLevel(price=decimal_price, size=decimal_size))
    return sorted(levels, key=lambda level: level.price, reverse=True)


def _coerce_price(value: Any) -> Decimal:
    price = Decimal(str(value))
    if price > ONE:
        price = price / Decimal("100")
    return price


def _coerce_kalshi_timestamp(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("ts_ms") or payload.get("timestamp_ms") or payload.get("timestamp")
    if raw in (None, ""):
        return None
    try:
        if isinstance(raw, str) and ("T" in raw or raw.endswith("Z")):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        numeric = int(str(raw))
    except (TypeError, ValueError):
        return None
    seconds = numeric / 1000 if numeric > 10_000_000_000 else numeric
    return datetime.fromtimestamp(seconds, tz=UTC)


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
