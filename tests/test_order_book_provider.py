from __future__ import annotations

import asyncio
import time
from decimal import Decimal

from src.polymarket.models import OrderBook, OrderBookLevel
from src.polymarket.order_book_provider import ProviderStats, WebSocketOrderBookProvider


def _book(token_id: str, ask: str = "0.50") -> OrderBook:
    return OrderBook(
        asset_id=token_id,
        asks=[OrderBookLevel(price=Decimal(ask), size=Decimal("10"))],
    )


class FakePollingProvider:
    def __init__(self, books: dict[str, OrderBook] | None = None) -> None:
        self.books = books or {}
        self.calls: list[list[str]] = []
        self.stats = ProviderStats()

    async def get_books(self, token_ids: list[str]) -> dict[str, OrderBook | None]:
        requested = [token_id for token_id in dict.fromkeys(token_ids) if token_id]
        self.calls.append(requested)
        unavailable = [token_id for token_id in requested if token_id not in self.books]
        self.stats = ProviderStats(
            requested_tokens=len(requested),
            unique_tokens_fetched=len(requested),
            failed_book_count=len(unavailable),
        )
        return {token_id: self.books.get(token_id) for token_id in requested}

    def book_age_ms(self, token_id: str) -> float | None:
        return None


class FakeWebSocketClient:
    def __init__(self, books: dict[str, OrderBook] | None = None) -> None:
        self.books = dict(books or {})
        self.connected = True
        self.reconnect_count = 0
        self.token_update_count = 0
        self.event_triggered_recomputes = 0
        self.started_with: list[str] | None = None
        self.updated_with: list[str] | None = None
        self.updated_at = {token_id: time.monotonic() for token_id in self.books}

    async def start(self, token_ids: list[str]) -> None:
        self.started_with = list(token_ids)

    async def close(self) -> None:
        return None

    async def update_subscriptions(self, token_ids: list[str]) -> None:
        self.updated_with = list(token_ids)

    async def wait_for_updates(self, timeout: float | None = None) -> set[str]:
        return set()

    def get_book(self, token_id: str) -> OrderBook | None:
        return self.books.get(token_id)

    def seed_books(self, books: dict[str, OrderBook]) -> None:
        now = time.monotonic()
        self.books.update(books)
        for token_id in books:
            self.updated_at[token_id] = now

    def get_book_age_ms(self, token_id: str) -> float | None:
        updated_at = self.updated_at.get(token_id)
        if updated_at is None:
            return None
        return max(0.0, (time.monotonic() - updated_at) * 1000)

    def get_update_latency_ms(self, token_id: str) -> float | None:
        return None


def test_websocket_fallback_does_not_fetch_unavailable_token_twice_in_one_scan() -> None:
    provider = WebSocketOrderBookProvider(
        FakeWebSocketClient(),
        polling_provider=FakePollingProvider(),
        fallback_to_polling=True,
        fallback_cache_ms=10000,
    )

    books = asyncio.run(provider.get_books(["missing"]))

    assert books == {"missing": None}
    assert provider.polling_provider.calls == [["missing"]]
    assert provider.stats.unique_tokens_fetched == 1
    assert provider.stats.failed_book_count == 1


def test_websocket_fallback_throttles_repeated_unavailable_tokens() -> None:
    polling = FakePollingProvider()
    provider = WebSocketOrderBookProvider(
        FakeWebSocketClient(),
        polling_provider=polling,
        fallback_to_polling=True,
        fallback_cache_ms=10000,
    )

    asyncio.run(provider.get_books(["missing"]))
    books = asyncio.run(provider.get_books(["missing"]))

    assert books == {"missing": None}
    assert polling.calls == [["missing"]]
    assert provider.stats.unique_tokens_fetched == 0
    assert provider.stats.fallback_to_polling_used is False


def test_websocket_fallback_can_retry_after_cache_window() -> None:
    polling = FakePollingProvider()
    provider = WebSocketOrderBookProvider(
        FakeWebSocketClient(),
        polling_provider=polling,
        fallback_to_polling=True,
        fallback_cache_ms=1,
    )

    asyncio.run(provider.get_books(["missing"]))
    time.sleep(0.002)
    asyncio.run(provider.get_books(["missing"]))

    assert polling.calls == [["missing"], ["missing"]]


def test_websocket_provider_uses_seeded_book_without_second_fallback() -> None:
    polling = FakePollingProvider({"token": _book("token")})
    provider = WebSocketOrderBookProvider(
        FakeWebSocketClient(),
        polling_provider=polling,
        fallback_to_polling=True,
        fallback_cache_ms=10000,
    )

    books = asyncio.run(provider.get_books(["token"]))
    books_again = asyncio.run(provider.get_books(["token"]))

    assert books["token"] is not None
    assert books_again["token"] is not None
    assert polling.calls == [["token"]]
