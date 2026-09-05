from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from src.polymarket.clob_client import ClobClient, OrderBookBatchResult
from src.polymarket.clob_ws_client import ClobWebSocketClient
from src.polymarket.models import OrderBook


@dataclass
class ProviderStats:
    requested_tokens: int = 0
    unique_tokens_fetched: int = 0
    failed_book_count: int = 0
    cache_hits: int = 0
    stale_book_count: int = 0
    websocket_connected: bool = False
    websocket_reconnect_count: int = 0
    fallback_to_polling_used: bool = False
    token_update_count: int = 0
    event_triggered_recomputes: int = 0
    update_latency_ms: float | None = None


class OrderBookProvider(Protocol):
    stats: ProviderStats

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def get_books(self, token_ids: list[str]) -> dict[str, OrderBook | None]: ...
    async def get_book(self, token_id: str) -> OrderBook | None: ...
    def book_age_ms(self, token_id: str) -> float | None: ...
    def is_stale(self, token_id: str) -> bool: ...


@dataclass
class _CacheEntry:
    fetched_at: float
    book: OrderBook | None


class PollingOrderBookProvider:
    def __init__(
        self,
        clob_client: ClobClient,
        *,
        max_concurrent_requests: int = 10,
        order_book_cache_ms: int = 0,
        stale_book_ms: int = 2000,
    ) -> None:
        self.clob_client = clob_client
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self.order_book_cache_ms = max(0, order_book_cache_ms)
        self.stale_book_ms = max(1, stale_book_ms)
        self._cache: dict[str, _CacheEntry] = {}
        self.stats = ProviderStats()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def get_book(self, token_id: str) -> OrderBook | None:
        return (await self.get_books([token_id])).get(token_id)

    async def get_books(self, token_ids: list[str]) -> dict[str, OrderBook | None]:
        requested = [token_id for token_id in dict.fromkeys(token_ids) if token_id]
        self.stats = ProviderStats(requested_tokens=len(requested))
        books: dict[str, OrderBook | None] = {}
        to_fetch: list[str] = []
        now = time.monotonic()
        for token_id in requested:
            entry = self._cache.get(token_id)
            if (
                self.order_book_cache_ms > 0
                and entry is not None
                and ((now - entry.fetched_at) * 1000) <= self.order_book_cache_ms
            ):
                self.stats.cache_hits += 1
                books[token_id] = entry.book
            else:
                to_fetch.append(token_id)
        if to_fetch:
            batch = await self.clob_client.get_order_books(
                to_fetch,
                max_concurrent_requests=self.max_concurrent_requests,
            )
            fetched_at = time.monotonic()
            self.stats.unique_tokens_fetched = len(to_fetch)
            self.stats.failed_book_count = len(batch.unavailable_token_ids)
            for token_id in to_fetch:
                book = batch.books.get(token_id)
                books[token_id] = book
                self._cache[token_id] = _CacheEntry(fetched_at=fetched_at, book=book)
        for token_id in requested:
            books.setdefault(token_id, None)
        self.stats.stale_book_count = sum(1 for token_id in requested if self.is_stale(token_id))
        return books

    def book_age_ms(self, token_id: str) -> float | None:
        entry = self._cache.get(token_id)
        if entry is None:
            return None
        return max(0.0, (time.monotonic() - entry.fetched_at) * 1000)

    def is_stale(self, token_id: str) -> bool:
        age = self.book_age_ms(token_id)
        return age is None or age > self.stale_book_ms


class WebSocketOrderBookProvider:
    def __init__(
        self,
        ws_client: ClobWebSocketClient,
        *,
        polling_provider: PollingOrderBookProvider | None = None,
        stale_book_ms: int = 2000,
        fallback_to_polling: bool = True,
        allow_stale_cache: bool = False,
        fallback_cache_ms: int = 10000,
    ) -> None:
        self.ws_client = ws_client
        self.polling_provider = polling_provider
        self.stale_book_ms = max(1, stale_book_ms)
        self.fallback_to_polling = fallback_to_polling
        self.allow_stale_cache = allow_stale_cache
        self.fallback_cache_ms = max(0, fallback_cache_ms)
        self._started = False
        self._desired_tokens: set[str] = set()
        self._fallback_attempted_at: dict[str, float] = {}
        self.stats = ProviderStats()

    async def start(self) -> None:
        if self._started:
            return
        await self.ws_client.start(sorted(self._desired_tokens))
        self._started = True

    async def stop(self) -> None:
        if self._started:
            await self.ws_client.close()
        self._started = False

    async def update_subscriptions(self, token_ids: list[str]) -> None:
        new_tokens = {token_id for token_id in token_ids if token_id}
        if new_tokens == self._desired_tokens and self._started:
            return
        self._desired_tokens = new_tokens
        if self._started:
            await self.ws_client.update_subscriptions(sorted(self._desired_tokens))
        else:
            await self.start()

    async def wait_for_updates(self, timeout: float | None = None) -> set[str]:
        return await self.ws_client.wait_for_updates(timeout=timeout)

    async def get_book(self, token_id: str) -> OrderBook | None:
        return (await self.get_books([token_id])).get(token_id)

    async def get_books(self, token_ids: list[str]) -> dict[str, OrderBook | None]:
        requested = [token_id for token_id in dict.fromkeys(token_ids) if token_id]
        self.stats = ProviderStats(requested_tokens=len(requested))
        if not requested:
            return {}
        await self.update_subscriptions(requested)

        now = time.monotonic()
        seed_targets = [
            token_id
            for token_id in requested
            if self._fallback_due(token_id, now) and (
                # Seed if the WS has no book yet, OR if it has a book but:
                # (a) the ask side has no price at all (only bid events received), OR
                # (b) the ask side has a price but zero total depth — this happens
                #     after Bug-1 fix where replace_best_ask stores a placeholder
                #     level with size=ZERO when no real size is available.  The
                #     N-leg scanner uses best_ask (price only) and would see a
                #     phantom positive edge, then fail at REST recheck.  Re-seeding
                #     via REST clears the stale price within fallback_cache_ms.
                (ws_book := self.ws_client.get_book(token_id)) is None
                or ws_book.best_ask is None
                or ws_book.ask_depth == 0.0
            )
        ]
        attempted_this_call: set[str] = set(seed_targets)
        if seed_targets and self.polling_provider is not None:
            self._mark_fallback_attempt(seed_targets, now)
            seed_books = await self.polling_provider.get_books(seed_targets)
            self.ws_client.seed_books({token_id: book for token_id, book in seed_books.items() if book is not None})
            self.stats.fallback_to_polling_used = True
            self.stats.unique_tokens_fetched += self.polling_provider.stats.unique_tokens_fetched
            self.stats.cache_hits += self.polling_provider.stats.cache_hits
            self.stats.failed_book_count += self.polling_provider.stats.failed_book_count

        books: dict[str, OrderBook | None] = {}
        fallback_tokens: list[str] = []
        for token_id in requested:
            book = self.ws_client.get_book(token_id)
            if book is not None and not self.is_stale(token_id):
                books[token_id] = book
            elif book is not None and self.allow_stale_cache:
                books[token_id] = book
            elif (
                self.fallback_to_polling
                and self.polling_provider is not None
                and token_id not in attempted_this_call
                and self._fallback_due(token_id, now)
            ):
                fallback_tokens.append(token_id)
            else:
                books[token_id] = None

        if fallback_tokens and self.polling_provider is not None:
            self._mark_fallback_attempt(fallback_tokens, now)
            fallback_books = await self.polling_provider.get_books(fallback_tokens)
            self.ws_client.seed_books({token_id: book for token_id, book in fallback_books.items() if book is not None})
            self.stats.fallback_to_polling_used = True
            self.stats.unique_tokens_fetched += self.polling_provider.stats.unique_tokens_fetched
            self.stats.cache_hits += self.polling_provider.stats.cache_hits
            self.stats.failed_book_count += self.polling_provider.stats.failed_book_count
            books.update(fallback_books)

        for token_id in requested:
            books.setdefault(token_id, None)
        self.stats.websocket_connected = self.ws_client.connected
        self.stats.websocket_reconnect_count = self.ws_client.reconnect_count
        self.stats.token_update_count = self.ws_client.token_update_count
        self.stats.event_triggered_recomputes = self.ws_client.event_triggered_recomputes
        self.stats.stale_book_count = sum(1 for token_id in requested if self.is_stale(token_id))
        self.stats.failed_book_count = max(
            self.stats.failed_book_count,
            sum(1 for token_id in requested if books.get(token_id) is None),
        )
        latencies = [latency for token_id in requested if (latency := self.ws_client.get_update_latency_ms(token_id)) is not None]
        self.stats.update_latency_ms = max(latencies) if latencies else None
        return books

    async def priority_seed(self, token_ids: list[str]) -> None:
        """REST-seed specific tokens immediately, bypassing fallback_cache_ms.

        Use for tokens belonging to near-arb pairs where stale WS prices can
        generate phantom opportunities.  After seeding, the fallback timestamp
        is updated so that regular seeding does not immediately repeat.
        """
        if not token_ids or self.polling_provider is None:
            return
        fresh = await self.polling_provider.get_books(token_ids)
        self.ws_client.seed_books(
            {tid: book for tid, book in fresh.items() if book is not None}
        )
        now = time.monotonic()
        self._mark_fallback_attempt(token_ids, now)

    def _fallback_due(self, token_id: str, now: float) -> bool:
        if self.fallback_cache_ms <= 0:
            return True
        last_attempt = self._fallback_attempted_at.get(token_id)
        return last_attempt is None or ((now - last_attempt) * 1000) >= self.fallback_cache_ms

    def _mark_fallback_attempt(self, token_ids: list[str], now: float) -> None:
        for token_id in token_ids:
            self._fallback_attempted_at[token_id] = now

    def book_age_ms(self, token_id: str) -> float | None:
        ws_age = self.ws_client.get_book_age_ms(token_id)
        if ws_age is not None:
            return ws_age
        if self.polling_provider is not None:
            return self.polling_provider.book_age_ms(token_id)
        return None

    def is_stale(self, token_id: str) -> bool:
        age = self.book_age_ms(token_id)
        return age is None or age > self.stale_book_ms
