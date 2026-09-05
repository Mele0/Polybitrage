from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from src.kalshi.client import KalshiClient, kalshi_token_id
from src.polymarket.models import OrderBook


async def discover_kalshi_pairs(
    *,
    out_path: Path,
    base_url: str = "https://external-api.kalshi.com/trade-api/v2",
    status: str = "open",
    event_ticker: str | None = None,
    series_ticker: str | None = None,
    tickers: list[str] | None = None,
    search: str | None = None,
    limit: int = 100,
    max_pages: int | None = 5,
    max_markets: int | None = 200,
    min_volume: float | None = None,
    min_open_interest: float | None = None,
    require_quotes: bool = True,
    require_orderbook_depth: bool = True,
    max_concurrent_requests: int = 10,
    enabled: bool = False,
) -> dict[str, Any]:
    async with KalshiClient(base_url=base_url) as client:
        markets = await client.get_markets(
            limit=limit,
            max_pages=max_pages,
            event_ticker=event_ticker,
            series_ticker=series_ticker,
            status=status,
            tickers=tickers,
        )
    filtered = _filter_markets(
        markets,
        search=search,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        require_quotes=require_quotes,
    )
    market_list_filtered_count = len(filtered)
    orderbook_depth_checked = 0
    if require_orderbook_depth:
        async with KalshiClient(base_url=base_url) as client:
            filtered, orderbook_depth_checked = await _filter_by_orderbook_depth(
                client,
                filtered,
                max_markets=max_markets,
                max_concurrent_requests=max_concurrent_requests,
            )
    elif max_markets is not None:
        filtered = filtered[: max(0, max_markets)]
    pairs = [same_market_complement_pair(market, enabled=enabled) for market in filtered]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(
            {
                "notes": (
                    "Auto-discovered Kalshi same-market YES+NO complement candidates. "
                    "Paper/audit only; live Kalshi execution is intentionally disabled."
                ),
                "exchange": "kalshi",
                "generated_at": datetime.now(UTC).isoformat(),
                "pair_mode": "same_market_yes_no_complement",
                "pairs": pairs,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {
        "markets_loaded": len(markets),
        "markets_after_market_list_filters": market_list_filtered_count,
        "markets_after_filters": len(filtered),
        "require_orderbook_depth": require_orderbook_depth,
        "orderbook_depth_checked": orderbook_depth_checked,
        "pairs_written": len(pairs),
        "require_quotes": require_quotes,
        "out": str(out_path),
        "first_pair": pairs[0]["name"] if pairs else None,
    }


def same_market_complement_pair(market: dict[str, Any], *, enabled: bool = False) -> dict[str, Any]:
    ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip().upper()
    if not ticker:
        raise ValueError("Kalshi market is missing ticker")
    event_ticker = str(market.get("event_ticker") or "").strip().upper()
    yes_label = str(market.get("yes_sub_title") or market.get("subtitle") or "YES")
    no_label = str(market.get("no_sub_title") or "NO")
    title = str(market.get("title") or market.get("market_title") or ticker)
    return {
        "name": f"KALSHI_{_safe_name(ticker)}_YES_NO_COMPLEMENT",
        "enabled": bool(enabled),
        "parent_market_ticker": ticker,
        "child_market_ticker": ticker,
        "parent_market_slug": ticker,
        "child_market_slug": ticker,
        "parent_frontend_url": f"https://kalshi.com/markets/{ticker.lower()}",
        "child_frontend_url": f"https://kalshi.com/markets/{ticker.lower()}",
        "parent_outcome_label": yes_label,
        "child_outcome_label": no_label,
        "parent_yes_token_id": kalshi_token_id(ticker, "yes"),
        "parent_no_token_id": kalshi_token_id(ticker, "no"),
        "child_yes_token_id": kalshi_token_id(ticker, "yes"),
        "child_no_token_id": kalshi_token_id(ticker, "no"),
        "parent_display_price": _float_or_none(market.get("yes_ask_dollars")),
        "child_display_price": _float_or_none(market.get("no_ask_dollars")),
        "relation": "same_market_complement",
        "relation_subtype": "same_market_yes_no_complement",
        "relation_safety": "clean",
        "boundary_ambiguity": False,
        "confidence": "high",
        "manual_rule_verification_required": False,
        "reason": "Buying YES and NO on the same Kalshi binary market guarantees $1 before fees if both legs fill.",
        "warnings": [
            "Kalshi fees and non-atomic multi-leg execution must be accounted for before live trading.",
        ],
        "kalshi_market": {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "title": title,
            "status": market.get("status"),
            "close_time": market.get("close_time"),
            "volume_fp": market.get("volume_fp"),
            "open_interest_fp": market.get("open_interest_fp"),
            "orderbook_depth": market.get("_kalshi_orderbook_depth"),
        },
        "trade_template": {
            "leg_1": {"market": "parent", "outcome": "YES", "side": "BUY"},
            "leg_2": {"market": "child", "outcome": "NO", "side": "BUY"},
        },
        "overrides": {
            "fee_mode": "conservative",
            "fee_category": "kalshi",
            "order_role": "taker",
            "slippage_buffer": 0.0,
        },
    }


def _filter_markets(
    markets: list[dict[str, Any]],
    *,
    search: str | None,
    min_volume: float | None,
    min_open_interest: float | None,
    require_quotes: bool,
) -> list[dict[str, Any]]:
    terms = [term.strip().lower() for term in str(search or "").split(",") if term.strip()]
    filtered: list[dict[str, Any]] = []
    for market in markets:
        ticker = str(market.get("ticker") or market.get("market_ticker") or "")
        if not ticker:
            continue
        haystack = " ".join(
            str(market.get(key) or "")
            for key in [
                "ticker",
                "event_ticker",
                "title",
                "market_title",
                "subtitle",
                "yes_sub_title",
                "no_sub_title",
                "category",
            ]
        ).lower()
        if terms and not all(term in haystack for term in terms):
            continue
        if require_quotes and not _has_two_sided_quote(market):
            continue
        if min_volume is not None and _float_or_none(market.get("volume_fp")) is not None:
            if (_float_or_none(market.get("volume_fp")) or 0.0) < min_volume:
                continue
        if min_open_interest is not None and _float_or_none(market.get("open_interest_fp")) is not None:
            if (_float_or_none(market.get("open_interest_fp")) or 0.0) < min_open_interest:
                continue
        filtered.append(market)
    return filtered


async def _filter_by_orderbook_depth(
    client: KalshiClient,
    markets: list[dict[str, Any]],
    *,
    max_markets: int | None,
    max_concurrent_requests: int,
) -> tuple[list[dict[str, Any]], int]:
    target = max(0, max_markets) if max_markets is not None else len(markets)
    if target == 0:
        return [], 0
    validated: list[dict[str, Any]] = []
    checked = 0
    batch_size = max(1, int(max_concurrent_requests))
    for start in range(0, len(markets), batch_size):
        batch = markets[start : start + batch_size]
        results = await _fetch_orderbook_depth_batch(client, batch)
        checked += len(batch)
        for market, summary in results:
            if summary.get("has_complement_depth"):
                enriched = dict(market)
                enriched["_kalshi_orderbook_depth"] = summary
                validated.append(enriched)
                if len(validated) >= target:
                    return validated, checked
    return validated, checked


async def _fetch_orderbook_depth_batch(
    client: KalshiClient,
    markets: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    async def fetch(market: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip().upper()
        if not ticker:
            return market, {"has_complement_depth": False, "error": "missing_ticker"}
        try:
            books = await client.get_market_books(ticker)
        except Exception as exc:  # noqa: BLE001 - discovery should skip one bad market, not fail the export.
            return market, {"has_complement_depth": False, "ticker": ticker, "error": str(exc)}
        return market, same_market_complement_depth_summary(books)

    return await asyncio.gather(*(fetch(market) for market in markets))


def same_market_complement_depth_summary(books: dict[str, OrderBook]) -> dict[str, Any]:
    yes_book = books.get("yes")
    no_book = books.get("no")
    yes_asks = list(yes_book.asks if yes_book else [])
    no_asks = list(no_book.asks if no_book else [])
    yes_best = yes_asks[0].price if yes_asks else None
    no_best = no_asks[0].price if no_asks else None
    return {
        "has_complement_depth": bool(yes_asks and no_asks),
        "yes_ask_levels": len(yes_asks),
        "no_ask_levels": len(no_asks),
        "yes_ask_size": _decimal_sum_to_float(level.size for level in yes_asks),
        "no_ask_size": _decimal_sum_to_float(level.size for level in no_asks),
        "yes_best_ask": _decimal_to_float(yes_best),
        "no_best_ask": _decimal_to_float(no_best),
    }


def _has_two_sided_quote(market: dict[str, Any]) -> bool:
    yes_ask = _float_or_none(market.get("yes_ask_dollars"))
    no_ask = _float_or_none(market.get("no_ask_dollars"))
    return yes_ask is not None and no_ask is not None and 0.0 < yes_ask < 1.0 and 0.0 < no_ask < 1.0


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decimal_sum_to_float(values: Any) -> float:
    total = Decimal("0")
    for value in values:
        total += Decimal(value)
    return float(total)


def _decimal_to_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None
