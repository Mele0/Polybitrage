from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.polymarket.gamma_client import GammaClient, resolve_binary_token_ids_from_market
from src.simulator import _extract_numbers

ASSETS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "xrp",
}


@dataclass
class ThresholdMarket:
    asset: str
    threshold: float
    label: str
    event_slug: str
    event_title: str
    market: dict[str, Any]
    yes_token_id: str | None
    no_token_id: str | None


@dataclass
class RangeMarket:
    asset: str
    low: float
    high: float
    kind: str
    label: str
    event_slug: str
    event_title: str
    market: dict[str, Any]
    yes_token_id: str | None
    no_token_id: str | None


async def discover_crypto_pairs(
    *,
    out_path: Path,
    assets: list[str] | None = None,
    days_ahead: int = 1,
    date_slug: str | None = None,
    date_year: int | None = None,
    include_boundary_ambiguous: bool = True,
    adjacent_only: bool = False,
    min_display_price: float | None = 0.001,
    spot_prices: dict[str, float] | None = None,
    max_threshold_distance_pct: float | None = None,
) -> dict[str, Any]:
    selected_assets = [asset.upper() for asset in (assets or list(ASSETS)) if asset.upper() in ASSETS]
    slug_date = date_slug or _date_slug(days_ahead)
    spot_prices = spot_prices or {}
    pairs: list[dict[str, Any]] = []
    scanned_events = 0
    missing_events: list[str] = []
    resolved_events: dict[str, str] = {}
    async with GammaClient() as gamma:
        for asset in selected_assets:
            prefix = ASSETS[asset]
            above_slugs = _event_slug_candidates(prefix, "above", slug_date, date_year=date_year)
            price_slugs = _event_slug_candidates(prefix, "price", slug_date, date_year=date_year)
            above_slug, above_event = await _first_existing_event(gamma, above_slugs)
            price_slug, price_event = await _first_existing_event(gamma, price_slugs)
            if not above_event:
                missing_events.append("|".join(above_slugs))
            if not price_event:
                missing_events.append("|".join(price_slugs))
            if not above_event or not price_event:
                continue
            resolved_events[f"{asset}_above"] = above_slug or str(above_event.get("slug") or "")
            resolved_events[f"{asset}_price"] = price_slug or str(price_event.get("slug") or "")
            scanned_events += 2
            pairs.extend(
                _build_day_pairs(
                    asset,
                    above_event,
                    price_event,
                    include_boundary_ambiguous=include_boundary_ambiguous,
                    adjacent_only=adjacent_only,
                    min_display_price=min_display_price,
                    spot_prices=spot_prices,
                    max_threshold_distance_pct=max_threshold_distance_pct,
                )
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(
            {
                "notes": "Auto-discovered crypto range/threshold candidates. Paper-only; verify rules manually.",
                "pairs": pairs,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {
        "date_slug": slug_date,
        "date_year": date_year,
        "assets": ",".join(selected_assets),
        "events_scanned": scanned_events,
        "missing_events": ",".join(missing_events),
        "resolved_events": ",".join(value for _, value in sorted(resolved_events.items())),
        "pairs_written": len(pairs),
        "clean_pairs": sum(1 for pair in pairs if pair.get("relation_safety") == "clean"),
        "boundary_ambiguous_pairs": sum(1 for pair in pairs if pair.get("relation_safety") == "boundary_ambiguous"),
        "out": str(out_path),
    }


def _build_day_pairs(
    asset: str,
    above_event: dict[str, Any],
    price_event: dict[str, Any],
    *,
    include_boundary_ambiguous: bool,
    adjacent_only: bool,
    min_display_price: float | None,
    spot_prices: dict[str, float],
    max_threshold_distance_pct: float | None,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    thresholds = _threshold_markets(asset, above_event)
    ranges = _range_markets(asset, price_event)
    thresholds.sort(key=lambda item: item.threshold)
    ranges.sort(key=lambda item: item.low)
    for range_market in ranges:
        finite_bounds = [value for value in [range_market.low, range_market.high] if abs(value) != float("inf")]
        if not _passes_spot_distance(asset, finite_bounds, spot_prices, max_threshold_distance_pct):
            continue
        if range_market.kind in {"range", "above_tail"}:
            eligible = [threshold for threshold in thresholds if threshold.threshold <= range_market.low]
            if adjacent_only and eligible:
                eligible = [eligible[-1]]
            for threshold in eligible:
                if not _passes_spot_distance(asset, [threshold.threshold], spot_prices, max_threshold_distance_pct):
                    continue
                if not _passes_display_filter(threshold.market, range_market.market, min_display_price):
                    continue
                boundary = range_market.kind == "range" and abs(threshold.threshold - range_market.low) < 1e-9
                if boundary and not include_boundary_ambiguous:
                    continue
                pairs.append(_candidate_pair(asset, threshold, range_market, boundary))
        if range_market.kind == "below_tail":
            eligible = [threshold for threshold in thresholds if threshold.threshold >= range_market.high]
            if adjacent_only and eligible:
                eligible = [eligible[0]]
            for threshold in eligible:
                if not _passes_spot_distance(asset, [threshold.threshold], spot_prices, max_threshold_distance_pct):
                    continue
                if not _passes_display_filter(threshold.market, range_market.market, min_display_price):
                    continue
                pairs.append(_negative_candidate_pair(asset, threshold, range_market))
    return pairs


def _event_date(event: dict[str, Any]) -> str | None:
    for key in ("startDate", "start_date", "endDate", "end_date"):
        value = event.get(key)
        if not value:
            continue
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            continue
    match = re.search(r"on-([a-z]+-\d{1,2}(?:-\d{4})?)", str(event.get("slug") or ""))
    return match.group(1) if match else None


async def discover_live_multiday_pairs(
    *,
    assets: list[str] | None = None,
    horizon_days: int = 14,
    include_boundary_ambiguous: bool = True,
    adjacent_only: bool = False,
    min_display_price: float | None = 0.001,
    spot_prices: dict[str, float] | None = None,
    max_threshold_distance_pct: float | None = None,
) -> dict[str, Any]:
    """Discover crypto range/threshold candidate pairs across every day currently
    listed on Polymarket (today plus `horizon_days - 1` days ahead), without
    writing any files. Walks each day's dated event-slug pattern directly so it
    needs no hard-coded series identifiers and naturally stops covering days for
    which Polymarket has not yet published markets.
    """
    selected_assets = [asset.upper() for asset in (assets or list(ASSETS)) if asset.upper() in ASSETS]
    spot_prices = spot_prices or {}
    pairs: list[dict[str, Any]] = []
    scanned_events = 0
    resolved_events: dict[str, str] = {}
    seen_event_pairs: set[tuple[str, str]] = set()
    async with GammaClient() as gamma:
        for asset in selected_assets:
            prefix = ASSETS[asset]
            for day_offset in range(max(1, horizon_days)):
                slug_date = _date_slug(day_offset)
                above_slugs = _event_slug_candidates(prefix, "above", slug_date)
                price_slugs = _event_slug_candidates(prefix, "price", slug_date)
                above_slug, above_event = await _first_existing_event(gamma, above_slugs)
                price_slug, price_event = await _first_existing_event(gamma, price_slugs)
                if not above_event or not price_event:
                    continue
                event_key = (str(above_event.get("slug") or ""), str(price_event.get("slug") or ""))
                if event_key in seen_event_pairs:
                    continue
                seen_event_pairs.add(event_key)
                resolved_events[f"{asset}_{slug_date}_above"] = above_slug or event_key[0]
                resolved_events[f"{asset}_{slug_date}_price"] = price_slug or event_key[1]
                scanned_events += 2
                event_date = _event_date(above_event) or _event_date(price_event) or slug_date
                day_pairs = _build_day_pairs(
                    asset,
                    above_event,
                    price_event,
                    include_boundary_ambiguous=include_boundary_ambiguous,
                    adjacent_only=adjacent_only,
                    min_display_price=min_display_price,
                    spot_prices=spot_prices,
                    max_threshold_distance_pct=max_threshold_distance_pct,
                )
                for pair in day_pairs:
                    pair["event_date"] = event_date
                    pair["overrides"] = {**pair.get("overrides", {}), "event_date": event_date}
                pairs.extend(day_pairs)
    return {
        "assets": ",".join(selected_assets),
        "horizon_days": horizon_days,
        "events_scanned": scanned_events,
        "event_dates_covered": ",".join(sorted({str(pair.get("event_date") or "") for pair in pairs} - {""})),
        "resolved_events": ",".join(f"{key}={value}" for key, value in sorted(resolved_events.items())),
        "pairs": pairs,
        "pairs_discovered": len(pairs),
        "clean_pairs": sum(1 for pair in pairs if pair.get("relation_safety") == "clean"),
        "boundary_ambiguous_pairs": sum(1 for pair in pairs if pair.get("relation_safety") == "boundary_ambiguous"),
    }


def _threshold_markets(asset: str, event: dict[str, Any]) -> list[ThresholdMarket]:
    out: list[ThresholdMarket] = []
    for market in event.get("markets") or []:
        threshold = _first_number(market.get("groupItemTitle") or market.get("question") or market.get("slug") or "")
        if threshold is None:
            continue
        yes, no = resolve_binary_token_ids_from_market(market)
        out.append(
            ThresholdMarket(
                asset=asset,
                threshold=threshold,
                label=_format_number(threshold),
                event_slug=str(event.get("slug") or ""),
                event_title=str(event.get("title") or ""),
                market=market,
                yes_token_id=yes,
                no_token_id=no,
            )
        )
    return out


def _range_markets(asset: str, event: dict[str, Any]) -> list[RangeMarket]:
    out: list[RangeMarket] = []
    for market in event.get("markets") or []:
        label = str(market.get("groupItemTitle") or market.get("question") or "")
        stripped = label.strip()
        nums = _extract_numbers(label)
        if stripped.startswith(">") and nums:
            low, high, kind = nums[0], float("inf"), "above_tail"
        elif stripped.startswith("<") and nums:
            low, high, kind = float("-inf"), nums[0], "below_tail"
        elif len(nums) >= 2:
            low, high, kind = nums[0], nums[1], "range"
        else:
            continue
        if high <= low:
            continue
        yes, no = resolve_binary_token_ids_from_market(market)
        out.append(
            RangeMarket(
                asset=asset,
                low=low,
                high=high,
                kind=kind,
                label=_range_label(low, high, kind),
                event_slug=str(event.get("slug") or ""),
                event_title=str(event.get("title") or ""),
                market=market,
                yes_token_id=yes,
                no_token_id=no,
            )
        )
    return out


def _candidate_pair(asset: str, parent: ThresholdMarket, child: RangeMarket, boundary: bool) -> dict[str, Any]:
    safety = "boundary_ambiguous" if boundary else "clean"
    warning = (
        "Boundary ambiguity: range lower bound equals above threshold and the above rule may be strict."
        if boundary
        else None
    )
    name = f"AUTO_{asset}_{_safe_num(parent.threshold)}_VS_{_safe_num(child.low)}_{_safe_num(child.high)}_CHILD_IMPLIES_PARENT"
    return {
        "name": name,
        "enabled": False,
        "parent_market_slug": parent.market.get("slug"),
        "child_market_slug": child.market.get("slug"),
        "parent_event_slug": parent.event_slug,
        "child_event_slug": child.event_slug,
        "parent_frontend_url": f"https://polymarket.com/event/{parent.event_slug}",
        "child_frontend_url": f"https://polymarket.com/event/{child.event_slug}",
        "parent_outcome_label": parent.label,
        "child_outcome_label": child.label,
        "parent_market_id": str(parent.market.get("id") or ""),
        "child_market_id": str(child.market.get("id") or ""),
        "parent_yes_token_id": parent.yes_token_id,
        "parent_no_token_id": parent.no_token_id,
        "child_yes_token_id": child.yes_token_id,
        "child_no_token_id": child.no_token_id,
        "parent_display_price": _display_price(parent.market),
        "child_display_price": _display_price(child.market),
        "parent_volume": _float_field(parent.market, "volume"),
        "child_volume": _float_field(child.market, "volume"),
        "parent_liquidity": _float_field(parent.market, "liquidity"),
        "child_liquidity": _float_field(child.market, "liquidity"),
        "relation": "child_implies_parent",
        "relation_subtype": "range_implies_above",
        "relation_safety": safety,
        "boundary_ambiguity": boundary,
        "boundary_warning": warning,
        "confidence": "medium",
        "relation_source": "deterministic_slug_event_rule",
        "manual_rule_verification_required": True,
        "reason": f"{asset} range {child.label} implies above {parent.label} if source/timestamp/rules match exactly.",
        "warnings": [warning] if warning else [],
        "trade_template": {
            "leg_1": {"market": "parent", "outcome": "YES", "side": "BUY"},
            "leg_2": {"market": "child", "outcome": "NO", "side": "BUY"},
        },
        "overrides": {
            "min_edge_threshold": 0.0025,
            "max_trade_size_usd": 20.0,
            "max_spread_per_leg": 0.1,
            "fee_mode": "conservative",
            "fee_category": "crypto",
            "order_role": "taker",
            "fee_buffer_usd": 0.0,
            "slippage_buffer": 0.0025,
        },
    }


def _negative_candidate_pair(asset: str, parent: ThresholdMarket, child: RangeMarket) -> dict[str, Any]:
    name = f"AUTO_{asset}_{_safe_num(parent.threshold)}_VS_BELOW_{_safe_num(child.high)}_CHILD_EXCLUDES_PARENT"
    return {
        "name": name,
        "enabled": False,
        "parent_market_slug": parent.market.get("slug"),
        "child_market_slug": child.market.get("slug"),
        "parent_event_slug": parent.event_slug,
        "child_event_slug": child.event_slug,
        "parent_frontend_url": f"https://polymarket.com/event/{parent.event_slug}",
        "child_frontend_url": f"https://polymarket.com/event/{child.event_slug}",
        "parent_outcome_label": parent.label,
        "child_outcome_label": child.label,
        "parent_market_id": str(parent.market.get("id") or ""),
        "child_market_id": str(child.market.get("id") or ""),
        "parent_yes_token_id": parent.yes_token_id,
        "parent_no_token_id": parent.no_token_id,
        "child_yes_token_id": child.yes_token_id,
        "child_no_token_id": child.no_token_id,
        "parent_display_price": _display_price(parent.market),
        "child_display_price": _display_price(child.market),
        "parent_volume": _float_field(parent.market, "volume"),
        "child_volume": _float_field(child.market, "volume"),
        "parent_liquidity": _float_field(parent.market, "liquidity"),
        "child_liquidity": _float_field(child.market, "liquidity"),
        "relation": "child_implies_not_parent",
        "relation_subtype": "range_below_excludes_above",
        "relation_safety": "clean",
        "boundary_ambiguity": False,
        "boundary_warning": None,
        "confidence": "medium",
        "relation_source": "deterministic_slug_event_rule",
        "manual_rule_verification_required": True,
        "reason": f"{asset} below-tail range {child.label} implies above {parent.label} is false if source/timestamp/rules match exactly.",
        "warnings": [],
        "trade_template": {
            "leg_1": {"market": "parent", "outcome": "NO", "side": "BUY"},
            "leg_2": {"market": "child", "outcome": "NO", "side": "BUY"},
        },
        "overrides": {
            "min_edge_threshold": 0.0025,
            "max_trade_size_usd": 20.0,
            "max_spread_per_leg": 0.1,
            "fee_mode": "conservative",
            "fee_category": "crypto",
            "order_role": "taker",
            "fee_buffer_usd": 0.0,
            "slippage_buffer": 0.0025,
        },
    }


def _date_slug(days_ahead: int) -> str:
    target = datetime.now().date() + timedelta(days=days_ahead)
    return f"{target.strftime('%B').lower()}-{target.day}"


async def _first_existing_event(gamma: GammaClient, slugs: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    for slug in slugs:
        event = await gamma.event_by_slug(slug)
        if event:
            return slug, event
    return None, None


def _event_slug_candidates(prefix: str, kind: str, slug_date: str, *, date_year: int | None = None) -> list[str]:
    clean_date = str(slug_date or "").strip().lower()
    base = f"{prefix}-{kind}-on-{clean_date}"
    if _slug_date_has_year(clean_date):
        return [base]
    year = date_year or datetime.now(UTC).year
    return [f"{base}-{year}", base]


def _slug_date_has_year(slug_date: str) -> bool:
    return bool(re.search(r"(?:^|-)\d{4}$", str(slug_date or "").strip()))


def _first_number(text: str) -> float | None:
    nums = _extract_numbers(text)
    return nums[0] if nums else None


def _display_price(market: dict[str, Any]) -> float | None:
    for key in ["bestAsk", "lastTradePrice"]:
        value = _float_field(market, key)
        if value is not None:
            return value
    prices = market.get("outcomePrices")
    if isinstance(prices, str):
        nums = _extract_numbers(prices)
        return nums[0] if nums else None
    if isinstance(prices, list) and prices:
        try:
            return float(prices[0])
        except (TypeError, ValueError):
            return None
    return None


def _passes_display_filter(parent_market: dict[str, Any], child_market: dict[str, Any], min_display_price: float | None) -> bool:
    if min_display_price is None:
        return True
    parent_price = _display_price(parent_market)
    child_price = _display_price(child_market)
    return max(parent_price or 0.0, child_price or 0.0) >= min_display_price


def _float_field(market: dict[str, Any], key: str) -> float | None:
    value = market.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    return f"{value:,.0f}" if value >= 100 else f"{value:g}"


def _range_label(low: float, high: float, kind: str) -> str:
    if kind == "above_tail":
        return f">{_format_number(low)}"
    if kind == "below_tail":
        return f"<{_format_number(high)}"
    return f"{_format_number(low)}-{_format_number(high)}"


def _safe_num(value: float) -> str:
    return re.sub(r"\W+", "_", _format_number(value)).strip("_")


def _passes_spot_distance(asset: str, thresholds: list[float], spot_prices: dict[str, float], max_pct: float | None) -> bool:
    if max_pct is None or not thresholds:
        return True
    spot = spot_prices.get(asset)
    if not spot or spot <= 0:
        return True
    return min(abs(threshold - spot) / spot for threshold in thresholds) <= max_pct / 100.0
