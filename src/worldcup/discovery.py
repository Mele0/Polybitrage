"""World Cup market discovery via the Polymarket Gamma API.

Enumerates active World Cup events (by tag/series), normalizes every binary
market inside each event into :class:`~src.worldcup.schema.WorldCupMarket`, and
groups match events by ``gameId`` so that markets which live in separate events
but belong to the same fixture (moneyline / totals / exact-score / props) are
collected under one logical game.  Futures events (tournament winner, player
ladders) have no ``gameId`` and are kept as standalone groups.

Results are cached to JSON with a short TTL so repeated scans do not hammer the
API but same-day prop markets are still picked up quickly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.polymarket.gamma_client import GammaClient
from src.worldcup.schema import MarketKind, WorldCupMarket

# tag 102232 == "FIFA World Cup"; series 11433 == "soccer-fifwc".
FIFA_WORLD_CUP_TAG_ID = "102232"
FIFA_WORLD_CUP_SERIES_ID = "11433"

# Default search terms used to keep only World-Cup-relevant events when scanning
# a broader tag.  Matching is case-insensitive against event slug/title.
DEFAULT_SEARCH_TERMS = (
    "world cup", "fifa", "fifwc", "world-cup",
    "goals", "corners", "cards", "shots", "exact score", "first team to score",
    "both teams to score", "missed penalt", "winner", "messi", "neymar",
)


@dataclass
class WorldCupEvent:
    event_id: str
    slug: str
    title: str
    game_id: str | None
    sport: str | None
    teams: tuple[str, ...]
    neg_risk: bool
    start_time: str | None
    end_date: str | None
    active: bool
    closed: bool
    volume: float | None
    liquidity: float | None
    markets: list[WorldCupMarket] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_match(self) -> bool:
        return self.game_id is not None

    @property
    def frontend_url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}"


@dataclass
class WorldCupGame:
    """All events/markets belonging to one fixture (grouped by gameId), or one
    standalone futures event when ``game_id`` is None."""

    game_id: str | None
    teams: tuple[str, ...]
    events: list[WorldCupEvent] = field(default_factory=list)

    @property
    def markets(self) -> list[WorldCupMarket]:
        return [m for ev in self.events for m in ev.markets]

    @property
    def primary_slug(self) -> str:
        return self.events[0].slug if self.events else ""

    @property
    def label(self) -> str:
        if self.teams:
            return " vs. ".join(self.teams[:2]) if len(self.teams) >= 2 else self.teams[0]
        return self.primary_slug

    @property
    def start_time(self) -> str | None:
        for ev in self.events:
            if ev.start_time:
                return ev.start_time
        return None


@dataclass
class DiscoveryResult:
    games: list[WorldCupGame]
    futures: list[WorldCupGame]
    fetched_at: float
    events_scanned: int

    @property
    def all_games(self) -> list[WorldCupGame]:
        return self.games + self.futures

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for g in self.all_games:
            for m in g.markets:
                kinds[m.kind.value] = kinds.get(m.kind.value, 0) + 1
        return {
            "events_scanned": self.events_scanned,
            "match_games": len(self.games),
            "futures_groups": len(self.futures),
            "total_markets": sum(len(g.markets) for g in self.all_games),
            "market_kinds": kinds,
            "fetched_at": self.fetched_at,
        }


def _matches_terms(event: dict[str, Any], terms: tuple[str, ...]) -> bool:
    if not terms:
        return True
    hay = f"{event.get('slug','')} {event.get('title','')}".lower()
    return any(term in hay for term in terms)


def _build_event(raw: dict[str, Any]) -> WorldCupEvent:
    teams = tuple(
        str(t.get("name") or t.get("abbreviation") or "")
        for t in (raw.get("teams") or [])
        if isinstance(t, dict) and (t.get("name") or t.get("abbreviation"))
    )
    sport = raw.get("sport")
    sport_name = sport.get("sport") if isinstance(sport, dict) else (str(sport) if sport else None)
    ev = WorldCupEvent(
        event_id=str(raw.get("id") or ""),
        slug=str(raw.get("slug") or ""),
        title=str(raw.get("title") or ""),
        game_id=(str(raw.get("gameId")) if raw.get("gameId") is not None else None),
        sport=sport_name,
        teams=teams,
        neg_risk=bool(raw.get("negRisk")),
        start_time=_optstr(raw.get("startTime") or raw.get("startDate")),
        end_date=_optstr(raw.get("endDate")),
        active=bool(raw.get("active", True)),
        closed=bool(raw.get("closed", False)),
        volume=_to_float(raw.get("volume")),
        liquidity=_to_float(raw.get("liquidity")),
        raw=raw,
    )
    for m in raw.get("markets") or []:
        if not isinstance(m, dict):
            continue
        market = WorldCupMarket.from_gamma(m, raw)
        ev.markets.append(market)
    return ev


def _optstr(v: Any) -> str | None:
    return None if v in (None, "") else str(v)


def _to_float(v: Any) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _group_by_game(events: list[WorldCupEvent]) -> tuple[list[WorldCupGame], list[WorldCupGame]]:
    by_game: dict[str, WorldCupGame] = {}
    futures: list[WorldCupGame] = []
    for ev in events:
        if ev.game_id is None:
            futures.append(WorldCupGame(game_id=None, teams=ev.teams, events=[ev]))
            continue
        game = by_game.get(ev.game_id)
        if game is None:
            game = WorldCupGame(game_id=ev.game_id, teams=ev.teams, events=[])
            by_game[ev.game_id] = game
        game.events.append(ev)
        if not game.teams and ev.teams:
            game.teams = ev.teams
    return list(by_game.values()), futures


async def discover_world_cup(
    *,
    tag_id: str = FIFA_WORLD_CUP_TAG_ID,
    series_id: str | None = FIFA_WORLD_CUP_SERIES_ID,
    include_closed: bool = False,
    max_pages: int = 20,
    limit: int = 100,
    search_terms: tuple[str, ...] = DEFAULT_SEARCH_TERMS,
    include_futures: bool = True,
    only_active_markets: bool = True,
    gamma: GammaClient | None = None,
) -> DiscoveryResult:
    """Discover and normalize active World Cup events.

    Tries the tag listing first, then merges the series listing (dedup by event
    id) so we catch both taxonomies.  Filters to events whose markets are still
    tradable.
    """
    owns_gamma = gamma is None
    gamma = gamma or GammaClient()
    raw_events: dict[str, dict[str, Any]] = {}
    try:
        for src_kwargs in (
            {"tag_id": tag_id},
            *([{"series_id": series_id}] if series_id else []),
        ):
            rows = await gamma.list_events(
                closed=None if include_closed else False,
                active=True,
                limit=limit,
                max_pages=max_pages,
                **src_kwargs,
            )
            for row in rows:
                eid = str(row.get("id") or "")
                if eid and eid not in raw_events and _matches_terms(row, search_terms):
                    raw_events[eid] = row
    finally:
        if owns_gamma:
            await gamma.close()

    events: list[WorldCupEvent] = []
    for raw in raw_events.values():
        ev = _build_event(raw)
        if only_active_markets:
            ev.markets = [m for m in ev.markets if m.active and not m.closed and m.yes_token_id]
        if not ev.markets:
            continue
        events.append(ev)

    games, futures = _group_by_game(events)
    if not include_futures:
        futures = []
    return DiscoveryResult(
        games=games,
        futures=futures,
        fetched_at=time.time(),
        events_scanned=len(raw_events),
    )


# ── Caching ─────────────────────────────────────────────────────────────────


_EVENT_CACHE_KEYS = ("id", "slug", "title", "gameId", "sport", "teams", "negRisk",
                     "startTime", "startDate", "endDate", "active", "closed", "volume", "liquidity")
_MARKET_CACHE_KEYS = ("id", "conditionId", "question", "groupItemTitle", "sportsMarketType",
                      "line", "outcomes", "clobTokenIds", "bestBid", "bestAsk", "lastTradePrice",
                      "volume", "volumeNum", "liquidity", "liquidityNum", "negRisk", "active",
                      "closed", "acceptingOrders", "gameStartTime", "startDate", "endDate",
                      "orderPriceMinTickSize", "orderMinSize")


def _slim_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the normalizer needs — raw Gamma payloads carry large
    description/reward/uma blobs that bloat the cache (tens of MB) for no benefit."""
    slim = {k: raw.get(k) for k in _EVENT_CACHE_KEYS if k in raw}
    slim["markets"] = [
        {k: m.get(k) for k in _MARKET_CACHE_KEYS if k in m}
        for m in (raw.get("markets") or []) if isinstance(m, dict)
    ]
    return slim


def _cache_payload(result: DiscoveryResult) -> dict[str, Any]:
    return {
        "fetched_at": result.fetched_at,
        "events_scanned": result.events_scanned,
        "events": [_slim_event(ev.raw) for g in result.all_games for ev in g.events],
    }


def save_discovery_cache(result: DiscoveryResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_cache_payload(result)), encoding="utf-8")


def load_discovery_cache(path: Path, *, max_age_s: float, only_active_markets: bool = True) -> DiscoveryResult | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    fetched_at = float(payload.get("fetched_at") or 0.0)
    if (time.time() - fetched_at) > max_age_s:
        return None
    events: list[WorldCupEvent] = []
    for raw in payload.get("events") or []:
        ev = _build_event(raw)
        if only_active_markets:
            ev.markets = [m for m in ev.markets if m.active and not m.closed and m.yes_token_id]
        if ev.markets:
            events.append(ev)
    games, futures = _group_by_game(events)
    return DiscoveryResult(
        games=games,
        futures=futures,
        fetched_at=fetched_at,
        events_scanned=int(payload.get("events_scanned") or len(events)),
    )


def collect_token_ids(result: DiscoveryResult, *, kinds: set[MarketKind] | None = None) -> list[str]:
    """All YES+NO token ids across discovered markets, optionally filtered by kind."""
    tokens: list[str] = []
    seen: set[str] = set()
    for g in result.all_games:
        for m in g.markets:
            if kinds is not None and m.kind not in kinds:
                continue
            for tid in (m.yes_token_id, m.no_token_id):
                if tid and tid not in seen:
                    seen.add(tid)
                    tokens.append(tid)
    return tokens
