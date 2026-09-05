"""Normalized World Cup market schema and market-type classifier.

Polymarket Gamma markets are all binary YES/NO contracts.  A 3-way moneyline is a
``negRisk`` group of three binary markets; a totals market is a single binary
market with a ``line`` (e.g. Over 2.5 -> YES).  This module normalizes the raw
Gamma payload into :class:`WorldCupMarket` and classifies each market into a
:class:`MarketKind` plus structured fields (line, team, player, score, n+ ladder
threshold) so the relation engine can wire markets together without hard-coding
any individual match.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MarketKind(str, Enum):
    MONEYLINE_TEAM = "moneyline_team"      # "<Team> to win"
    MONEYLINE_DRAW = "moneyline_draw"      # "Draw"
    TOTAL = "total"                        # Over/Under N.5 total goals
    TEAM_TOTAL = "team_total"              # <Team> Over/Under N.5 goals
    BTTS = "btts"                          # both teams to score (yes/no)
    SPREAD = "spread"                      # <Team> handicap +/- N.5
    EXACT_SCORE = "exact_score"            # exact final score a-b
    FIRST_TO_SCORE = "first_to_score"      # first team to score / no goal
    CORNERS = "corners"                    # Over/Under N.5 corners
    CARDS = "cards"                        # Over/Under N.5 cards
    PLAYER_SHOTS = "player_shots"          # <Player> Over/Under N.5 shots
    PLAYER_GOALS = "player_goals"          # <Player> N+ goals
    QUANTITY_BIN = "quantity_bin"          # "How many goals?" -> exact count / k+ bin
    FUTURES_WINNER = "futures_winner"      # tournament / group winner per team
    FUTURES_NPLUS = "futures_nplus"        # tournament N+ ladder (missed pens, player goals)
    UNKNOWN = "unknown"


class Side(str, Enum):
    OVER = "over"
    UNDER = "under"
    YES = "yes"
    NO = "no"
    NONE = "none"


_NUM = r"\d+(?:\.\d+)?"


def _parse_jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return [p.strip() for p in s.split(",") if p.strip()]
    return []


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(text: str) -> float | None:
    m = re.search(_NUM, text or "")
    return float(m.group(0)) if m else None


@dataclass
class WorldCupMarket:
    """A single normalized binary YES/NO market."""

    event_slug: str
    event_id: str
    game_id: str | None
    market_id: str
    condition_id: str
    question: str
    group_item_title: str
    sports_market_type: str | None
    line: float | None
    outcomes: list[str]
    yes_token_id: str | None
    no_token_id: str | None
    best_bid: float | None
    best_ask: float | None
    last_trade_price: float | None
    volume: float | None
    liquidity: float | None
    neg_risk: bool
    active: bool
    closed: bool
    accepting_orders: bool
    start_time: str | None
    end_date: str | None
    tick_size: float | None
    min_order_size: float | None
    teams: tuple[str, ...] = ()
    # Filled in by classify():
    kind: MarketKind = MarketKind.UNKNOWN
    side: Side = Side.NONE
    team: str | None = None
    player: str | None = None
    score: tuple[int, int] | None = None
    threshold_n: int | None = None  # for "N+" ladders / exact-count bins
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def token_ids(self) -> dict[str, str | None]:
        return {"yes": self.yes_token_id, "no": self.no_token_id}

    @property
    def text(self) -> str:
        return f"{self.question} || {self.group_item_title}"

    @classmethod
    def from_gamma(cls, market: dict[str, Any], event: dict[str, Any]) -> "WorldCupMarket":
        token_ids = _parse_jsonish_list(market.get("clobTokenIds"))
        yes = str(token_ids[0]) if len(token_ids) >= 1 else None
        no = str(token_ids[1]) if len(token_ids) >= 2 else None
        teams = tuple(
            str(t.get("name") or t.get("abbreviation") or "")
            for t in (event.get("teams") or [])
            if isinstance(t, dict) and (t.get("name") or t.get("abbreviation"))
        )
        obj = cls(
            event_slug=str(event.get("slug") or ""),
            event_id=str(event.get("id") or ""),
            game_id=(str(event.get("gameId")) if event.get("gameId") is not None else None),
            market_id=str(market.get("id") or ""),
            condition_id=str(market.get("conditionId") or ""),
            question=str(market.get("question") or ""),
            group_item_title=str(market.get("groupItemTitle") or ""),
            sports_market_type=(str(market["sportsMarketType"]) if market.get("sportsMarketType") else None),
            line=_to_float(market.get("line")),
            outcomes=[str(o) for o in _parse_jsonish_list(market.get("outcomes"))] or ["Yes", "No"],
            yes_token_id=yes,
            no_token_id=no,
            best_bid=_to_float(market.get("bestBid")),
            best_ask=_to_float(market.get("bestAsk")),
            last_trade_price=_to_float(market.get("lastTradePrice")),
            volume=_to_float(market.get("volume")) or _to_float(market.get("volumeNum")),
            liquidity=_to_float(market.get("liquidity")) or _to_float(market.get("liquidityNum")),
            neg_risk=bool(market.get("negRisk") or event.get("negRisk")),
            active=bool(market.get("active", True)),
            closed=bool(market.get("closed", False)),
            accepting_orders=bool(market.get("acceptingOrders", True)),
            start_time=_optstr(market.get("gameStartTime") or market.get("startDate") or event.get("startTime")),
            end_date=_optstr(market.get("endDate") or event.get("endDate")),
            tick_size=_to_float(market.get("orderPriceMinTickSize")),
            min_order_size=_to_float(market.get("orderMinSize")),
            teams=teams,
            raw=market,
        )
        classify(obj)
        return obj


def _optstr(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


# ── Classification ────────────────────────────────────────────────────────────


def classify(market: WorldCupMarket) -> WorldCupMarket:
    """Populate ``kind``/``side``/``line``/``team``/``player``/``score``/``threshold_n``.

    Combines ``sportsMarketType`` (authoritative when present) with regex on the
    question/groupItemTitle so it still works for futures markets where
    ``sportsMarketType`` is ``None``.
    """
    q = (market.question or "").lower()
    g = (market.group_item_title or "").lower()
    text = f"{q} {g}"
    smt = (market.sports_market_type or "").lower()
    teams_lower = [t.lower() for t in market.teams]

    # ── Exact score, e.g. "2-1" / "2 - 1" / "Qatar 2 - 1 Switzerland" ──────────
    score = _parse_exact_score(text)
    if score is not None and ("exact" in text or "final score" in text or re.search(rf"\b\d\s*[-:]\s*\d\b", g)):
        market.kind = MarketKind.EXACT_SCORE
        market.score = score
        return market

    # ── Over/Under family — relies on "over"/"under" + a half-line ─────────────
    side = _parse_over_under(text)
    line = market.line if market.line is not None else _line_after_keyword(text)
    if side in (Side.OVER, Side.UNDER) or smt in {"total", "totals", "team_total", "team_totals"}:
        if line is None:
            line = _first_number(text)
        market.side = side if side != Side.NONE else _parse_over_under(text)
        market.line = line
        if "corner" in text:
            market.kind = MarketKind.CORNERS
        elif "card" in text or "booking" in text:
            market.kind = MarketKind.CARDS
        elif "shot" in text:
            market.kind = MarketKind.PLAYER_SHOTS
            market.player = _guess_player(market)
        elif _mentions_single_team(text, teams_lower) or "team_total" in smt:
            market.kind = MarketKind.TEAM_TOTAL
            market.team = _match_team(text, market.teams)
        else:
            market.kind = MarketKind.TOTAL
        return market

    # ── Both teams to score ────────────────────────────────────────────────────
    if "both teams to score" in text or "btts" in text or "both teams score" in text:
        market.kind = MarketKind.BTTS
        return market

    # ── Spread / handicap ───────────────────────────────────────────────────────
    if smt in {"spread", "handicap"} or re.search(r"[+\-]\s*\d+(?:\.\d+)?", g) and ("spread" in text or "handicap" in text):
        market.kind = MarketKind.SPREAD
        market.team = _match_team(text, market.teams)
        m = re.search(r"([+\-]\s*\d+(?:\.\d+)?)", g) or re.search(r"([+\-]\s*\d+(?:\.\d+)?)", q)
        if m:
            market.line = float(m.group(1).replace(" ", ""))
        return market

    # ── First team to score / no goal ─────────────────────────────────────────
    if "first" in text and ("score" in text or "goal" in text):
        market.kind = MarketKind.FIRST_TO_SCORE
        if "no goal" in text or "neither" in text or "no team" in text:
            market.team = None
        else:
            market.team = _match_team(text, market.teams)
        return market

    # ── "N+" ladders (player tournament goals, missed penalties, ...) ──────────
    nplus = _parse_nplus(text)
    if nplus is not None:
        market.threshold_n = nplus
        if "penal" in text:
            market.kind = MarketKind.FUTURES_NPLUS
        elif "goal" in text:
            market.kind = MarketKind.PLAYER_GOALS if _looks_like_match_prop(text) else MarketKind.FUTURES_NPLUS
            market.player = _guess_player(market)
        else:
            market.kind = MarketKind.FUTURES_NPLUS
        return market

    # ── Quantity bins: "How many goals" -> exact count or k goals ──────────────
    if ("how many" in q) or re.fullmatch(r"\d+", g.strip()) or re.search(r"\bexactly\s+\d+\b", text):
        n = _first_number(g) if re.fullmatch(r"\d+", g.strip()) else _first_number(text)
        if n is not None:
            market.kind = MarketKind.QUANTITY_BIN
            market.threshold_n = int(n)
            return market

    # ── Moneyline (authoritative via sportsMarketType=moneyline) ───────────────
    if smt == "moneyline" or _looks_like_moneyline(text, teams_lower):
        if "draw" in text or "tie" in text:
            market.kind = MarketKind.MONEYLINE_DRAW
        else:
            market.kind = MarketKind.MONEYLINE_TEAM
            market.team = _match_team(text, market.teams)
        return market

    # ── Futures winner (per-team YES/NO, no sportsMarketType) ──────────────────
    if "win the world cup" in text or "winner" in text or _looks_like_bare_team(g, teams_lower):
        market.kind = MarketKind.FUTURES_WINNER
        market.team = market.group_item_title or None
        return market

    market.kind = MarketKind.UNKNOWN
    return market


def _parse_exact_score(text: str) -> tuple[int, int] | None:
    m = re.search(r"\b(\d)\s*[-:]\s*(\d)\b", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _parse_over_under(text: str) -> Side:
    if re.search(r"\bover\b", text) or re.search(r"\bmore than\b", text) or "at least" in text:
        return Side.OVER
    if re.search(r"\bunder\b", text) or re.search(r"\bfewer than\b", text) or "less than" in text:
        return Side.UNDER
    return Side.NONE


def _line_after_keyword(text: str) -> float | None:
    m = re.search(rf"(?:over|under|more than|fewer than|less than)\s+({_NUM})", text)
    return float(m.group(1)) if m else None


def _parse_nplus(text: str) -> int | None:
    m = re.search(r"\b(\d+)\s*\+", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s+or more\b", text)
    if m:
        return int(m.group(1))
    return None


def _looks_like_match_prop(text: str) -> bool:
    return any(w in text for w in ("in the match", "this match", "this game", "vs.", " vs ", "during the game"))


def _looks_like_moneyline(text: str, teams_lower: list[str]) -> bool:
    return ("to win" in text or "win on" in text or "draw" in text) and bool(teams_lower)


def _mentions_single_team(text: str, teams_lower: list[str]) -> bool:
    hits = [t for t in teams_lower if t and t in text]
    return len(hits) == 1


def _looks_like_bare_team(group_item_title: str, teams_lower: list[str]) -> bool:
    g = group_item_title.strip().lower()
    return bool(g) and (g in teams_lower or len(g.split()) <= 3)


def _match_team(text: str, teams: tuple[str, ...]) -> str | None:
    low = text.lower()
    for t in teams:
        if t and t.lower() in low:
            return t
    return None


def _guess_player(market: WorldCupMarket) -> str | None:
    # Player markets usually carry the player name in groupItemTitle.
    g = market.group_item_title.strip()
    if g and not any(w in g.lower() for w in ("over", "under", "yes", "no", "shots", "goals")):
        return g
    m = re.match(r"^([A-Z][a-zA-Z.\-']+(?:\s+[A-Z][a-zA-Z.\-']+){0,2})", market.group_item_title.strip())
    return m.group(1) if m else None
