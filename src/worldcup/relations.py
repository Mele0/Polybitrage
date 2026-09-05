"""Relation inference engine for World Cup markets.

Given normalized markets (from :mod:`src.worldcup.discovery`), infer the logical
constraints that connect their outcome tokens and emit :class:`Relation` objects.
Every relation reduces to one of two executable primitives the existing engine
already prices:

* ``subset``            — ``child YES ⊂ parent YES`` -> buy ``parent YES`` + ``child NO``,
                          worst-case payout 1 (best 2).  Emitted as a 2-leg ``PairConfig``.
* ``complement_basket`` / ``quantity_partition`` — buy a set of YES/NO tokens that
                          partition the outcome space -> guaranteed payout 1.
                          Emitted as an N-leg spec.

The engine is fully generic across matches/players/teams (nothing is hard-coded
to a specific fixture), and a manual overrides file can inject or blacklist
relations for fragile mappings.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.worldcup.discovery import DiscoveryResult, WorldCupGame
from src.worldcup.schema import MarketKind, Side, WorldCupMarket

SUBSET = "subset"
COMPLEMENT = "complement_basket"
QUANTITY = "quantity_partition"

# A real ladder (Over 0.5..5.5, 1+..6+) has well under this many rungs. A larger
# "group" means the base-key grouping merged distinct props — skip it rather than
# emit O(n^2) bogus relations.
MAX_LADDER_GROUP = 25
# Complement baskets larger than this are not executable (and exceed live leg caps);
# the only baskets we build are small (3-way moneyline, O/U, goal bins).
MAX_BASKET_LEGS = 12

_LADDER_STOP_WORDS = ("more than", "fewer than", "less than", "at least", "or more",
                      "over", "under", "exactly", "score", "goals", "goal")


def _ladder_base_key(market: WorldCupMarket) -> str:
    """Threshold-invariant identity of a prop, used to group a ladder's rungs.

    Strips the varying number / "N+" and over/under words from the question so
    that e.g. "Messi 1+ goals" and "Messi 2+ goals" share a key but "Ronaldo 1+
    goals" and a different stat do not. Scoped to one event by the caller, this
    prevents cross-prop merges (the source of the all-pairs ladder explosion)."""
    text = (market.question or market.group_item_title or "").lower()
    text = re.sub(r"\d+(?:\.\d+)?\s*\+?", " ", text)   # drop numbers and "N+"
    for w in _LADDER_STOP_WORDS:
        text = text.replace(w, " ")
    text = re.sub(r"[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class RelationLeg:
    token_id: str
    side: str           # "YES" | "NO": which token of the market to BUY
    label: str
    market_id: str = ""
    event_slug: str = ""


@dataclass
class Relation:
    name: str
    relation_type: str           # SUBSET | COMPLEMENT | QUANTITY
    relation_subtype: str
    legs: tuple[RelationLeg, ...]
    guaranteed_payout: float
    game_id: str | None = None
    teams: tuple[str, ...] = ()
    event_slugs: tuple[str, ...] = ()
    safety: str = "clean"        # clean | boundary_ambiguous
    confidence: str = "medium"   # high | medium | low
    reason: str = ""
    # Optional state-space description consumed by the validator. Each state is a
    # dict {token_id: payout_per_share_if_state_occurs}. When present the validator
    # checks worst-case payout across these states explicitly.
    states: tuple[dict[str, float], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def token_ids(self) -> tuple[str, ...]:
        return tuple(leg.token_id for leg in self.legs)


@dataclass
class RelationGraph:
    relations: list[Relation] = field(default_factory=list)

    def by_type(self, relation_type: str) -> list[Relation]:
        return [r for r in self.relations if r.relation_type == relation_type]

    @property
    def subset_relations(self) -> list[Relation]:
        return self.by_type(SUBSET)

    @property
    def basket_relations(self) -> list[Relation]:
        return [r for r in self.relations if r.relation_type in (COMPLEMENT, QUANTITY)]

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.relations:
            out[r.relation_subtype] = out.get(r.relation_subtype, 0) + 1
        out["total"] = len(self.relations)
        return out


# ── Public entrypoint ─────────────────────────────────────────────────────────


def infer_relations(
    result: DiscoveryResult,
    *,
    overrides_path: Path | None = None,
    ladder_all_pairs: bool = False,
    enable_exact_score: bool = True,
) -> RelationGraph:
    graph = RelationGraph()
    blacklist_market_ids: set[str] = set()
    if overrides_path is not None:
        manual, blacklist_market_ids = _load_overrides(overrides_path)
        graph.relations.extend(manual)

    for game in result.all_games:
        markets = [m for m in game.markets if m.market_id not in blacklist_market_ids and m.yes_token_id]
        graph.relations.extend(
            _infer_for_game(
                game, markets,
                ladder_all_pairs=ladder_all_pairs,
                enable_exact_score=enable_exact_score,
            )
        )
    _dedupe(graph)
    return graph


def _infer_for_game(
    game: WorldCupGame,
    markets: list[WorldCupMarket],
    *,
    ladder_all_pairs: bool,
    enable_exact_score: bool,
) -> list[Relation]:
    rels: list[Relation] = []
    rels += _moneyline_complement(game, markets)
    rels += _over_under_ladders_and_complements(game, markets, ladder_all_pairs=ladder_all_pairs)
    rels += _nplus_ladders(game, markets, ladder_all_pairs=ladder_all_pairs)
    rels += _spread_ladders(game, markets, ladder_all_pairs=ladder_all_pairs)
    rels += _quantity_vs_threshold(game, markets)
    if enable_exact_score:
        rels += _exact_score_subsets(game, markets)
    return rels


# ── Builders ───────────────────────────────────────────────────────────────────


def _ctx(game: WorldCupGame, markets: list[WorldCupMarket]) -> dict[str, Any]:
    return {
        "game_id": game.game_id,
        "teams": game.teams,
        "event_slugs": tuple(sorted({m.event_slug for m in markets if m.event_slug})),
    }


def _subset_relation(
    parent: WorldCupMarket,
    child: WorldCupMarket,
    *,
    game: WorldCupGame,
    subtype: str,
    safety: str = "clean",
    confidence: str = "medium",
    reason: str = "",
    boundary: bool = False,
) -> Relation | None:
    """child YES ⊂ parent YES -> buy parent YES + child NO."""
    if not parent.yes_token_id or not child.no_token_id:
        return None
    legs = (
        RelationLeg(parent.yes_token_id, "YES", f"{parent.group_item_title or parent.question} YES",
                    parent.market_id, parent.event_slug),
        RelationLeg(child.no_token_id, "NO", f"{child.group_item_title or child.question} NO",
                    child.market_id, child.event_slug),
    )
    name = f"WC_SUBSET::{subtype}::{parent.market_id}_implies_by_{child.market_id}"
    return Relation(
        name=name,
        relation_type=SUBSET,
        relation_subtype=subtype,
        legs=legs,
        guaranteed_payout=1.0,
        game_id=game.game_id,
        teams=game.teams,
        event_slugs=tuple(sorted({parent.event_slug, child.event_slug} - {""})),
        safety="boundary_ambiguous" if boundary else safety,
        confidence=confidence,
        reason=reason or f"{child.text} implies {parent.text}",
        meta={"parent_market": parent, "child_market": child},
    )


def _basket_relation(
    legs: list[RelationLeg],
    *,
    game: WorldCupGame,
    subtype: str,
    relation_type: str = COMPLEMENT,
    payout: float = 1.0,
    safety: str = "clean",
    confidence: str = "medium",
    reason: str = "",
    states: tuple[dict[str, float], ...] = (),
) -> Relation | None:
    legs = [leg for leg in legs if leg.token_id]
    if len(legs) < 2 or len(legs) > MAX_BASKET_LEGS:
        return None
    token_sig = "_".join(leg.token_id[-6:] for leg in legs)
    return Relation(
        name=f"WC_{relation_type.upper()}::{subtype}::{token_sig}",
        relation_type=relation_type,
        relation_subtype=subtype,
        legs=tuple(legs),
        guaranteed_payout=payout,
        game_id=game.game_id,
        teams=game.teams,
        event_slugs=tuple(sorted({leg.event_slug for leg in legs} - {""})),
        safety=safety,
        confidence=confidence,
        reason=reason,
        states=states,
    )


def _moneyline_complement(game: WorldCupGame, markets: list[WorldCupMarket]) -> list[Relation]:
    # A game can span several moneyline sub-events (full-time / 1st-half / 2nd-half),
    # each its own {home, draw, away} partition — scope every basket to one event so
    # we never merge full-time and half-time win markets into a bogus basket.
    rels: list[Relation] = []
    ml_all = [m for m in markets if m.kind in (MarketKind.MONEYLINE_TEAM, MarketKind.MONEYLINE_DRAW)]
    for _, ml in _group_by(ml_all, lambda m: m.event_id).items():
        if len(ml) < 2:
            continue
        legs = [
            RelationLeg(m.yes_token_id, "YES", f"{m.group_item_title} win", m.market_id, m.event_slug)
            for m in ml if m.yes_token_id
        ]
        has_draw = any(m.kind == MarketKind.MONEYLINE_DRAW for m in ml)
        # A soccer match outcome space is {home win, draw, away win}; 3 exhaustive legs.
        exhaustive = len(legs) == 3 and has_draw
        rel = _basket_relation(
            legs, game=game, subtype="moneyline_3way", payout=1.0,
            safety="clean" if exhaustive else "boundary_ambiguous",
            confidence="high" if exhaustive else "low",
            reason="Mutually exclusive exhaustive match result (home/draw/away).",
        )
        if rel:
            rels.append(rel)
    return rels


def _over_under_ladders_and_complements(
    game: WorldCupGame, markets: list[WorldCupMarket], *, ladder_all_pairs: bool
) -> list[Relation]:
    rels: list[Relation] = []
    ladder_kinds = {
        MarketKind.TOTAL: "total_goals",
        MarketKind.CORNERS: "corners",
        MarketKind.CARDS: "cards",
        MarketKind.TEAM_TOTAL: "team_total",
        MarketKind.PLAYER_SHOTS: "player_shots",
    }
    for kind, subtype in ladder_kinds.items():
        group = [m for m in markets if m.kind == kind and m.line is not None]
        # Group rungs by (event, threshold-invariant prop key) so we only ladder
        # markets that are the SAME prop at different lines.
        for _, items in _group_by(group, lambda m: (m.event_id, _ladder_base_key(m))).items():
            if len(items) > MAX_LADDER_GROUP:
                continue
            rels += _over_under_for_line_set(game, items, subtype=subtype, ladder_all_pairs=ladder_all_pairs)
    return rels


def _over_under_for_line_set(
    game: WorldCupGame, items: list[WorldCupMarket], *, subtype: str, ladder_all_pairs: bool
) -> list[Relation]:
    rels: list[Relation] = []
    overs = sorted([m for m in items if m.side == Side.OVER], key=lambda m: m.line or 0.0)
    unders = sorted([m for m in items if m.side == Side.UNDER], key=lambda m: m.line or 0.0)

    # Over ladder: Over(high) ⊂ Over(low) -> parent=Over(low) YES, child=Over(high) NO
    rels += _ladder_subsets(
        game, overs, subtype=f"{subtype}_over_ladder", ladder_all_pairs=ladder_all_pairs,
        parent_is_lower=True, reason_fmt="Over {hi} implies Over {lo}",
    )
    # Under ladder: Under(low) ⊂ Under(high) -> parent=Under(high) YES, child=Under(low) NO
    rels += _ladder_subsets(
        game, unders, subtype=f"{subtype}_under_ladder", ladder_all_pairs=ladder_all_pairs,
        parent_is_lower=False, reason_fmt="Under {lo} implies Under {hi}",
    )
    # Same-line Over/Under complement when they are *separate* markets.
    overs_by_line = {m.line: m for m in overs}
    for under in unders:
        over = overs_by_line.get(under.line)
        if over is None or over.condition_id == under.condition_id or not over.condition_id:
            continue
        legs = [
            RelationLeg(over.yes_token_id, "YES", f"Over {over.line} YES", over.market_id, over.event_slug),
            RelationLeg(under.yes_token_id, "YES", f"Under {under.line} YES", under.market_id, under.event_slug),
        ]
        rel = _basket_relation(
            legs, game=game, subtype=f"{subtype}_ou_complement", payout=1.0,
            confidence="high",
            reason=f"Over {over.line} and Under {under.line} partition the outcome space.",
        )
        if rel:
            rels.append(rel)
    return rels


def _ladder_subsets(
    game: WorldCupGame, ordered: list[WorldCupMarket], *, subtype: str,
    ladder_all_pairs: bool, parent_is_lower: bool, reason_fmt: str,
) -> list[Relation]:
    rels: list[Relation] = []
    pairs = itertools.combinations(range(len(ordered)), 2) if ladder_all_pairs else \
        ((i, i + 1) for i in range(len(ordered) - 1))
    for i, j in pairs:
        low, high = ordered[i], ordered[j]
        if low.line == high.line:
            continue
        parent, child = (low, high) if parent_is_lower else (high, low)
        rel = _subset_relation(
            parent, child, game=game, subtype=subtype,
            reason=reason_fmt.format(lo=low.line, hi=high.line),
        )
        if rel:
            rels.append(rel)
    return rels


def _nplus_ladders(game: WorldCupGame, markets: list[WorldCupMarket], *, ladder_all_pairs: bool) -> list[Relation]:
    rels: list[Relation] = []
    nplus = [m for m in markets if m.kind in (MarketKind.PLAYER_GOALS, MarketKind.FUTURES_NPLUS) and m.threshold_n is not None]
    # Group by (event, kind, threshold-invariant prop key): one ladder per
    # player/stat per event. Critically prevents merging every "N+" market whose
    # groupItemTitle is just "2+" into one giant cross-prop group.
    for _, items in _group_by(nplus, lambda m: (m.event_id, m.kind.value, _ladder_base_key(m))).items():
        if len(items) > MAX_LADDER_GROUP:
            continue
        ordered = sorted({m.threshold_n: m for m in items}.values(), key=lambda m: m.threshold_n or 0)
        pairs = itertools.combinations(range(len(ordered)), 2) if ladder_all_pairs else \
            ((i, i + 1) for i in range(len(ordered) - 1))
        # (n+1)+ ⊂ n+  ->  parent = n+ YES, child = (n+1)+ NO
        for i, j in pairs:
            low, high = ordered[i], ordered[j]
            rel = _subset_relation(
                low, high, game=game, subtype=f"{low.kind.value}_nplus_ladder",
                reason=f"{high.threshold_n}+ implies {low.threshold_n}+",
            )
            if rel:
                rels.append(rel)
    return rels


def _spread_ladders(game: WorldCupGame, markets: list[WorldCupMarket], *, ladder_all_pairs: bool) -> list[Relation]:
    rels: list[Relation] = []
    spreads = [m for m in markets if m.kind == MarketKind.SPREAD and m.line is not None and m.team]
    for _, items in _group_by(spreads, lambda m: (m.event_id, m.team or "", _ladder_base_key(m))).items():
        if len(items) > MAX_LADDER_GROUP:
            continue
        # "Team covers line X" event = margin > -X.  Lower X -> harder cover -> subset.
        ordered = sorted({m.line: m for m in items}.values(), key=lambda m: m.line or 0.0)
        pairs = itertools.combinations(range(len(ordered)), 2) if ladder_all_pairs else \
            ((i, i + 1) for i in range(len(ordered) - 1))
        for i, j in pairs:
            low, high = ordered[i], ordered[j]   # low.line < high.line
            if low.line == high.line:
                continue
            # cover(low.line) ⊂ cover(high.line) -> parent = high-line YES, child = low-line NO
            rel = _subset_relation(
                high, low, game=game, subtype="spread_ladder",
                reason=f"{low.team} {low.line:+g} implies {high.team} {high.line:+g}",
            )
            if rel:
                rels.append(rel)
    return rels


def _quantity_vs_threshold(game: WorldCupGame, markets: list[WorldCupMarket]) -> list[Relation]:
    """Link exact-count bins to Over/Under thresholds.

    If the count bins {0,1,...,K, K+} are exhaustive and we have a threshold Over t.5,
    then Over_t.5_NO together with bins {n>t}_YES partition the space (payout 1).
    """
    bins = [m for m in markets if m.kind == MarketKind.QUANTITY_BIN and m.threshold_n is not None]
    totals = [m for m in markets if m.kind == MarketKind.TOTAL and m.line is not None and m.side == Side.OVER]
    if len(bins) < 2 or not totals:
        return []
    bins_sorted = sorted(bins, key=lambda m: m.threshold_n or 0)
    counts = [b.threshold_n for b in bins_sorted]
    # require a contiguous 0..max set for a clean partition
    contiguous = counts == list(range(min(counts), max(counts) + 1))
    rels: list[Relation] = []
    for over in totals:
        t = over.line or 0.0
        high_bins = [b for b in bins_sorted if (b.threshold_n or 0) > t]
        if len(high_bins) < 1 or not over.no_token_id:
            continue
        legs = [RelationLeg(over.no_token_id, "NO", f"Over {t} NO", over.market_id, over.event_slug)]
        legs += [
            RelationLeg(b.yes_token_id, "YES", f"exactly {b.threshold_n} YES", b.market_id, b.event_slug)
            for b in high_bins if b.yes_token_id
        ]
        rel = _basket_relation(
            legs, game=game, subtype="goals_quantity_vs_threshold", relation_type=QUANTITY, payout=1.0,
            safety="clean" if contiguous else "boundary_ambiguous",
            confidence="high" if contiguous else "low",
            reason=f"P(Over {t}) == sum of exact-count bins above {t}.",
        )
        if rel:
            rels.append(rel)
    return rels


def _exact_score_subsets(game: WorldCupGame, markets: list[WorldCupMarket]) -> list[Relation]:
    rels: list[Relation] = []
    exacts = [m for m in markets if m.kind == MarketKind.EXACT_SCORE and m.score is not None]
    totals = [m for m in markets if m.kind == MarketKind.TOTAL and m.line is not None]
    btts = next((m for m in markets if m.kind == MarketKind.BTTS), None)
    overs = {m.line: m for m in totals if m.side == Side.OVER}
    unders = {m.line: m for m in totals if m.side == Side.UNDER}
    for ex in exacts:
        a, b = ex.score  # type: ignore[misc]
        total = a + b
        # Exact(a,b) ⊂ Over(total-0.5)
        for line, over in overs.items():
            if abs((line or 0.0) - (total - 0.5)) < 1e-9 or (line is not None and line < total):
                rel = _subset_relation(over, ex, game=game, subtype="exact_score_implies_over",
                                       confidence="medium", reason=f"Exact {a}-{b} implies Over {line}")
                if rel:
                    rels.append(rel)
                break
        # Exact(a,b) ⊂ Under(total+0.5)
        for line, under in unders.items():
            if abs((line or 0.0) - (total + 0.5)) < 1e-9 or (line is not None and line > total):
                rel = _subset_relation(under, ex, game=game, subtype="exact_score_implies_under",
                                       confidence="medium", reason=f"Exact {a}-{b} implies Under {line}")
                if rel:
                    rels.append(rel)
                break
        # BTTS relation (order-independent): both scored iff a>0 and b>0
        if btts is not None:
            if a > 0 and b > 0 and btts.yes_token_id:
                # parent = BTTS Yes (yes token), child = exact
                parent = _as_pseudo(btts, use="yes")
                rel = _subset_relation(parent, ex, game=game, subtype="exact_score_implies_btts_yes",
                                       confidence="medium", reason=f"Exact {a}-{b} implies both teams scored")
                if rel:
                    rels.append(rel)
            elif (a == 0 or b == 0) and btts.no_token_id:
                parent = _as_pseudo(btts, use="no")
                rel = _subset_relation(parent, ex, game=game, subtype="exact_score_implies_btts_no",
                                       confidence="medium", reason=f"Exact {a}-{b} implies NOT both teams scored")
                if rel:
                    rels.append(rel)
    return rels


def _as_pseudo(market: WorldCupMarket, *, use: str) -> WorldCupMarket:
    """Return a shallow view of ``market`` whose ``yes_token_id`` points at the
    desired side so it can be used as the 'parent YES' of a subset relation.

    For BTTS-No as a superset we buy the BTTS NO token as the parent 'YES' leg.
    """
    if use == "yes":
        return market
    import copy
    view = copy.copy(market)
    view.yes_token_id = market.no_token_id
    view.group_item_title = f"{market.group_item_title or 'BTTS'} (NO side)"
    return view


# ── overrides + dedupe + helpers ────────────────────────────────────────────────


def _load_overrides(path: Path) -> tuple[list[Relation], set[str]]:
    if not path.exists():
        return [], set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    blacklist = {str(x) for x in (data.get("blacklist_market_ids") or [])}
    relations: list[Relation] = []
    for item in data.get("relations") or []:
        legs = tuple(
            RelationLeg(
                token_id=str(leg.get("token_id")),
                side=str(leg.get("side") or "YES").upper(),
                label=str(leg.get("label") or ""),
                market_id=str(leg.get("market_id") or ""),
                event_slug=str(leg.get("event_slug") or ""),
            )
            for leg in (item.get("legs") or [])
            if leg.get("token_id")
        )
        if len(legs) < 2:
            continue
        relations.append(
            Relation(
                name=str(item.get("name") or "manual_relation"),
                relation_type=str(item.get("relation_type") or SUBSET),
                relation_subtype=str(item.get("relation_subtype") or "manual"),
                legs=legs,
                guaranteed_payout=float(item.get("guaranteed_payout") or 1.0),
                safety=str(item.get("safety") or "clean"),
                confidence=str(item.get("confidence") or "high"),
                reason=str(item.get("reason") or "manual override"),
                meta={"manual": True},
            )
        )
    return relations, blacklist


def _dedupe(graph: RelationGraph) -> None:
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    unique: list[Relation] = []
    for r in graph.relations:
        key = (r.relation_type, tuple((leg.token_id, leg.side) for leg in r.legs))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    graph.relations = unique


def _group_by(items: list[WorldCupMarket], keyer: Any) -> dict[Any, list[WorldCupMarket]]:
    out: dict[Any, list[WorldCupMarket]] = {}
    for m in items:
        out.setdefault(keyer(m), []).append(m)
    return out


# ── PairConfig (dict) emission for subset relations ─────────────────────────────


def relation_to_pair_dict(rel: Relation, *, enabled: bool = False) -> dict[str, Any]:
    """Serialize a 2-leg subset relation to the watchlist ``PairConfig`` schema
    consumed by the existing simulator (leg_1 = parent YES, leg_2 = child NO)."""
    if rel.relation_type != SUBSET or len(rel.legs) != 2:
        raise ValueError("relation_to_pair_dict requires a 2-leg subset relation")
    parent_leg, child_leg = rel.legs
    parent_m: WorldCupMarket | None = rel.meta.get("parent_market")
    child_m: WorldCupMarket | None = rel.meta.get("child_market")
    return {
        "name": rel.name,
        "enabled": enabled,
        "parent_market_slug": parent_m.event_slug if parent_m else "",
        "child_market_slug": child_m.event_slug if child_m else "",
        "parent_event_slug": parent_m.event_slug if parent_m else (rel.event_slugs[0] if rel.event_slugs else ""),
        "child_event_slug": child_m.event_slug if child_m else (rel.event_slugs[-1] if rel.event_slugs else ""),
        "parent_outcome_label": parent_leg.label,
        "child_outcome_label": child_leg.label,
        "parent_market_id": parent_leg.market_id,
        "child_market_id": child_leg.market_id,
        "parent_yes_token_id": parent_leg.token_id,
        "child_no_token_id": child_leg.token_id,
        "parent_display_price": (parent_m.best_ask if parent_m else None),
        "child_display_price": (child_m.best_ask if child_m else None),
        "relation": "child_implies_parent",
        "relation_subtype": rel.relation_subtype,
        "relation_safety": rel.safety,
        "boundary_ambiguity": rel.safety == "boundary_ambiguous",
        "confidence": rel.confidence,
        "relation_source": "worldcup_relation_engine",
        "manual_rule_verification_required": True,
        "reason": rel.reason,
        "game_id": rel.game_id,
        "teams": list(rel.teams),
        "trade_template": {
            "leg_1": {"market": "parent", "outcome": "YES", "side": "BUY"},
            "leg_2": {"market": "child", "outcome": "NO", "side": "BUY"},
        },
        "overrides": {
            "min_edge_threshold": 0.0025,
            "max_trade_size_usd": 20.0,
            "max_spread_per_leg": 0.08,
            "fee_mode": "conservative",
            "fee_category": "other",
            "order_role": "taker",
            "fee_buffer_usd": 0.0,
            "slippage_buffer": 0.0025,
        },
    }
