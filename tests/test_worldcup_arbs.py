"""Unit tests for the World Cup arbitrage subsystem.

Uses synthetic Gamma payloads and synthetic order books — no network. Covers:
complement-basket arb, over/under ladder subset arb, spread ladder subset arb,
exact-score subset arb, quantity-vs-threshold arb, the state-space validator,
the no-false-arb guard, and the exposure limiter.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.polymarket.models import OrderBook, OrderBookLevel
from src.worldcup.discovery import DiscoveryResult, _build_event, _group_by_game
from src.worldcup.relations import COMPLEMENT, QUANTITY, SUBSET, infer_relations
from src.worldcup.schema import MarketKind, Side, WorldCupMarket, classify
from src.worldcup.scanner import WorldCupScannerConfig, best_basket_size
from src.worldcup.state_validator import (
    build_btts_exact_basket_states,
    synthesize_states,
    validate_legs,
    validate_relation,
)
from src.worldcup.executor import ExposureConfig, ExposureLimiter, Opportunity, OpportunityLeg


# ── helpers ─────────────────────────────────────────────────────────────────────


def make_book(token: str, ask: float | None, *, bid: float | None = None, depth: float = 500.0) -> OrderBook:
    asks = [] if ask is None else [OrderBookLevel(Decimal(str(ask)), Decimal(str(depth)))]
    bids = [] if bid is None else [OrderBookLevel(Decimal(str(bid)), Decimal(str(depth)))]
    return OrderBook(asset_id=token, asks=asks, bids=bids)


def gmarket(mid: str, question: str, group: str, yes: str, no: str, *, smt=None, line=None) -> dict:
    return {
        "id": mid, "conditionId": f"c{mid}", "question": question, "groupItemTitle": group,
        "sportsMarketType": smt, "line": line, "outcomes": json.dumps(["Yes", "No"]),
        "clobTokenIds": json.dumps([yes, no]), "active": True, "closed": False,
        "negRisk": True, "bestAsk": 0.5, "bestBid": 0.49,
    }


def event(slug: str, markets: list[dict], *, teams=("Qatar", "Switzerland"), game_id="90086911") -> dict:
    return {
        "id": "E1", "slug": slug, "title": "Qatar vs. Switzerland", "gameId": game_id,
        "negRisk": True, "active": True, "closed": False,
        "teams": [{"name": t, "abbreviation": t[:3].lower()} for t in teams],
        "markets": markets,
    }


def discovery_from(*event_dicts: dict) -> DiscoveryResult:
    events = [_build_event(e) for e in event_dicts]
    games, futures = _group_by_game(events)
    return DiscoveryResult(games=games, futures=futures, fetched_at=0.0, events_scanned=len(events))


def loose_cfg(**kw) -> WorldCupScannerConfig:
    cfg = WorldCupScannerConfig()
    cfg.fee_rate = 0.0
    cfg.max_leg_slippage = 0.0
    cfg.fee_buffer_per_share = 0.0
    cfg.latency_buffer_per_share = 0.0
    cfg.min_edge_bps = 1.0
    cfg.min_liquidity = 1.0
    cfg.max_trade_size = 1000.0
    cfg.max_total_exposure = 100000.0
    cfg.max_spread_per_leg = 1.0
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def find_relation(graph, subtype_contains: str):
    for r in graph.relations:
        if subtype_contains in r.relation_subtype:
            return r
    return None


# ── classifier ──────────────────────────────────────────────────────────────────


def test_classifier_moneyline_total_btts_exact():
    ml = WorldCupMarket.from_gamma(
        gmarket("1", "Will Switzerland win on 2026-06-13?", "Switzerland", "y", "n", smt="moneyline"),
        event("fifwc-qat-che", []),
    )
    assert ml.kind == MarketKind.MONEYLINE_TEAM and ml.team == "Switzerland"

    total = WorldCupMarket.from_gamma(
        gmarket("2", "Will there be over 2.5 goals in the match?", "Over 2.5", "y", "n", smt="total", line=2.5),
        event("fifwc-qat-che", []),
    )
    assert total.kind == MarketKind.TOTAL and total.side == Side.OVER and total.line == 2.5

    btts = WorldCupMarket.from_gamma(
        gmarket("3", "Will both teams to score?", "Yes", "y", "n"),
        event("fifwc-qat-che", []),
    )
    assert btts.kind == MarketKind.BTTS

    ex = WorldCupMarket.from_gamma(
        gmarket("4", "Will the exact final score be 2-1?", "2 - 1", "y", "n"),
        event("fifwc-qat-che", []),
    )
    assert ex.kind == MarketKind.EXACT_SCORE and ex.score == (2, 1)


# ── complement basket (moneyline 3-way) ──────────────────────────────────────────


def test_complement_basket_moneyline_arb():
    ev = event("fifwc-qat-che", [
        gmarket("d", "Will the match end in a draw?", "Draw", "draw_yes", "draw_no", smt="moneyline"),
        gmarket("a", "Will Qatar win?", "Qatar", "qat_yes", "qat_no", smt="moneyline"),
        gmarket("b", "Will Switzerland win?", "Switzerland", "che_yes", "che_no", smt="moneyline"),
    ])
    graph = infer_relations(discovery_from(ev))
    rel = find_relation(graph, "moneyline_3way")
    assert rel is not None and rel.relation_type == COMPLEMENT
    assert rel.guaranteed_payout == 1.0 and rel.safety == "clean"

    books = {
        "draw_yes": make_book("draw_yes", 0.30),
        "qat_yes": make_book("qat_yes", 0.25),
        "che_yes": make_book("che_yes", 0.40),
    }
    leg_books = [books[t] for t in rel.token_ids]
    sized = best_basket_size(leg_books, rel.guaranteed_payout, loose_cfg(), max_capital=500.0)
    assert sized is not None
    assert abs(sized["unit_cost"] - 0.95) < 1e-9
    assert sized["edge"] > 0.0
    assert validate_relation(rel, cost=sized["net_unit"]).ok


# ── over/under ladder (subset) ────────────────────────────────────────────────────


def test_over_under_ladder_subset_arb():
    ev = event("fifwc-qat-che", [
        gmarket("o15", "Will there be over 1.5 goals?", "Over 1.5", "o15y", "o15n", smt="total", line=1.5),
        gmarket("o25", "Will there be over 2.5 goals?", "Over 2.5", "o25y", "o25n", smt="total", line=2.5),
    ])
    graph = infer_relations(discovery_from(ev))
    rel = find_relation(graph, "total_goals_over_ladder")
    assert rel is not None and rel.relation_type == SUBSET
    # parent = Over 1.5 YES, child = Over 2.5 NO
    assert rel.legs[0].token_id == "o15y" and rel.legs[1].token_id == "o25n"

    books = {"o15y": make_book("o15y", 0.60, bid=0.59), "o25n": make_book("o25n", 0.30, bid=0.29)}
    leg_books = [books[t] for t in rel.token_ids]
    sized = best_basket_size(leg_books, 1.0, loose_cfg(), max_capital=500.0)
    assert sized is not None and abs(sized["unit_cost"] - 0.90) < 1e-9
    assert validate_relation(rel, cost=sized["net_unit"]).ok


# ── spread ladder (subset) ────────────────────────────────────────────────────────


def test_spread_ladder_subset_arb():
    ev = event("fifwc-qat-che", [
        gmarket("s15", "Qatar handicap -1.5", "Qatar -1.5", "s15y", "s15n", smt="spread", line=-1.5),
        gmarket("s25", "Qatar handicap -2.5", "Qatar -2.5", "s25y", "s25n", smt="spread", line=-2.5),
    ])
    graph = infer_relations(discovery_from(ev))
    rel = find_relation(graph, "spread_ladder")
    assert rel is not None and rel.relation_type == SUBSET
    # cover(-2.5) ⊂ cover(-1.5): parent = -1.5 YES, child = -2.5 NO
    assert rel.legs[0].token_id == "s15y" and rel.legs[1].token_id == "s25n"
    books = {"s15y": make_book("s15y", 0.55), "s25n": make_book("s25n", 0.40)}
    leg_books = [books[t] for t in rel.token_ids]
    sized = best_basket_size(leg_books, 1.0, loose_cfg(), max_capital=500.0)
    assert sized is not None and sized["edge"] > 0.0


# ── exact-score subset (vs BTTS) ──────────────────────────────────────────────────


def test_exact_score_subset_btts_arb():
    ev = event("fifwc-qat-che", [
        gmarket("btts", "Will both teams to score?", "Yes", "btts_yes", "btts_no"),
        gmarket("ex11", "Will the exact final score be 1-1?", "1 - 1", "ex11_yes", "ex11_no"),
    ])
    graph = infer_relations(discovery_from(ev))
    rel = find_relation(graph, "exact_score_implies_btts_yes")
    assert rel is not None and rel.relation_type == SUBSET
    # parent = BTTS YES, child = exact 1-1 NO
    assert rel.legs[0].token_id == "btts_yes" and rel.legs[1].token_id == "ex11_no"
    books = {"btts_yes": make_book("btts_yes", 0.60), "ex11_no": make_book("ex11_no", 0.30)}
    leg_books = [books[t] for t in rel.token_ids]
    sized = best_basket_size(leg_books, 1.0, loose_cfg(), max_capital=500.0)
    assert sized is not None and validate_relation(rel, cost=sized["net_unit"]).ok


# ── quantity vs threshold ─────────────────────────────────────────────────────────


def test_quantity_vs_threshold_arb():
    markets = [gmarket(f"bin{n}", "How many goals will be scored?", str(n), f"bin{n}_yes", f"bin{n}_no")
               for n in range(0, 6)]
    markets.append(gmarket("o25", "Will there be over 2.5 goals?", "Over 2.5", "o25y", "o25n", smt="total", line=2.5))
    graph = infer_relations(discovery_from(event("fifwc-qat-che", markets)))
    rel = find_relation(graph, "goals_quantity_vs_threshold")
    assert rel is not None and rel.relation_type == QUANTITY and rel.safety == "clean"
    # legs: Over 2.5 NO + bins {3,4,5} YES
    token_set = set(rel.token_ids)
    assert "o25n" in token_set
    assert {"bin3_yes", "bin4_yes", "bin5_yes"} <= token_set
    assert "bin2_yes" not in token_set

    books = {"o25n": make_book("o25n", 0.30), "bin3_yes": make_book("bin3_yes", 0.25),
             "bin4_yes": make_book("bin4_yes", 0.12), "bin5_yes": make_book("bin5_yes", 0.10)}
    leg_books = [books[t] for t in rel.token_ids]
    sized = best_basket_size(leg_books, 1.0, loose_cfg(), max_capital=500.0)
    assert sized is not None and abs(sized["unit_cost"] - 0.77) < 1e-9
    assert validate_relation(rel, cost=sized["net_unit"]).ok


# ── state-space validator ─────────────────────────────────────────────────────────


def test_state_validator_rejects_unprofitable():
    ev = event("fifwc-qat-che", [
        gmarket("d", "draw?", "Draw", "draw_yes", "draw_no", smt="moneyline"),
        gmarket("a", "Qatar win?", "Qatar", "qat_yes", "qat_no", smt="moneyline"),
        gmarket("b", "Switzerland win?", "Switzerland", "che_yes", "che_no", smt="moneyline"),
    ])
    rel = find_relation(infer_relations(discovery_from(ev)), "moneyline_3way")
    assert validate_relation(rel, cost=0.95).ok          # cheap -> profitable
    assert not validate_relation(rel, cost=1.05).ok       # expensive -> rejected
    # synthesize gives one winning state per leg, each paying 1.0
    states = synthesize_states(rel)
    assert len(states) == 3
    assert all(abs(sum(s.values()) - 1.0) < 1e-9 for s in states)


def test_state_validator_btts_exact_basket():
    # buy BTTS-Yes + NO on exact 1-1, 2-2, 3-3
    legs = [
        OpportunityLeg("btts", "YES", "BTTS Yes", 0.0, 0.0),
        OpportunityLeg("ex11", "NO", "not 1-1", 0.0, 0.0),
        OpportunityLeg("ex22", "NO", "not 2-2", 0.0, 0.0),
        OpportunityLeg("ex33", "NO", "not 3-3", 0.0, 0.0),
    ]
    from src.worldcup.relations import RelationLeg
    rlegs = [RelationLeg(l.token_id, l.side, l.label) for l in legs]
    states = build_btts_exact_basket_states(
        "btts", {"ex11": (1, 1), "ex22": (2, 2), "ex33": (3, 3)},
    )
    # In a both-scored state other than the excluded exacts, payout = BTTS(1) + 3 NO(1 each) = 4.
    # In the worst case (one of the excluded exacts), payout = BTTS(1) + 2 surviving NO = 3.
    res_cheap = validate_legs(rlegs, states, cost=2.5)
    assert res_cheap.ok and res_cheap.min_payout == pytest.approx(3.0)
    res_rich = validate_legs(rlegs, states, cost=3.5)
    assert not res_rich.ok  # cost exceeds worst-case payout


# ── no false arb ──────────────────────────────────────────────────────────────────


def test_no_false_arb_when_cost_exceeds_payout():
    ev = event("fifwc-qat-che", [
        gmarket("d", "draw?", "Draw", "draw_yes", "draw_no", smt="moneyline"),
        gmarket("a", "Qatar win?", "Qatar", "qat_yes", "qat_no", smt="moneyline"),
        gmarket("b", "Switzerland win?", "Switzerland", "che_yes", "che_no", smt="moneyline"),
    ])
    rel = find_relation(infer_relations(discovery_from(ev)), "moneyline_3way")
    books = {"draw_yes": make_book("draw_yes", 0.40), "qat_yes": make_book("qat_yes", 0.35),
             "che_yes": make_book("che_yes", 0.35)}  # sum 1.10 > payout 1
    leg_books = [books[t] for t in rel.token_ids]
    assert best_basket_size(leg_books, 1.0, loose_cfg(), max_capital=500.0) is None


def test_no_arb_when_liquidity_too_thin():
    ev = event("fifwc-qat-che", [
        gmarket("o15", "over 1.5 goals?", "Over 1.5", "o15y", "o15n", smt="total", line=1.5),
        gmarket("o25", "over 2.5 goals?", "Over 2.5", "o25y", "o25n", smt="total", line=2.5),
    ])
    rel = find_relation(infer_relations(discovery_from(ev)), "total_goals_over_ladder")
    books = {"o15y": make_book("o15y", 0.60, depth=0.5), "o25n": make_book("o25n", 0.30, depth=0.5)}
    leg_books = [books[t] for t in rel.token_ids]
    assert best_basket_size(leg_books, 1.0, loose_cfg(min_liquidity=1.0), max_capital=500.0) is None


# ── exposure limiter ──────────────────────────────────────────────────────────────


def _opp(capital: float, subtype="moneyline_3way", teams=("Qatar",), slug="fifwc-qat-che") -> Opportunity:
    return Opportunity(
        name="x", relation_type=COMPLEMENT, relation_subtype=subtype,
        legs=[OpportunityLeg("t", "YES", "l", 0.3, 10.0)], guaranteed_payout=1.0,
        size=10.0, unit_cost=0.9, capital=capital, worst_case_profit=1.0, roi=0.1,
        edge_per_share=0.1, game_id="g", teams=teams, event_slugs=(slug,),
        confidence="high", safety="clean", reason="",
    )


def test_exposure_limiter_caps_and_kill_switch(tmp_path):
    ks = tmp_path / "kill"
    lim = ExposureLimiter(ExposureConfig(max_event_exposure=50.0, max_team_exposure=50.0,
                                         max_relation_exposure=1000.0, max_total_exposure=1000.0,
                                         max_daily_loss=100.0, kill_switch_path=ks))
    ok, _ = lim.can_enter(_opp(30.0))
    assert ok
    lim.commit(_opp(30.0))
    ok, reason = lim.can_enter(_opp(30.0))  # 30+30 > 50 event cap
    assert not ok and "event exposure" in reason
    # kill switch halts everything
    ks.write_text("halt", encoding="utf-8")
    ok, reason = lim.can_enter(_opp(1.0))
    assert not ok and "kill switch" in reason


# ── integration: emitted PairConfig resolves through the existing engine ──────────


def test_subset_relation_emits_engine_compatible_pair():
    pytest.importorskip("numpy")
    sim = pytest.importorskip("src.simulator")
    from src.worldcup.relations import relation_to_pair_dict
    ev = event("fifwc-qat-che", [
        gmarket("o15", "over 1.5 goals?", "Over 1.5", "o15y", "o15n", smt="total", line=1.5),
        gmarket("o25", "over 2.5 goals?", "Over 2.5", "o25y", "o25n", smt="total", line=2.5),
    ])
    rel = find_relation(infer_relations(discovery_from(ev)), "total_goals_over_ladder")
    pair_dict = relation_to_pair_dict(rel, enabled=True)
    pair = sim.PairConfig.from_dict(pair_dict)
    leg1, leg2 = sim._pair_leg_token_ids(pair)
    assert leg1 == "o15y"   # parent YES
    assert leg2 == "o25n"   # child NO
