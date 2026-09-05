from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pathlib import Path
from zoneinfo import ZoneInfo

from src.main import (
    _active_rollover_session,
    _date_slug_for_date,
    _format_rollover_path,
    _parse_et_wall_time,
    build_parser,
)
from src.discovery import _event_slug_candidates
from src.kalshi.client import kalshi_books_from_api, kalshi_token_id, parse_kalshi_token_id
from src.kalshi.discovery import same_market_complement_depth_summary, same_market_complement_pair
from src.polymarket.models import OrderBook, OrderBookLevel
from src.polymarket.clob_ws_client import ClobWebSocketClient
from src.simulator import (
    PairConfig,
    PaperArbSimulator,
    SimulatorSettings,
    _RunLock,
    _derive_range_threshold_n_leg_specs,
    estimated_fee_per_unit,
    resolve_missing_tokens,
)
from src.polymarket.live_trader import LiveOrderResult


def book(token: str, *, ask: float | None, bid: float | None, depth: float = 100.0) -> OrderBook:
    asks = [] if ask is None else [OrderBookLevel(price=Decimal(str(ask)), size=Decimal(str(depth)))]
    bids = [] if bid is None else [OrderBookLevel(price=Decimal(str(bid)), size=Decimal(str(depth)))]
    return OrderBook(asset_id=token, asks=asks, bids=bids)


class FakeProvider:
    def __init__(self, books):
        self.books = books

    async def start(self):
        return None

    async def stop(self):
        return None

    async def get_books(self, token_ids):
        self.stats = type("Stats", (), {"unique_tokens_fetched": len(set(token_ids)), "cache_hits": 0, "failed_book_count": 0, "websocket_connected": False, "websocket_reconnect_count": 0, "fallback_to_polling_used": False, "token_update_count": 0, "event_triggered_recomputes": 0, "update_latency_ms": None})()
        return {token: self.books.get(token) for token in token_ids}

    def book_age_ms(self, token_id):
        return 0.0

    def is_stale(self, token_id):
        return False


class FakeFeeClient:
    def __init__(self, rates):
        self.rates = rates

    async def get_fee_rates(self, token_ids, *, max_concurrent_requests=10):
        return {token: self.rates.get(token) for token in token_ids}


class FakeBatch:
    def __init__(self, books):
        self.books = books


class FakeLiveClob:
    def __init__(self, books):
        self.books = books
        self.requested_token_ids = []

    async def get_order_books(self, token_ids, *, max_concurrent_requests=10):
        self.requested_token_ids.append(list(token_ids))
        return FakeBatch({token: self.books.get(token) for token in token_ids})

    async def get_fee_rates(self, token_ids, *, max_concurrent_requests=10):
        return {token: 0.0 for token in token_ids}


class FakeLiveTrader:
    def __init__(self):
        self.legs = []

    async def buy_bundle_fok(self, legs):
        self.legs.append(legs)
        return LiveOrderResult(success=True, requested_notional=sum(leg.notional_cap for leg in legs), responses=[{"success": True}])


def test_fee_zero_by_default():
    assert estimated_fee_per_unit(0.5, 0.0) == 0.0


def test_fee_formula_matches_polymarket_per_share_formula():
    assert round(estimated_fee_per_unit(0.5, 0.072), 6) == 0.018
    assert round(estimated_fee_per_unit(0.01, 0.072), 7) == 0.0007128
    assert round(estimated_fee_per_unit(0.99, 0.072), 7) == 0.0007128


def test_observe_pair_classifies_executable_with_old_style_cost(tmp_path: Path):
    sim = PaperArbSimulator(SimulatorSettings(pairs_path=tmp_path / "pairs.yaml", once=True))
    pair = PairConfig(
        name="PAIR",
        parent_outcome_label="78,000",
        child_outcome_label="78,000-80,000",
        parent_yes_token_id="P",
        child_no_token_id="C",
        overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0},
    )
    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, {"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.64, bid=0.63)})
    assert obs.gross_total_cost == 0.99
    assert round(obs.net_total_cost or 0, 4) == 0.9925
    assert obs.classification == "EXECUTABLE_ARBITRAGE_CANDIDATE"


def test_scan_enters_one_paper_trade(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=0.1,
        entry_threshold=0.9975,
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(
        name="PAIR",
        parent_outcome_label="78,000",
        child_outcome_label="78,000-80,000",
        parent_yes_token_id="P",
        child_no_token_id="C",
        overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0},
    )
    clob = FakeProvider({"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.64, bid=0.63)})
    row = asyncio.run(sim._scan_once([pair], clob))
    assert row.executable_candidates_count == 1
    assert row.action_taken.startswith("entered guaranteed_arbitrage")
    assert len(sim.positions) == 1
    assert round(sim.positions[0].locked_capital, 6) == 10.0
    assert sim.cash < 100


def test_scan_skips_pair_below_min_roi_threshold(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=0.1,
        entry_threshold=0.9975,
        min_roi_threshold=0.01,
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(
        name="PAIR",
        parent_outcome_label="78,000",
        child_outcome_label="78,000-80,000",
        parent_yes_token_id="P",
        child_no_token_id="C",
        overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0},
    )

    row = asyncio.run(sim._scan_once([pair], FakeProvider({"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.64, bid=0.63)})))

    assert "below min ROI threshold" in row.action_taken
    assert not sim.positions


def test_duplicate_open_pair_is_skipped(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=0.1,
        entry_threshold=0.9975,
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C", overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0})
    clob = FakeProvider({"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.64, bid=0.63)})
    first = asyncio.run(sim._scan_once([pair], clob))
    second = asyncio.run(sim._scan_once([pair], clob))
    assert first.action_taken.startswith("entered")
    assert "duplicate open pair prevented" in second.action_taken
    assert len(sim.positions) == 1
    assert sim.open_pair_names == {"PAIR"}
    assert sim.open_count_by_pair["PAIR"] == 1


def test_missing_book_rejects_without_crash(tmp_path: Path):
    sim = PaperArbSimulator(SimulatorSettings(pairs_path=tmp_path / "pairs.yaml", once=True))
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C")
    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, {"P": book("P", ask=0.35, bid=0.34)})
    assert obs.classification == "REJECTED"
    assert obs.rejection_reason == "missing_order_book"


def test_kalshi_orderbook_bids_create_implied_asks():
    books = kalshi_books_from_api(
        "KXTEST-26JAN01",
        {
            "orderbook_fp": {
                "yes_dollars": [["0.4200", "13.00"]],
                "no_dollars": [["0.5600", "17.00"]],
            }
        },
    )

    yes = books["yes"]
    no = books["no"]
    assert yes.asset_id == "kalshi:KXTEST-26JAN01:yes"
    assert no.asset_id == "kalshi:KXTEST-26JAN01:no"
    assert yes.best_bid == pytest.approx(0.42)
    assert yes.best_ask == pytest.approx(0.44)
    assert no.best_bid == pytest.approx(0.56)
    assert no.best_ask == pytest.approx(0.58)
    assert yes.ask_depth == pytest.approx(17.0)
    assert no.ask_depth == pytest.approx(13.0)


def test_kalshi_complement_depth_requires_both_implied_asks():
    one_sided = kalshi_books_from_api(
        "KXTEST-26JAN01",
        {
            "orderbook_fp": {
                "yes_dollars": [],
                "no_dollars": [["0.5600", "17.00"]],
            }
        },
    )
    two_sided = kalshi_books_from_api(
        "KXTEST-26JAN01",
        {
            "orderbook_fp": {
                "yes_dollars": [["0.4200", "13.00"]],
                "no_dollars": [["0.5600", "17.00"]],
            }
        },
    )

    assert same_market_complement_depth_summary(one_sided)["has_complement_depth"] is False
    two_sided_summary = same_market_complement_depth_summary(two_sided)
    assert two_sided_summary["has_complement_depth"] is True
    assert two_sided_summary["yes_best_ask"] == 0.44
    assert two_sided_summary["no_best_ask"] == 0.58


def test_kalshi_token_resolution_from_market_tickers():
    pair = PairConfig(
        name="KALSHI_PAIR",
        parent_market_ticker="KXBTC-26JAN01-T80000",
        child_market_ticker="KXBTC-26JAN01-B80000",
    )

    asyncio.run(resolve_missing_tokens([pair], exchange="kalshi"))

    assert pair.parent_yes_token_id == kalshi_token_id("KXBTC-26JAN01-T80000", "yes")
    assert pair.parent_no_token_id == kalshi_token_id("KXBTC-26JAN01-T80000", "no")
    assert pair.child_yes_token_id == kalshi_token_id("KXBTC-26JAN01-B80000", "yes")
    assert pair.child_no_token_id == kalshi_token_id("KXBTC-26JAN01-B80000", "no")
    assert parse_kalshi_token_id(pair.parent_yes_token_id).ticker == "KXBTC-26JAN01-T80000"


def test_kalshi_same_market_discovery_pair_is_scanner_ready():
    row = same_market_complement_pair(
        {
            "ticker": "KXTEST-26JAN01",
            "event_ticker": "KXTEST",
            "title": "Example market",
            "yes_sub_title": "Yes",
            "no_sub_title": "No",
            "yes_ask_dollars": "0.4300",
            "no_ask_dollars": "0.5900",
            "volume_fp": "100.00",
        },
        enabled=True,
    )
    pair = PairConfig.from_dict(row)

    assert row["enabled"] is True
    assert row["relation"] == "same_market_complement"
    assert row["overrides"]["fee_category"] == "kalshi"
    assert pair.parent_yes_token_id == "kalshi:KXTEST-26JAN01:yes"
    assert pair.child_no_token_id == "kalshi:KXTEST-26JAN01:no"


def test_max_profit_sizing_stops_before_unprofitable_depth(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=1.0,
        entry_threshold=0.9975,
        sizing_mode="max_profit",
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C", overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0})
    parent = OrderBook(
        asset_id="P",
        asks=[OrderBookLevel(price=Decimal("0.35"), size=Decimal("5")), OrderBookLevel(price=Decimal("0.50"), size=Decimal("10"))],
        bids=[OrderBookLevel(price=Decimal("0.34"), size=Decimal("20"))],
    )
    child = OrderBook(
        asset_id="C",
        asks=[OrderBookLevel(price=Decimal("0.60"), size=Decimal("20"))],
        bids=[OrderBookLevel(price=Decimal("0.59"), size=Decimal("20"))],
    )
    row = asyncio.run(sim._scan_once([pair], FakeProvider({"P": parent, "C": child})))
    assert row.action_taken.startswith("entered guaranteed_arbitrage")
    assert row.best_optimal_size is not None
    assert round(sim.positions[0].size, 6) == 5
    assert sim.positions[0].parent_yes_entry_price == 0.35
    assert sim.positions[0].child_no_entry_price == 0.60


def test_close_when_edge_converges_recycles_cash(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=1.0,
        entry_threshold=0.9975,
        exit_mode="close_when_edge_converges",
        take_profit_pct=0.001,
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C", overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0})
    first_books = {"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.64, bid=0.63)}
    asyncio.run(sim._scan_once([pair], FakeProvider(first_books)))
    assert sim.positions[0].status == "open"
    second_books = {"P": book("P", ask=0.35, bid=0.50), "C": book("C", ask=0.64, bid=0.60)}
    asyncio.run(sim._scan_once([pair], FakeProvider(second_books)))
    assert sim.positions[0].status == "closed"
    assert sim.positions[0].realized_pnl > 0
    assert sim.positions[0].exit_fee_total == 0.0
    assert sim.positions[0].liquidation_value_net == sim.positions[0].liquidation_value_gross
    assert sim.cash > 100
    assert "PAIR" not in sim.open_pair_names
    assert sim.open_count_by_pair == {}


def test_guaranteed_arb_can_have_negative_liquidation_pnl(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=1.0,
        entry_threshold=0.9975,
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C", overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0})

    asyncio.run(sim._scan_once([pair], FakeProvider({"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.64, bid=0.63)})))
    assert sim.positions[0].worst_case_profit > 0

    asyncio.run(sim._scan_once([pair], FakeProvider({"P": book("P", ask=0.35, bid=0.30), "C": book("C", ask=0.64, bid=0.60)})))
    assert sim.positions[0].status == "open"
    assert sim.positions[0].liquidation_pnl < 0
    assert sim.positions[0].unrealized_pnl == sim.positions[0].liquidation_pnl
    assert sim.positions[0].worst_case_profit > 0


def test_close_when_edge_converges_deducts_exit_fees(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=1.0,
        entry_threshold=0.99,
        exit_mode="close_when_edge_converges",
        take_profit_pct=0.001,
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C", overrides={"slippage_buffer": 0.0, "estimated_fee_rate": 0.072})

    asyncio.run(sim._scan_once([pair], FakeProvider({"P": book("P", ask=0.30, bid=0.29), "C": book("C", ask=0.60, bid=0.59)})))
    locked = sim.positions[0].locked_capital
    asyncio.run(sim._scan_once([pair], FakeProvider({"P": book("P", ask=0.30, bid=0.55), "C": book("C", ask=0.60, bid=0.50)})))

    position = sim.positions[0]
    assert position.status == "closed"
    assert position.exit_fee_total > 0
    assert position.liquidation_value_net == position.liquidation_value_gross - position.exit_fee_total
    assert position.realized_pnl == position.liquidation_value_net - locked
    assert position.realized_pnl < (position.liquidation_value_gross - locked)


def test_boundary_ambiguous_candidate_not_labeled_guaranteed(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=0.1,
        entry_threshold=0.9975,
        out=tmp_path / "scan.csv",
        trades_out=tmp_path / "trades.csv",
        save_markdown=tmp_path / "report.md",
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(
        name="PAIR",
        parent_yes_token_id="P",
        child_no_token_id="C",
        boundary_ambiguity=True,
        relation_safety="boundary_ambiguous",
        overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0},
    )
    asyncio.run(sim._scan_once([pair], FakeProvider({"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.64, bid=0.63)})))
    assert sim.positions[0].entry_trade_type == "boundary_ambiguous_candidate"


def test_relation_safety_filter_excludes_boundary_ambiguous(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        relation_safety="clean",
    )
    sim = PaperArbSimulator(settings)
    clean = PairConfig(name="CLEAN", parent_yes_token_id="P", child_no_token_id="C", relation_safety="clean")
    ambiguous = PairConfig(
        name="AMBIG",
        parent_yes_token_id="P",
        child_no_token_id="C",
        relation_safety="boundary_ambiguous",
        boundary_ambiguity=True,
    )
    assert [pair.name for pair in sim._filter_pairs([clean, ambiguous])] == ["CLEAN"]


def test_dynamic_fee_rates_override_yaml_zero_when_available(tmp_path: Path):
    settings = SimulatorSettings(pairs_path=tmp_path / "pairs.yaml", once=True)
    sim = PaperArbSimulator(settings)
    pair = PairConfig(
        name="PAIR",
        parent_yes_token_id="P",
        child_no_token_id="C",
        overrides={"slippage_buffer": 0.0, "estimated_fee_rate": 0.0},
    )
    asyncio.run(sim._prepare_pair_caches([pair], FakeFeeClient({"P": 0.072, "C": 0.072})))
    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, {"P": book("P", ask=0.5, bid=0.49), "C": book("C", ask=0.49, bid=0.48)})
    assert round(obs.estimated_fee_total_per_unit, 6) == round(0.072 * 0.5 * 0.5 + 0.072 * 0.49 * 0.51, 6)
    assert obs.net_total_cost > obs.gross_total_cost


def test_yaml_min_edge_threshold_is_enforced(tmp_path: Path):
    sim = PaperArbSimulator(SimulatorSettings(pairs_path=tmp_path / "pairs.yaml", once=True))
    pair = PairConfig(
        name="PAIR",
        parent_yes_token_id="P",
        child_no_token_id="C",
        overrides={"slippage_buffer": 0.0, "estimated_fee_rate": 0.0, "min_edge_threshold": 0.005},
    )
    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, {"P": book("P", ask=0.5, bid=0.49), "C": book("C", ask=0.498, bid=0.49)})
    assert round(obs.net_total_cost or 0, 4) == 0.998
    assert obs.classification != "EXECUTABLE_ARBITRAGE_CANDIDATE"


def test_yaml_max_spread_per_leg_is_enforced(tmp_path: Path):
    sim = PaperArbSimulator(SimulatorSettings(pairs_path=tmp_path / "pairs.yaml", once=True))
    pair = PairConfig(
        name="PAIR",
        parent_yes_token_id="P",
        child_no_token_id="C",
        overrides={"slippage_buffer": 0.0, "estimated_fee_rate": 0.0, "max_spread_per_leg": 0.01},
    )
    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, {"P": book("P", ask=0.5, bid=0.45), "C": book("C", ask=0.49, bid=0.48)})
    assert obs.classification == "REJECTED"
    assert obs.rejection_reason == "spread_too_wide"


def test_generic_trade_template_uses_parent_no_and_child_no(tmp_path: Path):
    sim = PaperArbSimulator(SimulatorSettings(pairs_path=tmp_path / "pairs.yaml", once=True))
    pair = PairConfig(
        name="PAIR",
        parent_yes_token_id="PY",
        parent_no_token_id="PN",
        child_yes_token_id="CY",
        child_no_token_id="CN",
        trade_template={
            "leg_1": {"market": "parent", "outcome": "NO", "side": "BUY"},
            "leg_2": {"market": "child", "outcome": "NO", "side": "BUY"},
        },
        overrides={"slippage_buffer": 0.0, "estimated_fee_rate": 0.0},
    )
    books = {
        "PY": book("PY", ask=0.9, bid=0.89),
        "PN": book("PN", ask=0.1, bid=0.09),
        "CY": book("CY", ask=0.2, bid=0.19),
        "CN": book("CN", ask=0.79, bid=0.78),
    }
    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, books)
    assert obs.gross_total_cost == 0.89
    assert obs.classification == "EXECUTABLE_ARBITRAGE_CANDIDATE"


def test_websocket_best_bid_ask_updates_cached_top_levels():
    client = ClobWebSocketClient()
    client.seed_books({"T": book("T", ask=0.60, bid=0.40, depth=12)})

    updated = client._handle_message(
        {
            "event_type": "best_bid_ask",
            "asset_id": "T",
            "best_bid": "0.45",
            "best_ask": "0.55",
        }
    )

    cached = client.get_book("T")
    assert updated == {"T"}
    assert cached is not None
    assert cached.best_bid == pytest.approx(0.45)
    assert cached.best_ask == pytest.approx(0.55)
    assert cached.bid_depth == pytest.approx(12.0)
    assert cached.ask_depth == pytest.approx(12.0)
    assert cached.raw_json["best_bid_ask_depth_untrusted"] is True


def test_untrusted_websocket_depth_suppresses_optimistic_optimal_size(tmp_path: Path):
    client = ClobWebSocketClient()
    client.seed_books({"P": book("P", ask=0.60, bid=0.59, depth=12), "C": book("C", ask=0.40, bid=0.39, depth=12)})
    client._handle_message({"event_type": "best_bid_ask", "asset_id": "P", "best_bid": "0.59", "best_ask": "0.35"})
    client._handle_message({"event_type": "best_bid_ask", "asset_id": "C", "best_bid": "0.39", "best_ask": "0.60"})
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            sizing_mode="max_profit",
            entry_threshold=1.0,
            once=True,
        )
    )
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C", overrides={"slippage_buffer": 0.0, "estimated_fee_rate": 0.0})

    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, {"P": client.get_book("P"), "C": client.get_book("C")})

    assert obs.net_total_cost == 0.95
    assert obs.optimal_size is None
    assert obs.optimal_guaranteed_profit is None


def test_trusted_rest_books_still_produce_optimal_size(tmp_path: Path):
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            sizing_mode="max_profit",
            entry_threshold=1.0,
            max_trade_size=10,
            once=True,
        )
    )
    pair = PairConfig(name="PAIR", parent_yes_token_id="P", child_no_token_id="C", overrides={"slippage_buffer": 0.0, "estimated_fee_rate": 0.0})

    obs = sim._observe_pair("2026-04-26T12:00:00+00:00", pair, {"P": book("P", ask=0.35, bid=0.34), "C": book("C", ask=0.60, bid=0.59)})

    assert obs.optimal_size is not None
    assert obs.optimal_guaranteed_profit is not None


def test_n_leg_range_windows_include_contiguous_multi_range_opportunity(tmp_path: Path):
    pairs = [
        range_threshold_pair("P70_R70_72", "70,000", "70,000-72,000", "Y70", "N70", "R70_72"),
        range_threshold_pair("P72_R72_74", "72,000", "72,000-74,000", "Y72", "N72", "R72_74"),
        range_threshold_pair("P74_R72_74", "74,000", "72,000-74,000", "Y74", "N74", "R72_74"),
    ]

    specs = _derive_range_threshold_n_leg_specs(pairs, max_ranges=2)
    wide = next(spec for spec in specs if spec.name == "bitcoin-price-on-may-4 70000-74000 via 2 ranges")

    assert [leg.label for leg in wide.legs] == ["70000_YES", "74000_NO", "70000_72000_NO", "72000_74000_NO"]
    assert [leg.token_id for leg in wide.legs] == ["Y70", "N74", "R70_72", "R72_74"]
    assert wide.guaranteed_payout == 3.0


def test_best_n_leg_opportunity_scores_multi_range_window(tmp_path: Path):
    pairs = [
        range_threshold_pair("P70_R70_72", "70,000", "70,000-72,000", "Y70", "N70", "R70_72"),
        range_threshold_pair("P72_R72_74", "72,000", "72,000-74,000", "Y72", "N72", "R72_74"),
        range_threshold_pair("P74_R72_74", "74,000", "72,000-74,000", "Y74", "N74", "R72_74"),
    ]
    specs = _derive_range_threshold_n_leg_specs(pairs, max_ranges=2)
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            n_leg_max_ranges=2,
            max_trade_size=20,
            once=True,
        )
    )
    books = {
        "Y70": book("Y70", ask=0.30, bid=0.29),
        "N74": book("N74", ask=0.30, bid=0.29),
        "R70_72": book("R70_72", ask=0.30, bid=0.29),
        "R72_74": book("R72_74", ask=0.30, bid=0.29),
        "Y72": book("Y72", ask=0.90, bid=0.89),
        "N72": book("N72", ask=0.90, bid=0.89),
    }

    best = sim._best_n_leg_opportunity(specs, books)

    assert best["name"] == "bitcoin-price-on-may-4 70000-74000 via 2 ranges"
    assert best["leg_count"] == 4
    assert best["guaranteed_payout"] == 3.0
    assert round(best["gross_cost"], 6) == 1.2


def test_n_leg_skips_untrusted_websocket_depth_without_rest_recheck(tmp_path: Path):
    pairs = [
        range_threshold_pair("P70_R70_72", "70,000", "70,000-72,000", "Y70", "N70", "R70_72"),
        range_threshold_pair("P72_R70_72", "72,000", "70,000-72,000", "Y72", "N72", "R70_72"),
    ]
    specs = _derive_range_threshold_n_leg_specs(pairs, max_ranges=1)
    books = {
        "Y70": book("Y70", ask=0.30, bid=0.29),
        "N72": book("N72", ask=0.30, bid=0.29),
        "R70_72": book("R70_72", ask=0.30, bid=0.29),
    }
    for order_book in books.values():
        order_book.raw_json["best_bid_ask_depth_untrusted"] = True
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            n_leg_max_ranges=1,
            max_trade_size=1000,
            min_trade_size=1,
            once=True,
        )
    )
    best = sim._best_n_leg_opportunity(specs, books)

    action = asyncio.run(sim._try_enter_n_leg(best, books))

    assert action.startswith("skipped: untrusted websocket N-leg depth")
    assert sim.positions == []
    assert sim.cash == sim.settings.budget


def test_n_leg_rest_recheck_resizes_from_trusted_depth(tmp_path: Path):
    pairs = [
        range_threshold_pair("P70_R70_72", "70,000", "70,000-72,000", "Y70", "N70", "R70_72"),
        range_threshold_pair("P72_R70_72", "72,000", "70,000-72,000", "Y72", "N72", "R70_72"),
    ]
    specs = _derive_range_threshold_n_leg_specs(pairs, max_ranges=1)
    websocket_books = {
        "Y70": book("Y70", ask=0.25, bid=0.24, depth=1000),
        "N72": book("N72", ask=0.25, bid=0.24, depth=1000),
        "R70_72": book("R70_72", ask=0.25, bid=0.24, depth=1000),
    }
    for order_book in websocket_books.values():
        order_book.raw_json["best_bid_ask_depth_untrusted"] = True
    trusted_books = {
        token: OrderBook(
            asset_id=token,
            asks=[
                OrderBookLevel(price=Decimal("0.30"), size=Decimal("2")),
                OrderBookLevel(price=Decimal("0.80"), size=Decimal("100")),
            ],
        )
        for token in ["Y70", "N72", "R70_72"]
    }
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            n_leg_max_ranges=1,
            max_trade_size=1000,
            min_trade_size=1,
            trades_out=tmp_path / "trades.csv",
            once=True,
        )
    )
    fake_clob = FakeLiveClob(trusted_books)
    sim._clob_client = fake_clob
    best = sim._best_n_leg_opportunity(specs, websocket_books)

    action = asyncio.run(sim._try_enter_n_leg(best, websocket_books))

    assert action.startswith("entered 3-leg")
    assert fake_clob.requested_token_ids == [["Y70", "N72", "R70_72"]]
    assert len(sim.positions) == 1
    assert round(sim.positions[0].size, 6) == 2.0
    assert round(sim.positions[0].locked_capital, 6) == 1.8


def test_scan_records_n_leg_entry_and_duplicate_skip_actions(tmp_path: Path):
    pairs = [
        range_threshold_pair("P70_R70_72", "70,000", "70,000-72,000", "Y70", "N70", "R70_72"),
        range_threshold_pair("P72_R70_72", "72,000", "70,000-72,000", "Y72", "N72", "R70_72"),
    ]
    books = {
        "Y70": book("Y70", ask=0.60, bid=0.59, depth=20),
        "N72": book("N72", ask=0.70, bid=0.69, depth=20),
        "R70_72": book("R70_72", ask=0.69, bid=0.68, depth=20),
    }
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            enable_n_leg_trading=True,
            n_leg_max_ranges=1,
            max_trade_size=100,
            min_trade_size=1,
            entry_rest_recheck=False,
            trades_out=tmp_path / "trades.csv",
            once=True,
        )
    )

    first = asyncio.run(sim._scan_once(pairs, FakeProvider(books)))
    second = asyncio.run(sim._scan_once(pairs, FakeProvider(books)))

    assert first.action_taken.startswith("entered 3-leg")
    assert "duplicate open N-leg prevented" in second.action_taken
    assert len(sim.positions) == 1


def test_incremental_n_leg_enters_when_residual_depth_available(tmp_path: Path):
    """With allow_incremental_n_leg=True a second scan can top-up an open N-leg."""
    pairs = [
        range_threshold_pair("P70_R70_72", "70,000", "70,000-72,000", "Y70", "N70", "R70_72"),
        range_threshold_pair("P72_R70_72", "72,000", "70,000-72,000", "Y72", "N72", "R70_72"),
    ]
    # Shallow depth on first scan (5 units), deep depth on second (200 units).
    # The optimizer should suggest ~5 units on scan 1 and ~200 on scan 2, so the
    # incremental size on scan 2 is large enough to trigger a second entry.
    books_shallow = {
        "Y70": book("Y70", ask=0.60, bid=0.59, depth=5),
        "N72": book("N72", ask=0.70, bid=0.69, depth=5),
        "R70_72": book("R70_72", ask=0.69, bid=0.68, depth=5),
    }
    books_deep = {
        "Y70": book("Y70", ask=0.60, bid=0.59, depth=200),
        "N72": book("N72", ask=0.70, bid=0.69, depth=200),
        "R70_72": book("R70_72", ask=0.69, bid=0.68, depth=200),
    }
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            enable_n_leg_trading=True,
            n_leg_max_ranges=1,
            max_trade_size=1000,
            min_trade_size=1,
            budget=10_000,
            entry_rest_recheck=False,
            trades_out=tmp_path / "trades.csv",
            once=True,
            allow_incremental_n_leg=True,
            incremental_n_leg_min_profit=0.01,
        )
    )

    first = asyncio.run(sim._scan_once(pairs, FakeProvider(books_shallow)))
    assert first.action_taken.startswith("entered 3-leg"), f"Expected entry, got: {first.action_taken}"
    assert len(sim.positions) == 1
    first_size = sim.positions[0].size

    second = asyncio.run(sim._scan_once(pairs, FakeProvider(books_deep)))
    assert second.action_taken.startswith("incremental 3-leg"), f"Expected incremental, got: {second.action_taken}"
    assert len(sim.positions) == 2
    assert sim.positions[1].size > 0, "Incremental size should be positive"


def test_incremental_n_leg_blocked_when_flag_off(tmp_path: Path):
    """Without allow_incremental_n_leg, a second scan skips as duplicate."""
    pairs = [
        range_threshold_pair("P70_R70_72", "70,000", "70,000-72,000", "Y70", "N70", "R70_72"),
        range_threshold_pair("P72_R70_72", "72,000", "70,000-72,000", "Y72", "N72", "R70_72"),
    ]
    books = {
        "Y70": book("Y70", ask=0.60, bid=0.59, depth=200),
        "N72": book("N72", ask=0.70, bid=0.69, depth=200),
        "R70_72": book("R70_72", ask=0.69, bid=0.68, depth=200),
    }
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            enable_n_leg_trading=True,
            n_leg_max_ranges=1,
            max_trade_size=1000,
            min_trade_size=1,
            budget=10_000,
            entry_rest_recheck=False,
            trades_out=tmp_path / "trades.csv",
            once=True,
            allow_incremental_n_leg=False,  # default
        )
    )

    asyncio.run(sim._scan_once(pairs, FakeProvider(books)))
    second = asyncio.run(sim._scan_once(pairs, FakeProvider(books)))

    assert "duplicate open N-leg prevented" in second.action_taken
    assert len(sim.positions) == 1


def test_run_lock_blocks_second_owner_and_removes_stale_lock(tmp_path: Path):
    lock_path = tmp_path / "bot.lock"
    with _RunLock(lock_path):
        try:
            with _RunLock(lock_path):
                raise AssertionError("second lock unexpectedly acquired")
        except SystemExit as exc:
            assert "Another bot process appears to be running" in str(exc)
    lock_path.write_text("999999999\n", encoding="utf-8")
    with _RunLock(lock_path):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_live_bundle_uses_marginal_ask_limit_and_tick_buffer(tmp_path: Path):
    sim = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=tmp_path / "pairs.yaml",
            execution_mode="live",
            live_orders_out=tmp_path / "live.csv",
            live_price_buffer_ticks=1,
            fee_rate=0.0,
        )
    )
    first = OrderBook(
        asset_id="A",
        asks=[
            OrderBookLevel(price=Decimal("0.40"), size=Decimal("1")),
            OrderBookLevel(price=Decimal("0.41"), size=Decimal("5")),
        ],
        raw_json={"tick_size": "0.01", "min_order_size": "1", "neg_risk": False},
    )
    second = OrderBook(
        asset_id="B",
        asks=[OrderBookLevel(price=Decimal("0.50"), size=Decimal("5"))],
        raw_json={"tick_size": "0.01", "min_order_size": "1", "neg_risk": False},
    )
    fake_trader = FakeLiveTrader()
    sim._live_trader = fake_trader
    sim._clob_client = FakeLiveClob({"A": first, "B": second})

    result = asyncio.run(
        sim._execute_live_buy_bundle(
            timestamp="2026-04-26T12:00:00+00:00",
            strategy_name="LIVE_TEST",
            legs=[("A", "A", first), ("B", "B", second)],
            size=2.0,
            guaranteed_payout=1.0,
        )
    )

    assert result.success is True
    sent = fake_trader.legs[0]
    assert [leg.limit_price for leg in sent] == [0.42, 0.51]
    assert round(result.requested_notional, 6) == 1.86


def test_rollover_session_uses_current_date_before_et_close_delay():
    et = ZoneInfo("America/New_York")
    session_date, switch_at = _active_rollover_session(
        datetime(2026, 5, 1, 11, 59, tzinfo=et),
        _parse_et_wall_time("12:00"),
        timedelta(minutes=2),
    )

    assert _date_slug_for_date(session_date) == "may-1"
    assert switch_at == datetime(2026, 5, 1, 12, 2, tzinfo=et)


def test_rollover_session_switches_to_next_date_after_et_close_delay():
    et = ZoneInfo("America/New_York")
    session_date, switch_at = _active_rollover_session(
        datetime(2026, 5, 1, 12, 3, tzinfo=et),
        _parse_et_wall_time("12:00"),
        timedelta(minutes=2),
    )

    assert _date_slug_for_date(session_date) == "may-2"
    assert switch_at == datetime(2026, 5, 2, 12, 2, tzinfo=et)


def test_rollover_paths_accept_date_placeholders():
    path = _format_rollover_path(
        Path("reports/crypto_{date_slug}_{date_compact}_{date_iso}.csv"),
        datetime(2026, 5, 1).date(),
    )

    assert path == Path("reports/crypto_may-1_20260501_2026-05-01.csv")


def test_polymarket_daily_event_slug_candidates_prefer_year_suffix():
    assert _event_slug_candidates("bitcoin", "above", "june-6", date_year=2026) == [
        "bitcoin-above-on-june-6-2026",
        "bitcoin-above-on-june-6",
    ]
    assert _event_slug_candidates("bitcoin", "above", "june-6-2026", date_year=2026) == [
        "bitcoin-above-on-june-6-2026",
    ]


def test_cli_help_explains_commands_and_entrypoint():
    help_text = build_parser().format_help()

    assert "poly-arb --help" in help_text
    assert "paper-arb-sim" in help_text
    assert "discover-tomorrow-crypto-pairs" in help_text
    assert "audit-report" in help_text
    assert "audit-profit-report" in help_text
    assert "continuous rollover" in help_text

    parser = build_parser()
    args = parser.parse_args(["--min-roi-threshold", "0.01", "paper-arb-sim", "--once"])
    assert args.global_min_roi_threshold == 0.01


def range_threshold_pair(
    name: str,
    threshold_label: str,
    range_label: str,
    parent_yes_token_id: str,
    parent_no_token_id: str,
    child_no_token_id: str,
) -> PairConfig:
    return PairConfig(
        name=name,
        parent_market_slug=f"above-{threshold_label}",
        child_market_slug=f"range-{range_label}",
        parent_outcome_label=threshold_label,
        child_outcome_label=range_label,
        parent_yes_token_id=parent_yes_token_id,
        parent_no_token_id=parent_no_token_id,
        child_no_token_id=child_no_token_id,
        raw={
            "parent_event_slug": "bitcoin-above-on-may-4",
            "child_event_slug": "bitcoin-price-on-may-4",
        },
    )
