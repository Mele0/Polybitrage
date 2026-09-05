from __future__ import annotations

import json
import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest

from src.audit import AuditConfig, ForensicAudit
from src.audit_report import generate_audit_report
from src.polymarket.models import OrderBook, OrderBookLevel
from src.simulator import PairConfig, PaperArbSimulator, SimulatorSettings


pytest.importorskip("pyarrow")


def _book(token: str, *, ask: float, bid: float, depth: float = 10.0) -> OrderBook:
    return OrderBook(
        asset_id=token,
        asks=[OrderBookLevel(price=Decimal(str(ask)), size=Decimal(str(depth)))],
        bids=[OrderBookLevel(price=Decimal(str(bid)), size=Decimal(str(depth)))],
    )


class _FakeProvider:
    def __init__(self, books: dict[str, OrderBook]):
        self.books = books
        self.stats = type(
            "Stats",
            (),
            {
                "unique_tokens_fetched": len(books),
                "cache_hits": 0,
                "failed_book_count": 0,
                "websocket_connected": False,
                "websocket_reconnect_count": 0,
                "fallback_to_polling_used": False,
                "token_update_count": 0,
                "event_triggered_recomputes": 0,
                "update_latency_ms": None,
            },
        )()

    async def get_books(self, token_ids):
        return {token: self.books.get(token) for token in token_ids}

    def book_age_ms(self, token_id):
        return 1.0

    def is_stale(self, token_id):
        return False


def test_forensic_audit_writes_parquet_datasets_and_report(tmp_path: Path):
    audit = ForensicAudit(AuditConfig(mode="forensic", audit_dir=tmp_path, run_id="audit-test"))
    audit.start(settings={"test": True})

    now_ns = time.time_ns()
    audit.record(
        "pair_observations",
        {
            "observation_id": "obs-1",
            "scan_id": "scan-1",
            "opportunity_id": "pair:PAIR",
            "timestamp": "2026-05-07T12:00:00+00:00",
            "pair_name": "PAIR",
            "candidate_type": "pair",
            "gross_total_cost": 0.99,
            "net_total_cost": 0.995,
            "entry_threshold": 1.0,
            "classification": "EXECUTABLE_ARBITRAGE_CANDIDATE",
            "spread_check_passed": True,
            "depth_check_passed": True,
            "edge_check_passed": True,
        },
    )
    audit.record(
        "decisions",
        {
            "decision_id": "decision-1",
            "scan_id": "scan-1",
            "opportunity_id": "pair:PAIR",
            "candidate_type": "pair",
            "candidate_name": "PAIR",
            "observation_id": "obs-1",
            "timestamp": "2026-05-07T12:00:00+00:00",
            "decision_wall_ns": now_ns,
            "decision_perf_ns": time.perf_counter_ns(),
            "outcome": "skipped",
            "skip_reason": "skipped: entry REST recheck no longer below threshold (PAIR)",
            "action": "skipped: entry REST recheck no longer below threshold (PAIR)",
            "passed_spread_check": True,
            "passed_depth_check": True,
            "passed_edge_check": True,
            "passed_capital_check": True,
            "submitted": False,
            "filled": False,
            "book_to_detection_ms": 12.0,
            "detection_to_decision_ms": 2.0,
            "decision_to_ack_ms": 15.0,
            "edge": 0.005,
            "gross_cost": 0.99,
            "net_cost": 0.995,
        },
    )
    audit.record(
        "missed_fills",
        {
            "missed_fill_id": "miss-1",
            "decision_id": "decision-1",
            "scan_id": "scan-1",
            "opportunity_id": "pair:PAIR",
            "candidate_name": "PAIR",
            "candidate_type": "pair",
            "classification": "rest_recheck_lost",
            "reason": "skipped: entry REST recheck no longer below threshold (PAIR)",
            "expected_profit": 0.05,
            "edge": 0.005,
            "gross_cost": 0.99,
            "net_cost": 0.995,
            "detected_ts_ns": now_ns,
            "decision_ts_ns": now_ns + 2_000_000,
            "market_activity_score": 0.6,
        },
    )
    audit.record(
        "timeline_events",
        {
            "timeline_event_id": "timeline-1",
            "event_kind": "decision",
            "scan_id": "scan-1",
            "decision_id": "decision-1",
            "opportunity_id": "pair:PAIR",
            "candidate_name": "PAIR",
            "timestamp_ns": now_ns,
            "perf_ns": time.perf_counter_ns(),
            "metric_1": 0.005,
            "metric_2": 15.0,
            "message": "decision",
        },
    )
    audit.close(status="completed")

    run_dir = tmp_path / "audit-test"
    assert (run_dir / "manifest.json").exists()
    assert any((run_dir / "decisions.parquet").glob("*.parquet"))

    summary = generate_audit_report(run_dir, top_missed=5, formats={"markdown", "json"})

    assert summary["opportunity_funnel"]["found"] == 1
    assert summary["miss_reasons"]["rest_recheck_lost"] == 1
    assert (run_dir / "report" / "summary.md").exists()
    assert json.loads((run_dir / "report" / "summary.json").read_text(encoding="utf-8"))["run_id"] == "audit-test"


def test_forensic_audit_records_scanner_decision_graph(tmp_path: Path):
    settings = SimulatorSettings(
        pairs_path=tmp_path / "pairs.yaml",
        budget=100,
        max_trade_size=10,
        capital_fraction_per_trade=0.1,
        entry_threshold=0.9975,
        trades_out=tmp_path / "trades.csv",
        audit_mode="forensic",
        audit_dir=tmp_path / "audit",
        audit_run_id="scan-test",
        audit_clock_sync_seconds=0,
        audit_network_probe_seconds=0,
        once=True,
    )
    sim = PaperArbSimulator(settings)
    pair = PairConfig(
        name="PAIR",
        parent_yes_token_id="P",
        child_no_token_id="C",
        overrides={"slippage_buffer": 0.0025, "estimated_fee_rate": 0.0},
    )

    sim._audit.start(settings=settings)
    row = asyncio.run(sim._scan_once([pair], _FakeProvider({"P": _book("P", ask=0.35, bid=0.34), "C": _book("C", ask=0.64, bid=0.63)})))
    sim._audit_finalize_open_windows()
    sim._audit.close(status="completed", settings=settings)

    assert row.action_taken.startswith("entered")
    summary = generate_audit_report(tmp_path / "audit" / "scan-test", top_missed=5, formats={"json"})
    assert summary["row_counts"]["book_snapshots"] == 2
    assert summary["row_counts"]["pair_observations"] == 1
    assert summary["row_counts"]["decisions"] == 1
    assert summary["opportunity_funnel"]["filled"] == 1
