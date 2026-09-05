from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.audit_profit_report import AuditProfitReportConfig, generate_audit_profit_report


def _write_dataset(run_dir: Path, table: str, rows: list[dict]):
    path = run_dir / f"{table}.parquet"
    path.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path / "part-000001.parquet")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_by(rows: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    for row in rows:
        if row[column] == value:
            return row
    raise AssertionError(f"No row where {column}={value!r}: {rows}")


def test_audit_profit_report_outputs_profit_forensics_and_dedupes_windows(tmp_path: Path):
    run_dir = tmp_path / "audit-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "audit-run", "status": "completed"}), encoding="utf-8")

    _write_dataset(
        run_dir,
        "opportunity_windows",
        [
            {
                "window_id": "w-dup",
                "opportunity_id": "n_leg:dup",
                "candidate_name": "duplicate range",
                "candidate_type": "n_leg",
                "first_seen_scan_id": "s1",
                "last_seen_scan_id": "s1",
                "first_seen_ns": 1_000_000_000,
                "last_seen_ns": 1_000_000_000,
                "duration_ms": 0.0,
                "scan_count": 1,
                "best_edge": 0.01,
                "status": "active",
            },
            {
                "window_id": "w-dup",
                "opportunity_id": "n_leg:dup",
                "candidate_name": "duplicate range",
                "candidate_type": "n_leg",
                "first_seen_scan_id": "s1",
                "last_seen_scan_id": "s2",
                "first_seen_ns": 1_000_000_000,
                "last_seen_ns": 1_100_000_000,
                "duration_ms": 100.0,
                "scan_count": 2,
                "best_edge": 0.01,
                "status": "active",
            },
            {
                "window_id": "w-depth",
                "opportunity_id": "n_leg:depth",
                "candidate_name": "depth depleted range",
                "candidate_type": "n_leg",
                "first_seen_scan_id": "s3",
                "last_seen_scan_id": "s3",
                "first_seen_ns": 2_000_000_000,
                "last_seen_ns": 2_250_000_000,
                "duration_ms": 250.0,
                "scan_count": 1,
                "best_edge": 0.02,
                "status": "closed",
                "close_reason": "no_longer_executable",
            },
            {
                "window_id": "w-stale",
                "opportunity_id": "n_leg:stale",
                "candidate_name": "stale range",
                "candidate_type": "n_leg",
                "first_seen_scan_id": "s4",
                "last_seen_scan_id": "s4",
                "first_seen_ns": 3_000_000_000,
                "last_seen_ns": 3_500_000_000,
                "duration_ms": 500.0,
                "scan_count": 1,
                "best_edge": 0.015,
                "status": "closed",
                "close_reason": "rest_recheck_missing",
            },
        ],
    )
    _write_dataset(
        run_dir,
        "decisions",
        [
            {
                "decision_id": "d-filled",
                "scan_id": "s1",
                "opportunity_id": "n_leg:dup",
                "candidate_type": "n_leg",
                "candidate_name": "duplicate range",
                "n_leg_candidate_id": "n-dup-1",
                "decision_wall_ns": 1_000_000_000,
                "decision_perf_ns": 1_000_000,
                "outcome": "filled",
                "filled": True,
                "submitted": False,
                "passed_capital_check": True,
                "book_to_detection_ms": 25.0,
                "detection_to_decision_ms": 5.0,
                "decision_to_ack_ms": 1.0,
                "edge": 0.01,
                "size": 10.0,
                "locked_capital": 20.0,
            },
            {
                "decision_id": "d-dup",
                "scan_id": "s2",
                "opportunity_id": "n_leg:dup",
                "candidate_type": "n_leg",
                "candidate_name": "duplicate range",
                "n_leg_candidate_id": "n-dup-2",
                "decision_wall_ns": 1_100_000_000,
                "decision_perf_ns": 1_050_000,
                "outcome": "skipped",
                "skip_reason": "skipped: duplicate open N-leg prevented (duplicate range)",
                "action": "skipped: duplicate open N-leg prevented (duplicate range)",
                "filled": False,
                "submitted": False,
                "passed_capital_check": True,
                "book_to_detection_ms": 30.0,
                "detection_to_decision_ms": 4.0,
                "decision_to_ack_ms": 0.1,
                "edge": 0.01,
                "size": 10.0,
            },
            {
                "decision_id": "d-depth",
                "scan_id": "s3",
                "opportunity_id": "n_leg:depth",
                "candidate_type": "n_leg",
                "candidate_name": "depth depleted range",
                "n_leg_candidate_id": "n-depth",
                "decision_wall_ns": 2_250_000_000,
                "decision_perf_ns": 2_000_000,
                "outcome": "skipped",
                "skip_reason": "depth depleted",
                "filled": False,
                "submitted": False,
                "passed_capital_check": True,
                "book_to_detection_ms": 40.0,
                "detection_to_decision_ms": 20.0,
                "decision_to_ack_ms": 0.1,
                "edge": 0.02,
            },
            {
                "decision_id": "d-stale",
                "scan_id": "s4",
                "opportunity_id": "n_leg:stale",
                "candidate_type": "n_leg",
                "candidate_name": "stale range",
                "n_leg_candidate_id": "n-stale",
                "decision_wall_ns": 3_500_000_000,
                "decision_perf_ns": 3_000_000,
                "outcome": "skipped",
                "skip_reason": "N-leg REST recheck missing quote",
                "filled": False,
                "submitted": False,
                "passed_capital_check": True,
                "book_to_detection_ms": 6000.0,
                "detection_to_decision_ms": 10.0,
                "decision_to_ack_ms": 0.1,
                "edge": 0.015,
            },
        ],
    )
    _write_dataset(
        run_dir,
        "missed_fills",
        [
            {
                "missed_fill_id": "m-dup",
                "decision_id": "d-dup",
                "scan_id": "s2",
                "opportunity_id": "n_leg:dup",
                "candidate_name": "duplicate range",
                "candidate_type": "n_leg",
                "classification": "duplicate_guard_blocked",
                "reason": "skipped: duplicate open N-leg prevented",
                "expected_profit": 5.0,
                "edge": 0.01,
                "detected_ts_ns": 1_050_000_000,
                "decision_ts_ns": 1_100_000_000,
                "market_activity_score": 0.2,
            },
            {
                "missed_fill_id": "m-depth",
                "decision_id": "d-depth",
                "scan_id": "s3",
                "opportunity_id": "n_leg:depth",
                "candidate_name": "depth depleted range",
                "candidate_type": "n_leg",
                "classification": "depth_depleted",
                "reason": "depth depleted before submit",
                "expected_profit": 7.0,
                "edge": 0.02,
                "detected_ts_ns": 2_000_000_000,
                "decision_ts_ns": 2_250_000_000,
                "market_activity_score": 0.9,
            },
            {
                "missed_fill_id": "m-stale",
                "decision_id": "d-stale",
                "scan_id": "s4",
                "opportunity_id": "n_leg:stale",
                "candidate_name": "stale range",
                "candidate_type": "n_leg",
                "classification": "rest_recheck_missing",
                "reason": "REST recheck missing quote",
                "expected_profit": 6.0,
                "edge": 0.015,
                "detected_ts_ns": 3_000_000_000,
                "decision_ts_ns": 3_500_000_000,
                "market_activity_score": 0.1,
            },
        ],
    )
    _write_dataset(
        run_dir,
        "n_leg_candidates",
        [
            {
                "n_leg_candidate_id": "n-dup-1",
                "scan_id": "s1",
                "opportunity_id": "n_leg:dup",
                "name": "duplicate range",
                "leg_count": 3,
                "gross_edge": 0.01,
                "optimal_size": 10.0,
                "optimal_profit": 5.0,
                "classification": "EXECUTABLE_ARBITRAGE_CANDIDATE",
                "optimizer_ms": 1.0,
            },
            {
                "n_leg_candidate_id": "n-dup-2",
                "scan_id": "s2",
                "opportunity_id": "n_leg:dup",
                "name": "duplicate range",
                "leg_count": 3,
                "gross_edge": 0.01,
                "optimal_size": 10.0,
                "optimal_profit": 5.0,
                "classification": "EXECUTABLE_ARBITRAGE_CANDIDATE",
                "optimizer_ms": 1.1,
            },
            {
                "n_leg_candidate_id": "n-depth",
                "scan_id": "s3",
                "opportunity_id": "n_leg:depth",
                "name": "depth depleted range",
                "leg_count": 3,
                "gross_edge": 0.02,
                "optimal_size": 20.0,
                "optimal_profit": 7.0,
                "classification": "EXECUTABLE_ARBITRAGE_CANDIDATE",
                "optimizer_ms": 2.0,
            },
            {
                "n_leg_candidate_id": "n-stale",
                "scan_id": "s4",
                "opportunity_id": "n_leg:stale",
                "name": "stale range",
                "leg_count": 3,
                "gross_edge": 0.015,
                "optimal_size": 20.0,
                "optimal_profit": 6.0,
                "classification": "EXECUTABLE_ARBITRAGE_CANDIDATE",
                "optimizer_ms": 2.0,
            },
            {
                "n_leg_candidate_id": "n-noise",
                "scan_id": "s4",
                "opportunity_id": "n_leg:noise",
                "name": "noise",
                "leg_count": 3,
                "classification": "REJECTED",
                "rejection_reason": "missing_ask",
            },
        ],
    )
    _write_dataset(
        run_dir,
        "pair_observations",
        [
            {
                "observation_id": "p-noise",
                "scan_id": "s4",
                "opportunity_id": "pair:noise",
                "pair_name": "PAIR_NOISE",
                "candidate_type": "pair",
                "classification": "REJECTED",
                "rejection_reason": "no_ask_depth",
                "spread_check_passed": True,
                "depth_check_passed": False,
                "edge_check_passed": False,
            }
        ],
    )
    _write_dataset(
        run_dir,
        "book_snapshots",
        [
            {"book_snapshot_id": "b1", "scan_id": "s1", "token_id": "t1", "source_event_id": "e1", "book_age_ms": 100.0, "ask_depth": 50.0, "depth_untrusted": False},
            {"book_snapshot_id": "b2", "scan_id": "s2", "token_id": "t1", "source_event_id": "e1", "book_age_ms": 120.0, "ask_depth": 50.0, "depth_untrusted": False},
            {"book_snapshot_id": "b3", "scan_id": "s3", "token_id": "t2", "source_event_id": "e2", "book_age_ms": 150.0, "ask_depth": 0.0, "depth_untrusted": False},
            {"book_snapshot_id": "b4", "scan_id": "s4", "token_id": "t3", "source_event_id": "e3", "book_age_ms": 6000.0, "ask_depth": 10.0, "depth_untrusted": False},
        ],
    )
    _write_dataset(
        run_dir,
        "timeline_events",
        [
            {"timeline_event_id": "ts1", "event_kind": "scan_start", "scan_id": "s1", "timestamp_ns": 1_000_000_000, "perf_ns": 1_000_000, "message": "tokens=3"},
            {"timeline_event_id": "te1", "event_kind": "scan_end", "scan_id": "s1", "timestamp_ns": 1_020_000_000, "perf_ns": 21_000_000, "message": "scan_end"},
            {"timeline_event_id": "ts2", "event_kind": "scan_start", "scan_id": "s2", "timestamp_ns": 1_050_000_000, "perf_ns": 30_000_000, "message": "tokens=3"},
            {"timeline_event_id": "te2", "event_kind": "scan_end", "scan_id": "s2", "timestamp_ns": 1_090_000_000, "perf_ns": 70_000_000, "message": "scan_end"},
            {"timeline_event_id": "ts3", "event_kind": "scan_start", "scan_id": "s3", "timestamp_ns": 2_000_000_000, "perf_ns": 80_000_000, "message": "tokens=3"},
            {"timeline_event_id": "te3", "event_kind": "scan_end", "scan_id": "s3", "timestamp_ns": 2_080_000_000, "perf_ns": 160_000_000, "message": "scan_end"},
            {"timeline_event_id": "td3", "event_kind": "decision", "scan_id": "s3", "decision_id": "d-depth", "opportunity_id": "n_leg:depth", "timestamp_ns": 2_250_000_000, "perf_ns": 200_000_000, "message": "depth_depleted"},
            {"timeline_event_id": "ts4", "event_kind": "scan_start", "scan_id": "s4", "timestamp_ns": 3_000_000_000, "perf_ns": 210_000_000, "message": "tokens=3"},
            {"timeline_event_id": "te4", "event_kind": "scan_end", "scan_id": "s4", "timestamp_ns": 3_050_000_000, "perf_ns": 260_000_000, "message": "scan_end"},
        ],
    )
    _write_dataset(
        run_dir,
        "network",
        [
            {"network_id": "nb1", "endpoint": "/books", "method": "POST", "status_code": 200, "latency_ms": 20.0, "is_429": False, "is_425": False, "is_5xx": False},
            {"network_id": "nb2", "endpoint": "/books", "method": "POST", "status_code": 200, "latency_ms": 40.0, "is_429": False, "is_425": False, "is_5xx": False},
            {"network_id": "nt1", "endpoint": "/time", "method": "GET", "status_code": 200, "latency_ms": 500.0, "is_429": False, "is_425": False, "is_5xx": False},
        ],
    )
    _write_dataset(
        run_dir,
        "system_metrics",
        [
            {"metric_id": "sm1", "timestamp_ns": 1_000_000_000, "perf_ns": 1, "audit_queue_depth": 0, "flush_ms": 2.0, "rss_mb": 100.0},
            {"metric_id": "sm2", "timestamp_ns": 3_000_000_000, "perf_ns": 2, "audit_queue_depth": 0, "flush_ms": 30.0, "rss_mb": 150.0},
        ],
    )
    _write_dataset(run_dir, "ws_events", [{"event_id": "ws1", "event_type": "price_change", "latency_ms": 12.0, "payload_bytes": 100}])
    _write_dataset(run_dir, "portfolio_snapshots", [])
    _write_dataset(run_dir, "market_activity", [])
    _write_dataset(run_dir, "orders", [])

    out_dir = run_dir / "profit_forensics"
    summary = generate_audit_profit_report(
        AuditProfitReportConfig(
            audit_dir=run_dir,
            out_dir=out_dir,
            top_windows=3,
            freshness_ms=(250.0, 1000.0, 5000.0),
            edge_threshold_grid=(0.0, 0.01, 0.02),
        )
    )

    expected_files = [
        "profit_forensics.md",
        "summary.json",
        "bottleneck_ranking.csv",
        "opportunity_leakage.csv",
        "candidate_quality.csv",
        "latency_waterfall.csv",
        "freshness_sweep.csv",
        "edge_threshold_sweep.csv",
        "scan_efficiency.csv",
        "duplicate_guard_replay.csv",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), name

    leakage = _read_csv(out_dir / "opportunity_leakage.csv")
    assert len(leakage) == 3
    duplicate = _row_by(leakage, "window_id", "w-dup")
    depth = _row_by(leakage, "window_id", "w-depth")
    stale = _row_by(leakage, "window_id", "w-stale")

    assert duplicate["captured_paper"] == "true"
    assert duplicate["duplicate_guard"] == "true"
    assert float(duplicate["conservative_recoverable_profit"]) == 0.0
    assert depth["true_competitive_miss"] == "true"
    assert float(depth["conservative_recoverable_profit"]) == 7.0
    assert stale["stale_or_phantom"] == "true"
    assert float(stale["conservative_recoverable_profit"]) == 0.0

    replay = _read_csv(out_dir / "duplicate_guard_replay.csv")
    duplicate_replay = _row_by(replay, "opportunity_id", "n_leg:dup")
    assert float(duplicate_replay["raw_duplicate_profit"]) == 5.0
    assert float(duplicate_replay["conservative_incremental_profit"]) == 0.0
    assert "does not exceed" in duplicate_replay["reason_not_counted"]

    latency_metrics = {row["metric"] for row in _read_csv(out_dir / "latency_waterfall.csv")}
    assert "network:/books.latency_ms" in latency_metrics
    assert "network:/time.latency_ms" in latency_metrics

    bottlenecks = _read_csv(out_dir / "bottleneck_ranking.csv")
    assert _row_by(bottlenecks, "bottleneck", "live order unknown")["category"] == "live_order_unknown"
    assert summary["orders_empty_or_paper_only"] is True
    assert summary["profit_leakage"]["unique_windows"] == 3
    assert len(list((out_dir / "top_windows").glob("*.md"))) == 3

