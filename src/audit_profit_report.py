from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


class AuditProfitReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuditProfitReportConfig:
    audit_dir: Path
    out_dir: Path
    top_windows: int = 50
    freshness_ms: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 5000.0)
    edge_threshold_grid: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005, 0.01)
    memory_limit: str = "8GB"
    threads: int | None = None
    book_level_drilldown_file_limit: int = 200


_SQL_TYPES = {
    "str": "VARCHAR",
    "int": "BIGINT",
    "float": "DOUBLE",
    "bool": "BOOLEAN",
}


_TABLE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "opportunity_windows": [
        ("window_id", "str"),
        ("opportunity_id", "str"),
        ("candidate_name", "str"),
        ("candidate_type", "str"),
        ("first_seen_scan_id", "str"),
        ("last_seen_scan_id", "str"),
        ("first_seen_ns", "int"),
        ("last_seen_ns", "int"),
        ("duration_ms", "float"),
        ("scan_count", "int"),
        ("best_edge", "float"),
        ("status", "str"),
        ("close_reason", "str"),
        ("polling_uncertainty_ms", "float"),
    ],
    "missed_fills": [
        ("missed_fill_id", "str"),
        ("decision_id", "str"),
        ("scan_id", "str"),
        ("opportunity_id", "str"),
        ("candidate_name", "str"),
        ("candidate_type", "str"),
        ("classification", "str"),
        ("reason", "str"),
        ("expected_profit", "float"),
        ("edge", "float"),
        ("gross_cost", "float"),
        ("net_cost", "float"),
        ("detected_ts_ns", "int"),
        ("decision_ts_ns", "int"),
        ("gone_ts_ns", "int"),
        ("market_activity_score", "float"),
    ],
    "decisions": [
        ("decision_id", "str"),
        ("scan_id", "str"),
        ("opportunity_id", "str"),
        ("candidate_type", "str"),
        ("candidate_name", "str"),
        ("observation_id", "str"),
        ("n_leg_candidate_id", "str"),
        ("portfolio_snapshot_id", "str"),
        ("order_ids_json", "str"),
        ("timestamp", "str"),
        ("decision_wall_ns", "int"),
        ("decision_perf_ns", "int"),
        ("outcome", "str"),
        ("skip_reason", "str"),
        ("action", "str"),
        ("passed_spread_check", "bool"),
        ("passed_depth_check", "bool"),
        ("passed_edge_check", "bool"),
        ("passed_capital_check", "bool"),
        ("submitted", "bool"),
        ("filled", "bool"),
        ("book_to_detection_ms", "float"),
        ("detection_to_decision_ms", "float"),
        ("decision_to_ack_ms", "float"),
        ("edge", "float"),
        ("gross_cost", "float"),
        ("net_cost", "float"),
        ("size", "float"),
        ("locked_capital", "float"),
    ],
    "n_leg_candidates": [
        ("n_leg_candidate_id", "str"),
        ("scan_id", "str"),
        ("opportunity_id", "str"),
        ("name", "str"),
        ("leg_count", "int"),
        ("leg_token_ids_json", "str"),
        ("leg_labels_json", "str"),
        ("guaranteed_payout", "float"),
        ("gross_cost", "float"),
        ("gross_edge", "float"),
        ("score_edge", "float"),
        ("optimal_size", "float"),
        ("optimal_capital", "float"),
        ("optimal_profit", "float"),
        ("max_spend_size", "float"),
        ("max_spend_capital", "float"),
        ("max_spend_profit", "float"),
        ("classification", "str"),
        ("rejection_reason", "str"),
        ("optimizer_ms", "float"),
    ],
    "pair_observations": [
        ("observation_id", "str"),
        ("scan_id", "str"),
        ("opportunity_id", "str"),
        ("timestamp", "str"),
        ("pair_name", "str"),
        ("candidate_type", "str"),
        ("parent_token_id", "str"),
        ("child_token_id", "str"),
        ("gross_total_cost", "float"),
        ("net_total_cost", "float"),
        ("distance_to_entry", "float"),
        ("worst_case_profit_per_unit", "float"),
        ("best_case_profit_per_unit", "float"),
        ("max_executable_size", "float"),
        ("classification", "str"),
        ("rejection_reason", "str"),
        ("optimal_size", "float"),
        ("optimal_required_capital", "float"),
        ("optimal_guaranteed_profit", "float"),
        ("spread_check_passed", "bool"),
        ("depth_check_passed", "bool"),
        ("edge_check_passed", "bool"),
    ],
    "book_snapshots": [
        ("book_snapshot_id", "str"),
        ("scan_id", "str"),
        ("token_id", "str"),
        ("source", "str"),
        ("source_event_id", "str"),
        ("exchange_ts_ns", "int"),
        ("local_ts_ns", "int"),
        ("book_age_ms", "float"),
        ("best_bid", "float"),
        ("best_ask", "float"),
        ("spread", "float"),
        ("bid_depth", "float"),
        ("ask_depth", "float"),
        ("bid_level_count", "int"),
        ("ask_level_count", "int"),
        ("depth_untrusted", "bool"),
    ],
    "book_levels": [
        ("book_snapshot_id", "str"),
        ("scan_id", "str"),
        ("token_id", "str"),
        ("side", "str"),
        ("level_index", "int"),
        ("price", "float"),
        ("size", "float"),
        ("cumulative_size", "float"),
        ("local_ts_ns", "int"),
    ],
    "timeline_events": [
        ("timeline_event_id", "str"),
        ("event_kind", "str"),
        ("scan_id", "str"),
        ("decision_id", "str"),
        ("opportunity_id", "str"),
        ("token_id", "str"),
        ("candidate_name", "str"),
        ("timestamp_ns", "int"),
        ("perf_ns", "int"),
        ("metric_1", "float"),
        ("metric_2", "float"),
        ("message", "str"),
    ],
    "network": [
        ("network_id", "str"),
        ("scan_id", "str"),
        ("decision_id", "str"),
        ("opportunity_id", "str"),
        ("order_id", "str"),
        ("method", "str"),
        ("endpoint", "str"),
        ("status_code", "int"),
        ("is_429", "bool"),
        ("is_425", "bool"),
        ("is_5xx", "bool"),
        ("queue_wait_ms", "float"),
        ("semaphore_wait_ms", "float"),
        ("backoff_ms", "float"),
        ("latency_ms", "float"),
        ("request_bytes", "int"),
        ("response_bytes", "int"),
        ("exception", "str"),
        ("wall_start_ns", "int"),
        ("wall_end_ns", "int"),
        ("perf_start_ns", "int"),
        ("perf_end_ns", "int"),
    ],
    "system_metrics": [
        ("metric_id", "str"),
        ("timestamp_ns", "int"),
        ("perf_ns", "int"),
        ("audit_queue_depth", "int"),
        ("flush_ms", "float"),
        ("event_loop_lag_ms", "float"),
        ("rss_mb", "float"),
        ("open_table_count", "int"),
    ],
    "ws_events": [
        ("event_id", "str"),
        ("event_index", "int"),
        ("event_type", "str"),
        ("market", "str"),
        ("token_ids_json", "str"),
        ("exchange_ts_ns", "int"),
        ("local_receipt_ts_ns", "int"),
        ("processing_ts_ns", "int"),
        ("receipt_perf_ns", "int"),
        ("processing_perf_ns", "int"),
        ("latency_ms", "float"),
        ("hash", "str"),
        ("sequence", "str"),
        ("payload_bytes", "int"),
    ],
    "orders": [
        ("order_id", "str"),
        ("decision_id", "str"),
        ("scan_id", "str"),
        ("opportunity_id", "str"),
        ("strategy_name", "str"),
        ("leg_count", "int"),
        ("token_ids_json", "str"),
        ("requested_size", "float"),
        ("requested_notional", "float"),
        ("limit_prices_json", "str"),
        ("order_type", "str"),
        ("signing_started_ns", "int"),
        ("submission_started_ns", "int"),
        ("ack_received_ns", "int"),
        ("signing_ms", "float"),
        ("submission_ms", "float"),
        ("success", "bool"),
        ("error", "str"),
        ("responses_json", "str"),
    ],
    "portfolio_snapshots": [
        ("portfolio_snapshot_id", "str"),
        ("scan_id", "str"),
        ("decision_id", "str"),
        ("timestamp", "str"),
        ("cash_available", "float"),
        ("locked_capital", "float"),
        ("open_positions_count", "int"),
        ("live_session_spend", "float"),
        ("max_trade_size", "float"),
        ("min_trade_size", "float"),
        ("max_total_locked_capital", "float"),
        ("capital_fraction_per_trade", "float"),
        ("cooldown_seconds_per_pair", "float"),
        ("open_positions_json", "str"),
        ("exposure_by_pair_json", "str"),
    ],
    "market_activity": [
        ("activity_id", "str"),
        ("event_id", "str"),
        ("event_type", "str"),
        ("market", "str"),
        ("token_id", "str"),
        ("exchange_ts_ns", "int"),
        ("local_receipt_ts_ns", "int"),
        ("interarrival_ms", "float"),
        ("recent_event_count_1000ms", "int"),
        ("recent_trade_count_1000ms", "int"),
        ("price_change_count", "int"),
        ("last_trade_count", "int"),
        ("update_burst_size", "int"),
        ("contested_score", "float"),
    ],
}


_BASE_TABLES = [
    "opportunity_windows",
    "missed_fills",
    "decisions",
    "n_leg_candidates",
    "pair_observations",
    "book_snapshots",
    "timeline_events",
    "network",
    "system_metrics",
    "ws_events",
    "orders",
    "portfolio_snapshots",
    "market_activity",
]


def generate_audit_profit_report(config: AuditProfitReportConfig) -> dict[str, Any]:
    if not config.audit_dir.exists():
        raise AuditProfitReportError(f"Audit directory does not exist: {config.audit_dir}")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "top_windows").mkdir(parents=True, exist_ok=True)

    con = _connect(config)
    try:
        manifest = _read_manifest(config.audit_dir)
        inventory = _inventory(config.audit_dir)
        for table in _BASE_TABLES:
            create_view(con, config.audit_dir, table, _TABLE_COLUMNS[table])

        _create_threshold_tables(con, config)
        _create_derived_tables(con)
        _write_output_tables(con, config)
        bottlenecks = _build_bottleneck_rows(con, config, inventory)
        _write_rows_csv(config.out_dir / "bottleneck_ranking.csv", bottlenecks)
        top_window_files = _write_top_window_reports(con, config, inventory)

        summary = _build_summary(con, config, manifest, inventory, bottlenecks, top_window_files)
        (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (config.out_dir / "profit_forensics.md").write_text(_markdown_report(summary, con), encoding="utf-8")
        return summary
    finally:
        con.close()


def table_exists(audit_dir: Path, name: str) -> bool:
    path = audit_dir / f"{name}.parquet"
    return path.is_dir() and any(path.glob("*.parquet"))


def create_view(con: Any, audit_dir: Path, name: str, columns: Sequence[tuple[str, str]] | None = None) -> None:
    columns = list(columns or _TABLE_COLUMNS.get(name, []))
    if not columns:
        raise AuditProfitReportError(f"No projected columns configured for audit table {name!r}")

    if not table_exists(audit_dir, name):
        exprs = [f"CAST(NULL AS {_SQL_TYPES[kind]}) AS {_q(col)}" for col, kind in columns]
        con.execute(f"CREATE OR REPLACE TEMP VIEW {_q(name)} AS SELECT {', '.join(exprs)} WHERE FALSE")
        return

    glob = _sql_string(str((audit_dir / f"{name}.parquet" / "*.parquet").as_posix()))
    existing = _describe_parquet_columns(con, glob)
    exprs = []
    for col, kind in columns:
        sql_type = _SQL_TYPES[kind]
        if col in existing:
            exprs.append(f"TRY_CAST({_q(col)} AS {sql_type}) AS {_q(col)}")
        else:
            exprs.append(f"CAST(NULL AS {sql_type}) AS {_q(col)}")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW {_q(name)} AS
        SELECT {", ".join(exprs)}
        FROM read_parquet({glob}, union_by_name=true)
        """
    )


def quantiles_sql(expr: str) -> str:
    return (
        f"count({expr}) AS count, "
        f"avg({expr}) AS avg, "
        f"quantile_cont({expr}, 0.5) AS p50, "
        f"quantile_cont({expr}, 0.9) AS p90, "
        f"quantile_cont({expr}, 0.95) AS p95, "
        f"quantile_cont({expr}, 0.99) AS p99, "
        f"max({expr}) AS max"
    )


def write_csv(con: Any, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO {_sql_string(str(path))} (HEADER, DELIMITER ',')")


def write_markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None, *, max_rows: int = 25) -> str:
    if not rows:
        return "_No rows._"
    columns = list(columns or rows[0].keys())
    lines = ["| " + " | ".join(f"`{col}`" for col in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(_md_cell(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def _connect(config: AuditProfitReportConfig) -> Any:
    try:
        import duckdb
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise AuditProfitReportError("Install audit report dependencies with: python -m pip install -e '.[audit]'") from exc
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit={_sql_string(config.memory_limit)}")
    if config.threads is not None:
        con.execute(f"SET threads={int(config.threads)}")
    return con


def _read_manifest(audit_dir: Path) -> dict[str, Any]:
    path = audit_dir / "manifest.json"
    if not path.exists():
        return {"run_id": audit_dir.name, "status": "unknown"}
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory(audit_dir: Path) -> dict[str, Any]:
    tables: dict[str, dict[str, Any]] = {}
    for table in sorted(set(_TABLE_COLUMNS) | {"book_levels"}):
        path = audit_dir / f"{table}.parquet"
        files = list(path.glob("*.parquet")) if path.is_dir() else []
        tables[table] = {
            "path": str(path),
            "exists": bool(files),
            "part_files": len(files),
            "bytes": sum(part.stat().st_size for part in files),
        }
    return {"tables": tables}


def _create_threshold_tables(con: Any, config: AuditProfitReportConfig) -> None:
    freshness_values = ", ".join(f"({float(value)})" for value in config.freshness_ms)
    edge_values = ", ".join(f"({float(value)})" for value in config.edge_threshold_grid)
    con.execute(f"CREATE OR REPLACE TEMP TABLE freshness_thresholds AS SELECT * FROM (VALUES {freshness_values}) AS t(threshold_ms)")
    con.execute(f"CREATE OR REPLACE TEMP TABLE edge_thresholds AS SELECT * FROM (VALUES {edge_values}) AS t(edge_threshold)")


def _create_derived_tables(con: Any) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE windows_final AS
        WITH ranked AS (
            SELECT
                coalesce(window_id, opportunity_id, candidate_name, 'unknown-window') AS window_key,
                window_id,
                opportunity_id,
                candidate_name,
                candidate_type,
                first_seen_scan_id,
                last_seen_scan_id,
                first_seen_ns,
                last_seen_ns,
                duration_ms,
                scan_count,
                best_edge,
                status,
                close_reason,
                polling_uncertainty_ms,
                row_number() OVER (
                    PARTITION BY coalesce(window_id, opportunity_id, candidate_name, 'unknown-window')
                    ORDER BY coalesce(last_seen_ns, first_seen_ns, 0) DESC,
                             coalesce(duration_ms, 0) DESC,
                             coalesce(scan_count, 0) DESC
                ) AS rn
            FROM opportunity_windows
        )
        SELECT
            window_key,
            window_id,
            opportunity_id,
            candidate_name,
            candidate_type,
            first_seen_scan_id,
            last_seen_scan_id,
            first_seen_ns,
            last_seen_ns,
            duration_ms,
            scan_count,
            best_edge,
            status,
            close_reason,
            polling_uncertainty_ms
        FROM ranked
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE missed_without_windows AS
        SELECT
            'synthetic:' || coalesce(opportunity_id, 'unknown') AS window_key,
            NULL::VARCHAR AS window_id,
            opportunity_id,
            any_value(candidate_name) AS candidate_name,
            any_value(candidate_type) AS candidate_type,
            min(detected_ts_ns) AS first_seen_ns,
            max(coalesce(gone_ts_ns, decision_ts_ns, detected_ts_ns)) AS last_seen_ns,
            (max(coalesce(gone_ts_ns, decision_ts_ns, detected_ts_ns)) - min(detected_ts_ns)) / 1e6 AS duration_ms,
            count(*) AS scan_count,
            max(edge) AS best_edge,
            'synthetic_from_missed_fills' AS status,
            any_value(classification) AS close_reason
        FROM missed_fills
        WHERE opportunity_id IS NOT NULL
          AND opportunity_id NOT IN (SELECT opportunity_id FROM windows_final WHERE opportunity_id IS NOT NULL)
        GROUP BY opportunity_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE opportunity_base AS
        SELECT
            window_key,
            window_id,
            opportunity_id,
            candidate_name,
            candidate_type,
            first_seen_scan_id,
            last_seen_scan_id,
            first_seen_ns,
            last_seen_ns,
            duration_ms,
            scan_count,
            best_edge,
            status,
            close_reason,
            polling_uncertainty_ms
        FROM windows_final
        UNION ALL
        SELECT
            window_key,
            window_id,
            opportunity_id,
            candidate_name,
            candidate_type,
            NULL::VARCHAR AS first_seen_scan_id,
            NULL::VARCHAR AS last_seen_scan_id,
            first_seen_ns,
            last_seen_ns,
            duration_ms,
            scan_count,
            best_edge,
            status,
            close_reason,
            NULL::DOUBLE AS polling_uncertainty_ms
        FROM missed_without_windows
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE scan_book_age AS
        SELECT
            scan_id,
            count(*) AS book_snapshot_count,
            max(book_age_ms) AS scan_max_book_age_ms,
            quantile_cont(book_age_ms, 0.95) AS scan_p95_book_age_ms,
            avg(CASE WHEN coalesce(depth_untrusted, false) THEN 1.0 ELSE 0.0 END) AS depth_untrusted_rate,
            count(DISTINCT source_event_id) AS distinct_source_event_count
        FROM book_snapshots
        WHERE scan_id IS NOT NULL
        GROUP BY scan_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE scan_book_change AS
        WITH ordered AS (
            SELECT
                scan_id,
                token_id,
                source_event_id,
                lag(source_event_id) OVER (PARTITION BY token_id ORDER BY scan_id) AS prev_source_event_id
            FROM book_snapshots
            WHERE scan_id IS NOT NULL AND token_id IS NOT NULL
        )
        SELECT
            scan_id,
            avg(CASE WHEN source_event_id IS NOT NULL AND source_event_id = prev_source_event_id THEN 1.0 ELSE 0.0 END) AS unchanged_book_rate
        FROM ordered
        GROUP BY scan_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE scan_times AS
        SELECT
            scan_id,
            (max(CASE WHEN event_kind = 'scan_end' THEN perf_ns END)
             - max(CASE WHEN event_kind = 'scan_start' THEN perf_ns END)) / 1e6 AS scan_ms,
            max(CASE WHEN event_kind = 'scan_start' THEN timestamp_ns END) AS scan_start_ns,
            max(CASE WHEN event_kind = 'scan_end' THEN timestamp_ns END) AS scan_end_ns
        FROM timeline_events
        WHERE event_kind IN ('scan_start', 'scan_end')
        GROUP BY scan_id
        HAVING scan_id IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE nleg_scan AS
        SELECT
            scan_id,
            count(*) AS n_leg_candidate_count,
            sum(CASE WHEN classification = 'EXECUTABLE_ARBITRAGE_CANDIDATE' THEN 1 ELSE 0 END) AS n_leg_executable_count,
            sum(coalesce(optimizer_ms, 0)) AS n_leg_optimizer_ms_sum,
            quantile_cont(optimizer_ms, 0.95) AS n_leg_optimizer_ms_p95
        FROM n_leg_candidates
        WHERE scan_id IS NOT NULL
        GROUP BY scan_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pair_scan AS
        SELECT
            scan_id,
            count(*) AS pair_observation_count,
            sum(CASE WHEN classification = 'EXECUTABLE_ARBITRAGE_CANDIDATE' THEN 1 ELSE 0 END) AS pair_executable_count
        FROM pair_observations
        WHERE scan_id IS NOT NULL
        GROUP BY scan_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE decision_scan AS
        SELECT scan_id, count(*) AS decision_count
        FROM decisions
        WHERE scan_id IS NOT NULL
        GROUP BY scan_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE system_buckets AS
        SELECT
            floor(timestamp_ns / 1000000000) AS wall_second,
            avg(flush_ms) AS flush_ms_avg,
            max(flush_ms) AS flush_ms_max,
            max(audit_queue_depth) AS audit_queue_depth_max,
            max(rss_mb) AS rss_mb_max
        FROM system_metrics
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE scan_efficiency AS
        SELECT
            st.scan_id,
            st.scan_start_ns,
            st.scan_end_ns,
            st.scan_ms,
            coalesce(n.n_leg_candidate_count, 0) AS n_leg_candidate_count,
            coalesce(p.pair_observation_count, 0) AS pair_observation_count,
            coalesce(n.n_leg_executable_count, 0) + coalesce(p.pair_executable_count, 0) AS executable_count,
            coalesce(n.n_leg_optimizer_ms_sum, 0) AS optimizer_ms_sum,
            n.n_leg_optimizer_ms_p95,
            coalesce(d.decision_count, 0) AS decision_count,
            b.book_snapshot_count,
            b.scan_max_book_age_ms,
            b.scan_p95_book_age_ms,
            b.depth_untrusted_rate,
            bc.unchanged_book_rate,
            sb.flush_ms_avg,
            sb.flush_ms_max,
            sb.audit_queue_depth_max,
            sb.rss_mb_max
        FROM scan_times st
        LEFT JOIN nleg_scan n USING (scan_id)
        LEFT JOIN pair_scan p USING (scan_id)
        LEFT JOIN decision_scan d USING (scan_id)
        LEFT JOIN scan_book_age b USING (scan_id)
        LEFT JOIN scan_book_change bc USING (scan_id)
        LEFT JOIN system_buckets sb ON floor(st.scan_start_ns / 1000000000) = sb.wall_second
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE miss_class_counts AS
        SELECT
            opportunity_id,
            classification,
            count(*) AS class_count,
            max(expected_profit) AS class_best_profit
        FROM missed_fills
        WHERE opportunity_id IS NOT NULL
        GROUP BY opportunity_id, classification
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE miss_agg AS
        SELECT
            m.opportunity_id,
            count(*) AS missed_rows,
            count(DISTINCT decision_id) AS missed_decisions,
            arg_max(m.classification, c.class_count) AS dominant_miss_classification,
            max(coalesce(m.expected_profit, 0)) AS best_miss_expected_profit,
            max(coalesce(m.edge, 0)) AS best_miss_edge,
            max(coalesce(m.market_activity_score, 0)) AS max_market_activity_score,
            max(CASE WHEN m.classification IN ('depth_depleted', 'book_vanished', 'depth_vanished', 'trade_hit_our_level') THEN 1 ELSE 0 END) > 0 AS has_depth_depleted,
            max(CASE WHEN m.classification IN ('rest_recheck_missing', 'rest_recheck_lost', 'stale_feed', 'stale_or_gapped_feed') THEN 1 ELSE 0 END) > 0 AS has_rest_recheck_missing,
            max(CASE WHEN m.classification = 'duplicate_guard_blocked' THEN 1 ELSE 0 END) > 0 AS has_duplicate_guard,
            max(CASE WHEN lower(coalesce(m.reason, '')) LIKE '%capital%' OR lower(coalesce(m.reason, '')) LIKE '%cash%' THEN 1 ELSE 0 END) > 0 AS has_capital_text,
            max(CASE WHEN lower(coalesce(m.reason, '')) LIKE '%cooldown%' THEN 1 ELSE 0 END) > 0 AS has_cooldown_text,
            max(CASE WHEN lower(coalesce(m.reason, '')) LIKE '%open%' AND lower(coalesce(m.reason, '')) LIKE '%position%' THEN 1 ELSE 0 END) > 0 AS has_open_position_text
        FROM missed_fills m
        LEFT JOIN miss_class_counts c
          ON m.opportunity_id = c.opportunity_id AND m.classification = c.classification
        WHERE m.opportunity_id IS NOT NULL
        GROUP BY m.opportunity_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE decision_agg AS
        SELECT
            d.opportunity_id,
            count(*) AS decision_rows,
            max(CASE WHEN coalesce(d.filled, false) OR coalesce(d.submitted, false) OR lower(coalesce(d.outcome, '')) IN ('filled', 'submitted') THEN 1 ELSE 0 END) > 0 AS captured_paper,
            max(CASE WHEN coalesce(d.filled, false) THEN 1 ELSE 0 END) > 0 AS filled_paper,
            max(CASE WHEN coalesce(d.submitted, false) THEN 1 ELSE 0 END) > 0 AS submitted_paper,
            max(CASE WHEN lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%duplicate%' THEN 1 ELSE 0 END) > 0 AS duplicate_guard,
            max(CASE WHEN lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%capital%' OR lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%cash%' OR lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%locked%' THEN 1 ELSE 0 END) > 0 AS capital_limited,
            max(CASE WHEN lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%cooldown%' THEN 1 ELSE 0 END) > 0 AS cooldown_limited,
            max(CASE WHEN lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%open%' AND lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%position%' THEN 1 ELSE 0 END) > 0 AS open_position_limited,
            max(coalesce(d.edge, 0) * coalesce(d.size, 0)) AS decision_profit_proxy,
            max(d.size) AS max_decision_size,
            max(d.locked_capital) AS max_decision_locked_capital,
            max(d.book_to_detection_ms) AS max_book_to_detection_ms,
            max(d.detection_to_decision_ms) AS max_detection_to_decision_ms,
            max(d.decision_to_ack_ms) AS max_decision_to_ack_ms,
            max(b.scan_max_book_age_ms) AS max_book_age_ms,
            max(b.scan_p95_book_age_ms) AS p95_book_age_ms
        FROM decisions d
        LEFT JOIN scan_book_age b USING (scan_id)
        WHERE d.opportunity_id IS NOT NULL
        GROUP BY d.opportunity_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE nleg_agg AS
        SELECT
            opportunity_id,
            any_value(name) AS name,
            count(*) AS nleg_rows,
            sum(CASE WHEN classification = 'EXECUTABLE_ARBITRAGE_CANDIDATE' THEN 1 ELSE 0 END) AS nleg_executable_rows,
            max(coalesce(optimal_profit, max_spend_profit, 0)) AS best_nleg_profit,
            max(coalesce(gross_edge, score_edge, 0)) AS best_nleg_edge,
            max(coalesce(optimal_size, max_spend_size, 0)) AS max_nleg_size,
            max(coalesce(optimal_capital, max_spend_capital, 0)) AS max_nleg_capital,
            sum(coalesce(optimizer_ms, 0)) AS optimizer_ms_sum,
            quantile_cont(optimizer_ms, 0.95) AS optimizer_ms_p95,
            sum(CASE WHEN rejection_reason = 'missing_ask' THEN 1 ELSE 0 END) AS missing_ask_rows,
            sum(CASE WHEN rejection_reason = 'not_positive_edge' THEN 1 ELSE 0 END) AS not_positive_edge_rows
        FROM n_leg_candidates
        WHERE opportunity_id IS NOT NULL
        GROUP BY opportunity_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pair_agg AS
        SELECT
            opportunity_id,
            any_value(pair_name) AS name,
            count(*) AS pair_rows,
            sum(CASE WHEN classification = 'EXECUTABLE_ARBITRAGE_CANDIDATE' THEN 1 ELSE 0 END) AS pair_executable_rows,
            max(coalesce(optimal_guaranteed_profit, worst_case_profit_per_unit, 0)) AS best_pair_profit,
            max(coalesce(worst_case_profit_per_unit, best_case_profit_per_unit, 0)) AS best_pair_edge,
            sum(CASE WHEN rejection_reason = 'missing_ask' THEN 1 ELSE 0 END) AS missing_ask_rows,
            sum(CASE WHEN rejection_reason = 'no_ask_depth' THEN 1 ELSE 0 END) AS no_ask_depth_rows,
            sum(CASE WHEN rejection_reason = 'too_expensive' THEN 1 ELSE 0 END) AS too_expensive_rows,
            sum(CASE WHEN rejection_reason = 'spread_too_wide' THEN 1 ELSE 0 END) AS spread_too_wide_rows,
            sum(CASE WHEN rejection_reason = 'not_positive_edge' THEN 1 ELSE 0 END) AS not_positive_edge_rows
        FROM pair_observations
        WHERE opportunity_id IS NOT NULL
        GROUP BY opportunity_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE opportunity_leakage AS
        WITH joined AS (
            SELECT
                ob.window_key,
                ob.window_id,
                ob.opportunity_id,
                coalesce(ob.candidate_name, n.name, p.name, ob.opportunity_id) AS candidate_name,
                coalesce(ob.candidate_type, CASE WHEN n.opportunity_id IS NOT NULL THEN 'n_leg' END, 'pair') AS candidate_type,
                ob.first_seen_scan_id,
                ob.last_seen_scan_id,
                ob.first_seen_ns,
                ob.last_seen_ns,
                ob.duration_ms,
                ob.scan_count,
                greatest(coalesce(ob.best_edge, 0), coalesce(m.best_miss_edge, 0), coalesce(n.best_nleg_edge, 0), coalesce(p.best_pair_edge, 0)) AS best_edge,
                greatest(
                    coalesce(m.best_miss_expected_profit, 0),
                    coalesce(n.best_nleg_profit, 0),
                    coalesce(p.best_pair_profit, 0),
                    coalesce(d.decision_profit_proxy, 0),
                    coalesce(ob.best_edge, 0)
                ) AS best_expected_profit_raw,
                coalesce(d.captured_paper, false) AS captured_paper,
                coalesce(m.dominant_miss_classification, CASE WHEN coalesce(d.captured_paper, false) THEN 'captured_paper' ELSE NULL END, 'unknown') AS dominant_miss_classification,
                coalesce(m.has_depth_depleted, false) AS has_depth_depleted,
                coalesce(m.has_rest_recheck_missing, false) AS has_rest_recheck_missing,
                coalesce(m.has_duplicate_guard, false) OR coalesce(d.duplicate_guard, false) AS duplicate_guard,
                coalesce(d.capital_limited, false) OR coalesce(m.has_capital_text, false) AS capital_limited,
                coalesce(d.cooldown_limited, false) OR coalesce(m.has_cooldown_text, false) AS cooldown_limited,
                coalesce(d.open_position_limited, false) OR coalesce(m.has_open_position_text, false) AS open_position_limited,
                coalesce(m.missed_rows, 0) AS missed_rows,
                coalesce(d.decision_rows, 0) AS decision_rows,
                coalesce(n.nleg_rows, 0) + coalesce(p.pair_rows, 0) AS evaluated_candidate_rows,
                coalesce(n.nleg_executable_rows, 0) + coalesce(p.pair_executable_rows, 0) AS executable_candidate_rows,
                coalesce(m.max_market_activity_score, 0) AS market_activity_score,
                coalesce(d.max_book_age_ms, last_scan_age.scan_max_book_age_ms, first_scan_age.scan_max_book_age_ms) AS max_book_age_ms,
                coalesce(d.p95_book_age_ms, last_scan_age.scan_p95_book_age_ms, first_scan_age.scan_p95_book_age_ms) AS p95_book_age_ms,
                d.max_book_to_detection_ms,
                d.max_detection_to_decision_ms,
                d.max_decision_to_ack_ms
            FROM opportunity_base ob
            LEFT JOIN miss_agg m USING (opportunity_id)
            LEFT JOIN decision_agg d USING (opportunity_id)
            LEFT JOIN nleg_agg n USING (opportunity_id)
            LEFT JOIN pair_agg p USING (opportunity_id)
            LEFT JOIN scan_book_age first_scan_age ON ob.first_seen_scan_id = first_scan_age.scan_id
            LEFT JOIN scan_book_age last_scan_age ON ob.last_seen_scan_id = last_scan_age.scan_id
        ),
        classified AS (
            SELECT
                *,
                has_depth_depleted AND NOT captured_paper AND NOT duplicate_guard AND NOT capital_limited AND NOT cooldown_limited AND NOT open_position_limited AS true_competitive_miss,
                duplicate_guard OR cooldown_limited OR open_position_limited AS policy_limited,
                has_rest_recheck_missing OR coalesce(max_book_age_ms, 0) > 5000 AS stale_or_phantom,
                coalesce(max_book_age_ms, 0) > 2000 AS freshness_limited,
                has_depth_depleted AS latency_limited
            FROM joined
        )
        SELECT
            window_key,
            window_id,
            opportunity_id,
            candidate_name,
            candidate_type,
            first_seen_scan_id,
            last_seen_scan_id,
            first_seen_ns,
            last_seen_ns,
            duration_ms,
            scan_count,
            best_edge,
            best_expected_profit_raw,
            captured_paper,
            dominant_miss_classification,
            true_competitive_miss,
            policy_limited,
            stale_or_phantom,
            capital_limited,
            latency_limited,
            freshness_limited,
            duplicate_guard,
            cooldown_limited,
            open_position_limited,
            missed_rows,
            decision_rows,
            evaluated_candidate_rows,
            executable_candidate_rows,
            market_activity_score,
            max_book_age_ms,
            p95_book_age_ms,
            max_book_to_detection_ms,
            max_detection_to_decision_ms,
            max_decision_to_ack_ms,
            CASE
                WHEN true_competitive_miss AND NOT stale_or_phantom THEN best_expected_profit_raw
                ELSE 0
            END AS conservative_recoverable_profit,
            CASE
                WHEN captured_paper THEN 'Captured in paper run; not missed profit.'
                WHEN true_competitive_miss AND NOT stale_or_phantom THEN 'Depth depleted after detection with no policy/capital blocker; likely recoverable only if faster.'
                WHEN true_competitive_miss AND stale_or_phantom THEN 'Depth-depletion-like miss, but stale book evidence weakens recoverability.'
                WHEN stale_or_phantom THEN 'REST recheck/cache freshness evidence points to stale or phantom opportunity.'
                WHEN duplicate_guard THEN 'Duplicate/open-position guard blocked repeat entry; not counted as speed loss.'
                WHEN capital_limited THEN 'Capital/cash/locked-capital limit blocked the decision.'
                WHEN cooldown_limited OR open_position_limited THEN 'Policy guard blocked the decision.'
                ELSE 'No clear recoverable profit signal from available audit tables.'
            END AS evidence_summary
        FROM classified
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE decision_freshness AS
        SELECT
            d.*,
            b.scan_max_book_age_ms
        FROM decisions d
        LEFT JOIN scan_book_age b USING (scan_id)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE nleg_executable_freshness AS
        SELECT
            n.*,
            b.scan_max_book_age_ms
        FROM n_leg_candidates n
        LEFT JOIN scan_book_age b USING (scan_id)
        WHERE n.classification = 'EXECUTABLE_ARBITRAGE_CANDIDATE'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE missed_freshness AS
        SELECT
            m.*,
            b.scan_max_book_age_ms
        FROM missed_fills m
        LEFT JOIN scan_book_age b USING (scan_id)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE freshness_sweep AS
        SELECT
            t.threshold_ms,
            (SELECT count(*) FROM decision_freshness d WHERE coalesce(d.scan_max_book_age_ms, 0) > t.threshold_ms) AS decisions_blocked,
            (SELECT count(*) FROM nleg_executable_freshness n WHERE coalesce(n.scan_max_book_age_ms, 0) > t.threshold_ms) AS executable_candidates_blocked,
            (SELECT count(*) FROM missed_freshness m WHERE m.classification IN ('depth_depleted', 'book_vanished', 'depth_vanished', 'trade_hit_our_level') AND coalesce(m.scan_max_book_age_ms, 0) > t.threshold_ms) AS true_misses_blocked,
            (SELECT count(*) FROM missed_freshness m WHERE m.classification IN ('rest_recheck_missing', 'rest_recheck_lost', 'stale_feed', 'stale_or_gapped_feed') AND coalesce(m.scan_max_book_age_ms, 0) > t.threshold_ms) AS rest_recheck_missing_blocked,
            (SELECT count(*) FROM missed_freshness m WHERE m.classification = 'duplicate_guard_blocked' AND coalesce(m.scan_max_book_age_ms, 0) > t.threshold_ms) AS duplicate_guards_blocked,
            (SELECT count(*) FROM opportunity_leakage l WHERE l.captured_paper AND coalesce(l.max_book_age_ms, 0) > t.threshold_ms) AS captured_opportunities_accidentally_blocked,
            (SELECT count(*) FROM opportunity_leakage l WHERE coalesce(l.max_book_age_ms, 0) > t.threshold_ms) AS unique_windows_blocked,
            (SELECT sum(best_expected_profit_raw) FROM opportunity_leakage l WHERE coalesce(l.max_book_age_ms, 0) > t.threshold_ms) AS raw_expected_profit_removed,
            (SELECT sum(conservative_recoverable_profit) FROM opportunity_leakage l WHERE coalesce(l.max_book_age_ms, 0) > t.threshold_ms) AS conservative_recoverable_profit_removed,
            (SELECT count(*) FROM opportunity_leakage l WHERE l.stale_or_phantom AND coalesce(l.max_book_age_ms, 0) > t.threshold_ms) AS stale_or_phantom_windows_blocked,
            (SELECT count(*) FROM opportunity_leakage l WHERE l.stale_or_phantom) AS stale_or_phantom_windows_total,
            CASE
                WHEN (SELECT count(*) FROM opportunity_leakage l WHERE l.stale_or_phantom) = 0 THEN 0
                ELSE (SELECT count(*) FROM opportunity_leakage l WHERE l.stale_or_phantom AND coalesce(l.max_book_age_ms, 0) > t.threshold_ms)::DOUBLE
                     / (SELECT count(*) FROM opportunity_leakage l WHERE l.stale_or_phantom)
            END AS false_positive_reduction
        FROM freshness_thresholds t
        ORDER BY t.threshold_ms
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE edge_threshold_sweep AS
        SELECT
            t.edge_threshold,
            count(*) FILTER (WHERE best_edge >= t.edge_threshold) AS unique_windows_above_threshold,
            count(*) FILTER (WHERE captured_paper AND best_edge >= t.edge_threshold) AS captured_windows_above_threshold,
            count(*) FILTER (WHERE true_competitive_miss AND best_edge >= t.edge_threshold) AS true_competitive_misses_above_threshold,
            count(*) FILTER (WHERE policy_limited AND best_edge >= t.edge_threshold) AS duplicate_policy_windows_above_threshold,
            count(*) FILTER (WHERE stale_or_phantom AND best_edge >= t.edge_threshold) AS stale_or_phantom_windows_above_threshold,
            sum(best_expected_profit_raw) FILTER (WHERE best_edge >= t.edge_threshold) AS best_expected_profit_raw,
            sum(conservative_recoverable_profit) FILTER (WHERE best_edge >= t.edge_threshold) AS conservative_recoverable_profit,
            count(*) FILTER (WHERE stale_or_phantom AND best_edge < t.edge_threshold) AS estimated_false_positives_removed
        FROM edge_thresholds t
        CROSS JOIN opportunity_leakage
        GROUP BY t.edge_threshold
        ORDER BY t.edge_threshold
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE candidate_quality AS
        WITH quality_base AS (
            SELECT
                'n_leg' AS candidate_type,
                opportunity_id,
                any_value(name) AS candidate_name,
                count(*) AS rows_evaluated,
                sum(CASE WHEN classification = 'EXECUTABLE_ARBITRAGE_CANDIDATE' THEN 1 ELSE 0 END) AS executable_rows,
                sum(CASE WHEN rejection_reason = 'missing_ask' THEN 1 ELSE 0 END) AS missing_ask_rows,
                0::BIGINT AS no_ask_depth_rows,
                0::BIGINT AS too_expensive_rows,
                0::BIGINT AS spread_too_wide_rows,
                sum(CASE WHEN rejection_reason = 'not_positive_edge' THEN 1 ELSE 0 END) AS not_positive_edge_rows,
                sum(coalesce(optimizer_ms, 0)) AS optimizer_ms_sum,
                quantile_cont(optimizer_ms, 0.95) AS optimizer_ms_p95,
                max(coalesce(optimal_profit, max_spend_profit, 0)) AS best_expected_profit
            FROM n_leg_candidates
            GROUP BY opportunity_id
            UNION ALL
            SELECT
                'pair' AS candidate_type,
                opportunity_id,
                any_value(pair_name) AS candidate_name,
                count(*) AS rows_evaluated,
                sum(CASE WHEN classification = 'EXECUTABLE_ARBITRAGE_CANDIDATE' THEN 1 ELSE 0 END) AS executable_rows,
                sum(CASE WHEN rejection_reason = 'missing_ask' THEN 1 ELSE 0 END) AS missing_ask_rows,
                sum(CASE WHEN rejection_reason = 'no_ask_depth' THEN 1 ELSE 0 END) AS no_ask_depth_rows,
                sum(CASE WHEN rejection_reason = 'too_expensive' THEN 1 ELSE 0 END) AS too_expensive_rows,
                sum(CASE WHEN rejection_reason = 'spread_too_wide' THEN 1 ELSE 0 END) AS spread_too_wide_rows,
                sum(CASE WHEN rejection_reason = 'not_positive_edge' THEN 1 ELSE 0 END) AS not_positive_edge_rows,
                0::DOUBLE AS optimizer_ms_sum,
                NULL::DOUBLE AS optimizer_ms_p95,
                max(coalesce(optimal_guaranteed_profit, worst_case_profit_per_unit, 0)) AS best_expected_profit
            FROM pair_observations
            GROUP BY opportunity_id
        ),
        leakage_by_opportunity AS (
            SELECT
                opportunity_id,
                count(*) AS unique_profitable_windows,
                sum(CASE WHEN true_competitive_miss THEN 1 ELSE 0 END) AS true_competitive_miss_windows,
                sum(CASE WHEN stale_or_phantom THEN 1 ELSE 0 END) AS stale_or_phantom_windows,
                sum(CASE WHEN duplicate_guard THEN 1 ELSE 0 END) AS duplicate_guard_windows,
                max(best_expected_profit_raw) AS best_window_profit,
                sum(conservative_recoverable_profit) AS conservative_recoverable_profit
            FROM opportunity_leakage
            GROUP BY opportunity_id
        )
        SELECT
            qb.candidate_type,
            qb.opportunity_id,
            qb.candidate_name,
            qb.rows_evaluated,
            qb.executable_rows,
            CASE WHEN qb.rows_evaluated = 0 THEN 0 ELSE qb.executable_rows::DOUBLE / qb.rows_evaluated END AS executable_retention_rate,
            qb.missing_ask_rows,
            qb.no_ask_depth_rows,
            qb.too_expensive_rows,
            qb.spread_too_wide_rows,
            qb.not_positive_edge_rows,
            qb.optimizer_ms_sum,
            qb.optimizer_ms_p95,
            coalesce(lb.unique_profitable_windows, 0) AS unique_profitable_windows,
            coalesce(lb.true_competitive_miss_windows, 0) AS true_competitive_miss_windows,
            coalesce(lb.stale_or_phantom_windows, 0) AS stale_or_phantom_windows,
            coalesce(lb.duplicate_guard_windows, 0) AS duplicate_guard_windows,
            greatest(coalesce(qb.best_expected_profit, 0), coalesce(lb.best_window_profit, 0)) AS best_expected_profit,
            coalesce(lb.conservative_recoverable_profit, 0) AS conservative_recoverable_profit,
            qb.rows_evaluated AS estimated_audit_rows_created,
            (qb.missing_ask_rows + qb.no_ask_depth_rows + qb.too_expensive_rows + qb.spread_too_wide_rows + qb.not_positive_edge_rows) AS estimated_compute_audit_waste_rows,
            CASE WHEN qb.rows_evaluated = 0 THEN 0 ELSE 1000.0 * coalesce(lb.conservative_recoverable_profit, 0) / qb.rows_evaluated END AS profit_per_1000_evaluated_rows
        FROM quality_base qb
        LEFT JOIN leakage_by_opportunity lb USING (opportunity_id)
        ORDER BY profit_per_1000_evaluated_rows DESC, rows_evaluated DESC
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE duplicate_guard_replay AS
        WITH dup AS (
            SELECT
                m.opportunity_id,
                any_value(m.candidate_name) AS candidate_name,
                count(*) AS duplicate_rows,
                count(DISTINCT m.decision_id) AS duplicate_decisions,
                max(coalesce(m.expected_profit, 0)) AS raw_duplicate_profit,
                max(coalesce(m.edge, 0)) AS best_duplicate_edge
            FROM missed_fills m
            WHERE m.classification = 'duplicate_guard_blocked'
            GROUP BY m.opportunity_id
        ),
        filled AS (
            SELECT
                opportunity_id,
                count(*) AS earlier_filled_or_submitted_decisions,
                max(size) AS already_open_size,
                max(locked_capital) AS already_open_capital
            FROM decisions
            WHERE coalesce(filled, false) OR coalesce(submitted, false) OR lower(coalesce(outcome, '')) IN ('filled', 'submitted')
            GROUP BY opportunity_id
        ),
        later_candidate AS (
            SELECT
                d.opportunity_id,
                max(coalesce(n.optimal_size, n.max_spend_size, 0)) AS later_candidate_size,
                max(coalesce(n.optimal_capital, n.max_spend_capital, 0)) AS later_candidate_capital,
                max(coalesce(n.optimal_profit, n.max_spend_profit, 0)) AS later_candidate_profit
            FROM decisions d
            LEFT JOIN n_leg_candidates n ON d.n_leg_candidate_id = n.n_leg_candidate_id
            WHERE lower(coalesce(d.skip_reason, '') || ' ' || coalesce(d.action, '')) LIKE '%duplicate%'
            GROUP BY d.opportunity_id
        )
        SELECT
            dup.opportunity_id,
            dup.candidate_name,
            dup.duplicate_rows,
            dup.duplicate_decisions,
            coalesce(f.earlier_filled_or_submitted_decisions, 0) AS earlier_filled_or_submitted_decisions,
            f.already_open_size,
            f.already_open_capital,
            lc.later_candidate_size,
            lc.later_candidate_capital,
            lc.later_candidate_profit,
            dup.raw_duplicate_profit,
            CASE
                WHEN coalesce(f.earlier_filled_or_submitted_decisions, 0) = 0 THEN 0
                WHEN coalesce(lc.later_candidate_size, 0) <= coalesce(f.already_open_size, 0) THEN 0
                ELSE 0
            END AS conservative_incremental_profit,
            CASE
                WHEN coalesce(f.earlier_filled_or_submitted_decisions, 0) = 0 THEN 'No earlier filled/submitted paper decision for same opportunity.'
                WHEN coalesce(lc.later_candidate_size, 0) <= coalesce(f.already_open_size, 0) THEN 'Later optimal size does not exceed already-open size/capital.'
                ELSE 'Residual size suggested, but residual book-level depth was not proven; not counted conservatively.'
            END AS reason_not_counted,
            false AS residual_book_depth_verified
        FROM dup
        LEFT JOIN filled f USING (opportunity_id)
        LEFT JOIN later_candidate lc USING (opportunity_id)
        ORDER BY raw_duplicate_profit DESC, duplicate_rows DESC
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE latency_waterfall AS
        SELECT 'ws_events.latency_ms' AS metric, 'websocket' AS category, 'exchange/local event lag' AS note, """ + quantiles_sql("latency_ms") + """ FROM ws_events WHERE latency_ms IS NOT NULL
        UNION ALL
        SELECT 'book_snapshots.book_age_ms' AS metric, 'feed_freshness' AS category, 'book cache age at scan' AS note, """ + quantiles_sql("book_age_ms") + """ FROM book_snapshots WHERE book_age_ms IS NOT NULL
        UNION ALL
        SELECT 'decisions.book_to_detection_ms' AS metric, 'decision' AS category, 'book update to detection' AS note, """ + quantiles_sql("book_to_detection_ms") + """ FROM decisions WHERE book_to_detection_ms IS NOT NULL
        UNION ALL
        SELECT 'decisions.detection_to_decision_ms' AS metric, 'decision' AS category, 'scanner detection to decision' AS note, """ + quantiles_sql("detection_to_decision_ms") + """ FROM decisions WHERE detection_to_decision_ms IS NOT NULL
        UNION ALL
        SELECT 'decisions.decision_to_ack_ms' AS metric, 'paper_only' AS category, 'paper-only acknowledgement; not live exchange ack' AS note, """ + quantiles_sql("decision_to_ack_ms") + """ FROM decisions WHERE decision_to_ack_ms IS NOT NULL
        UNION ALL
        SELECT 'network:/books.latency_ms' AS metric, 'network_books' AS category, 'REST /books only' AS note, """ + quantiles_sql("latency_ms") + """ FROM network WHERE endpoint = '/books' AND latency_ms IS NOT NULL
        UNION ALL
        SELECT 'network:/time.latency_ms' AS metric, 'network_clock' AS category, 'clock/probe /time only; separate from trading REST' AS note, """ + quantiles_sql("latency_ms") + """ FROM network WHERE endpoint = '/time' AND latency_ms IS NOT NULL
        UNION ALL
        SELECT 'scan_times.scan_ms' AS metric, 'scan_loop' AS category, 'scan_start to scan_end from timeline_events' AS note, """ + quantiles_sql("scan_ms") + """ FROM scan_times WHERE scan_ms IS NOT NULL
        UNION ALL
        SELECT 'system_metrics.flush_ms' AS metric, 'audit_io' AS category, 'audit flush latency' AS note, """ + quantiles_sql("flush_ms") + """ FROM system_metrics WHERE flush_ms IS NOT NULL
        """
    )


def _write_output_tables(con: Any, config: AuditProfitReportConfig) -> None:
    write_csv(
        con,
        "SELECT * FROM opportunity_leakage ORDER BY conservative_recoverable_profit DESC, best_expected_profit_raw DESC, window_key",
        config.out_dir / "opportunity_leakage.csv",
    )
    write_csv(con, "SELECT * FROM candidate_quality", config.out_dir / "candidate_quality.csv")
    write_csv(con, "SELECT * FROM latency_waterfall", config.out_dir / "latency_waterfall.csv")
    write_csv(con, "SELECT * FROM freshness_sweep", config.out_dir / "freshness_sweep.csv")
    write_csv(con, "SELECT * FROM edge_threshold_sweep", config.out_dir / "edge_threshold_sweep.csv")
    write_csv(con, "SELECT * FROM scan_efficiency ORDER BY scan_start_ns NULLS LAST", config.out_dir / "scan_efficiency.csv")
    write_csv(con, "SELECT * FROM duplicate_guard_replay", config.out_dir / "duplicate_guard_replay.csv")


def _build_bottleneck_rows(con: Any, config: AuditProfitReportConfig, inventory: dict[str, Any]) -> list[dict[str, Any]]:
    total_windows = _scalar(con, "SELECT count(*) FROM opportunity_leakage") or 0
    depth = _profit_counts(con, "true_competitive_miss")
    stale = _profit_counts(con, "stale_or_phantom")
    duplicate = _profit_counts(con, "duplicate_guard")
    capital = _profit_counts(con, "capital_limited")
    policy = _profit_counts(con, "policy_limited")
    rest_missing = _profit_counts(con, "dominant_miss_classification IN ('rest_recheck_missing', 'rest_recheck_lost')")
    books_latency = _metric_row(con, "network:/books.latency_ms")
    time_latency = _metric_row(con, "network:/time.latency_ms")
    book_age = _metric_row(con, "book_snapshots.book_age_ms")
    scan_latency = _metric_row(con, "scan_times.scan_ms")
    flush = _metric_row(con, "system_metrics.flush_ms")
    candidate_noise = _candidate_noise(con)
    unchanged = _scan_efficiency_summary(con)
    orders_count = _scalar(con, "SELECT count(*) FROM orders") or 0
    book_parts = inventory["tables"].get("book_levels", {}).get("part_files", 0)
    all_parts = sum(table["part_files"] for table in inventory["tables"].values())

    rows = [
        {
            "bottleneck": "stale websocket/cache books",
            "category": "feed",
            "evidence": f"{stale['count']} / {total_windows} unique windows marked stale/phantom; book_age_ms p95={_fmt(book_age.get('p95'))}, p99={_fmt(book_age.get('p99'))}.",
            "affected_unique_windows": stale["count"],
            "raw_profit_at_stake": stale["raw_profit"],
            "conservative_recoverable_profit": stale["conservative_profit"],
            "confidence": "high" if stale["count"] else "medium",
            "suggested_fix": "Test a max candidate book-age gate and/or run without --allow-stale-websocket-cache.",
            "validation_experiment": "Run the same audit with freshness gates at 250/500/1000/2000ms and compare true misses versus REST-recheck false positives.",
            "risk_or_tradeoff": "A strict gate can discard real opportunities when the feed is slow but still tradable.",
        },
        {
            "bottleneck": "REST recheck false positives",
            "category": "feed",
            "evidence": f"{rest_missing['count']} unique windows dominated by REST recheck missing/lost.",
            "affected_unique_windows": rest_missing["count"],
            "raw_profit_at_stake": rest_missing["raw_profit"],
            "conservative_recoverable_profit": 0.0,
            "confidence": "medium",
            "suggested_fix": "Treat REST-recheck-missing as stale/phantom unless fresh book evidence proves otherwise.",
            "validation_experiment": "Compare websocket cache age and REST /books recheck outcomes around these decisions.",
            "risk_or_tradeoff": "May hide real opportunities if REST recheck is slower than the opportunity lifetime.",
        },
        {
            "bottleneck": "depth depletion after detection",
            "category": "scan",
            "evidence": f"{depth['count']} true competitive miss windows; conservative recoverable profit={_fmt(depth['conservative_profit'])}.",
            "affected_unique_windows": depth["count"],
            "raw_profit_at_stake": depth["raw_profit"],
            "conservative_recoverable_profit": depth["conservative_profit"],
            "confidence": "medium" if depth["count"] else "low",
            "suggested_fix": "Reduce detection-to-submit path and prefer event-triggered checks for active candidates.",
            "validation_experiment": "For each true miss, compare window_duration_ms to scan_ms + detection_to_decision_ms + /books p95.",
            "risk_or_tradeoff": "Faster acting can increase stale fills unless REST rechecks remain strict.",
        },
        {
            "bottleneck": "scan cadence/tail latency",
            "category": "scan",
            "evidence": f"scan_ms p95={_fmt(scan_latency.get('p95'))}, p99={_fmt(scan_latency.get('p99'))}; fixed poll target was 50ms if run used --poll-seconds 0.05.",
            "affected_unique_windows": depth["count"],
            "raw_profit_at_stake": depth["raw_profit"],
            "conservative_recoverable_profit": depth["conservative_profit"],
            "confidence": "medium",
            "suggested_fix": "Test event-driven candidate invalidation instead of fixed full-universe rescans.",
            "validation_experiment": "Compare missed-window durations against scan_ms distribution and scan_efficiency unchanged-book rate.",
            "risk_or_tradeoff": "Event-driven code is more complex and can miss cross-token dependencies if invalidation is incomplete.",
        },
        {
            "bottleneck": "unchanged-book rescans",
            "category": "scan",
            "evidence": f"Average unchanged-book rate={_fmt(unchanged.get('avg_unchanged_book_rate'))}; p95 scan candidate rows={_fmt(unchanged.get('p95_total_candidate_rows'))}.",
            "affected_unique_windows": total_windows,
            "raw_profit_at_stake": 0.0,
            "conservative_recoverable_profit": 0.0,
            "confidence": "medium",
            "suggested_fix": "Cache candidate validity by token source_event_id and only recompute affected structures.",
            "validation_experiment": "Replay scans counting candidates whose leg books did not change since the prior scan.",
            "risk_or_tradeoff": "Needs careful dependency mapping for N-leg bundles.",
        },
        {
            "bottleneck": "missing-ask/no-depth candidate noise",
            "category": "candidate_universe",
            "evidence": f"missing_ask={candidate_noise['missing_ask_rows']}, no_ask_depth={candidate_noise['no_ask_depth_rows']}, total rows={candidate_noise['rows_evaluated']}.",
            "affected_unique_windows": total_windows,
            "raw_profit_at_stake": 0.0,
            "conservative_recoverable_profit": 0.0,
            "confidence": "high" if candidate_noise["rows_evaluated"] else "low",
            "suggested_fix": "Add early missing-ask/no-depth filters before optimizer work and before verbose audit logging.",
            "validation_experiment": "Run candidate_quality before/after pruning and compare profit per 1,000 evaluated rows.",
            "risk_or_tradeoff": "Over-pruning may miss newly active thin books if no refresh path exists.",
        },
        {
            "bottleneck": "duplicate/open-position policy",
            "category": "duplicate_policy",
            "evidence": f"{duplicate['count']} unique windows had duplicate guard evidence; conservative duplicate replay does not count repeated paper liquidity as profit.",
            "affected_unique_windows": duplicate["count"],
            "raw_profit_at_stake": duplicate["raw_profit"],
            "conservative_recoverable_profit": 0.0,
            "confidence": "high",
            "suggested_fix": "Only allow duplicate upsize when residual depth, risk budget, and incremental optimal size are proven.",
            "validation_experiment": "Use duplicate_guard_replay.csv and targeted book-level drilldowns on small datasets/top windows.",
            "risk_or_tradeoff": "Upsizing increases exposure and can double-count paper-only unconsumed liquidity.",
        },
        {
            "bottleneck": "capital allocation",
            "category": "capital_policy",
            "evidence": f"{capital['count']} unique windows show capital/cash/locked-capital limits; policy-limited windows total={policy['count']}.",
            "affected_unique_windows": capital["count"],
            "raw_profit_at_stake": capital["raw_profit"],
            "conservative_recoverable_profit": 0.0,
            "confidence": "medium" if capital["count"] else "low",
            "suggested_fix": "Review max_total_locked_capital, max_trade_size, and per-opportunity allocation rules.",
            "validation_experiment": "Replay decisions with alternative capital caps while preserving duplicate and risk limits.",
            "risk_or_tradeoff": "More capital increases market/risk exposure and may reduce diversification.",
        },
        {
            "bottleneck": "audit I/O tiny-file fragmentation",
            "category": "audit_io",
            "evidence": f"Audit tables have {all_parts} parquet part files; book_levels alone has {book_parts}; flush_ms p95={_fmt(flush.get('p95'))}, p99={_fmt(flush.get('p99'))}.",
            "affected_unique_windows": total_windows,
            "raw_profit_at_stake": 0.0,
            "conservative_recoverable_profit": 0.0,
            "confidence": "high" if all_parts > 10000 else "medium",
            "suggested_fix": "Use longer audit flush intervals and table-specific batching/compaction for heavy tables.",
            "validation_experiment": "Compare scan_ms p99 with --audit-flush-seconds 1.0 versus 0.2.",
            "risk_or_tradeoff": "Longer flush intervals lose more buffered audit rows on a crash.",
        },
        {
            "bottleneck": "/books REST latency/rate limiting",
            "category": "network",
            "evidence": f"/books latency p50={_fmt(books_latency.get('p50'))}, p95={_fmt(books_latency.get('p95'))}, p99={_fmt(books_latency.get('p99'))}; /time p95 separately={_fmt(time_latency.get('p95'))}.",
            "affected_unique_windows": 0,
            "raw_profit_at_stake": 0.0,
            "conservative_recoverable_profit": 0.0,
            "confidence": "medium",
            "suggested_fix": "Keep /books separate from /time probes when diagnosing trading REST performance.",
            "validation_experiment": "Measure entry REST recheck latency only, with clock probes reduced or disabled.",
            "risk_or_tradeoff": "Reducing probes weakens clock-skew observability.",
        },
        {
            "bottleneck": "live order unknown",
            "category": "live_order_unknown",
            "evidence": f"orders.parquet rows={orders_count}; this paper run cannot infer signing/submission/fill latency.",
            "affected_unique_windows": total_windows,
            "raw_profit_at_stake": 0.0,
            "conservative_recoverable_profit": 0.0,
            "confidence": "high",
            "suggested_fix": "Run a tiny guarded live audit with strict spend limits before making live-latency claims.",
            "validation_experiment": "Populate orders.parquet with signing_ms, submission_ms, ack_received_ns, and success/error outcomes.",
            "risk_or_tradeoff": "Live testing risks real capital even with small limits.",
        },
    ]
    return rows


def _build_summary(
    con: Any,
    config: AuditProfitReportConfig,
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    bottlenecks: list[dict[str, Any]],
    top_window_files: list[str],
) -> dict[str, Any]:
    leakage = _scalar_row(
        con,
        """
        SELECT
            count(*) AS unique_windows,
            sum(CASE WHEN captured_paper THEN 1 ELSE 0 END) AS captured_paper_windows,
            sum(CASE WHEN true_competitive_miss THEN 1 ELSE 0 END) AS true_competitive_miss_windows,
            sum(CASE WHEN stale_or_phantom THEN 1 ELSE 0 END) AS stale_or_phantom_windows,
            sum(CASE WHEN policy_limited THEN 1 ELSE 0 END) AS policy_limited_windows,
            sum(CASE WHEN capital_limited THEN 1 ELSE 0 END) AS capital_limited_windows,
            sum(best_expected_profit_raw) AS raw_profit_at_stake,
            sum(conservative_recoverable_profit) AS conservative_recoverable_profit
        FROM opportunity_leakage
        """,
    )
    orders_count = _scalar(con, "SELECT count(*) FROM orders") or 0
    outputs = {
        "profit_forensics_md": str(config.out_dir / "profit_forensics.md"),
        "summary_json": str(config.out_dir / "summary.json"),
        "bottleneck_ranking_csv": str(config.out_dir / "bottleneck_ranking.csv"),
        "opportunity_leakage_csv": str(config.out_dir / "opportunity_leakage.csv"),
        "candidate_quality_csv": str(config.out_dir / "candidate_quality.csv"),
        "latency_waterfall_csv": str(config.out_dir / "latency_waterfall.csv"),
        "freshness_sweep_csv": str(config.out_dir / "freshness_sweep.csv"),
        "edge_threshold_sweep_csv": str(config.out_dir / "edge_threshold_sweep.csv"),
        "scan_efficiency_csv": str(config.out_dir / "scan_efficiency.csv"),
        "duplicate_guard_replay_csv": str(config.out_dir / "duplicate_guard_replay.csv"),
        "top_windows": top_window_files,
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": manifest.get("run_id") or config.audit_dir.name,
        "status": manifest.get("status", "unknown"),
        "audit_dir": str(config.audit_dir),
        "out_dir": str(config.out_dir),
        "settings": {
            "top_windows": config.top_windows,
            "freshness_ms": list(config.freshness_ms),
            "edge_threshold_grid": list(config.edge_threshold_grid),
            "memory_limit": config.memory_limit,
            "threads": config.threads,
        },
        "profit_leakage": leakage,
        "orders_empty_or_paper_only": orders_count == 0,
        "orders_note": "orders.parquet is empty, so live signing/submission/fill latency is unknown." if orders_count == 0 else "orders.parquet has rows; inspect order latency separately.",
        "inventory": inventory,
        "bottleneck_ranking": bottlenecks,
        "outputs": outputs,
    }


def _markdown_report(summary: dict[str, Any], con: Any) -> str:
    leakage_rows = _rows(
        con,
        """
        SELECT
            window_id,
            opportunity_id,
            candidate_name,
            duration_ms,
            best_edge,
            best_expected_profit_raw,
            conservative_recoverable_profit,
            dominant_miss_classification,
            true_competitive_miss,
            stale_or_phantom,
            policy_limited,
            evidence_summary
        FROM opportunity_leakage
        ORDER BY conservative_recoverable_profit DESC, best_expected_profit_raw DESC
        LIMIT 20
        """,
    )
    latency_rows = _rows(con, "SELECT metric, category, count, p50, p95, p99, max, note FROM latency_waterfall ORDER BY category, metric")
    freshness_rows = _rows(con, "SELECT * FROM freshness_sweep ORDER BY threshold_ms")
    edge_rows = _rows(con, "SELECT * FROM edge_threshold_sweep ORDER BY edge_threshold")
    candidate_rows = _rows(
        con,
        """
        SELECT
            candidate_type,
            opportunity_id,
            rows_evaluated,
            executable_rows,
            missing_ask_rows,
            no_ask_depth_rows,
            too_expensive_rows,
            not_positive_edge_rows,
            profit_per_1000_evaluated_rows
        FROM candidate_quality
        ORDER BY rows_evaluated DESC
        LIMIT 20
        """,
    )
    scan_rows = _rows(
        con,
        """
        SELECT
            count(*) AS scans,
            avg(scan_ms) AS avg_scan_ms,
            quantile_cont(scan_ms, 0.95) AS p95_scan_ms,
            quantile_cont(scan_ms, 0.99) AS p99_scan_ms,
            avg(unchanged_book_rate) AS avg_unchanged_book_rate,
            quantile_cont(n_leg_candidate_count + pair_observation_count, 0.95) AS p95_total_candidate_rows,
            avg(flush_ms_max) AS avg_flush_ms_max
        FROM scan_efficiency
        """,
    )
    duplicate_rows = _rows(con, "SELECT * FROM duplicate_guard_replay ORDER BY raw_duplicate_profit DESC LIMIT 20")
    bottleneck_rows = summary["bottleneck_ranking"]
    profit = summary["profit_leakage"]

    lines = [
        "# Profit Forensics Report",
        "",
        f"Run: `{summary['run_id']}`",
        "",
        f"Audit dir: `{summary['audit_dir']}`",
        "",
        "This report is read-only. It separates recoverable missed profit from stale/phantom opportunities, policy-limited upside, efficiency waste, and unknown live execution risk.",
        "",
        "## Executive Summary",
        "",
        f"- Unique opportunity windows analyzed: `{profit.get('unique_windows', 0)}`",
        f"- Paper-captured windows: `{profit.get('captured_paper_windows', 0)}`",
        f"- True competitive miss windows: `{profit.get('true_competitive_miss_windows', 0)}`",
        f"- Stale/phantom windows: `{profit.get('stale_or_phantom_windows', 0)}`",
        f"- Policy-limited windows: `{profit.get('policy_limited_windows', 0)}`",
        f"- Raw profit at stake: `{_fmt(profit.get('raw_profit_at_stake'))}`",
        f"- Conservative recoverable profit: `{_fmt(profit.get('conservative_recoverable_profit'))}`",
        f"- Orders/live execution: {summary['orders_note']}",
        "",
        "## Bottleneck Ranking",
        "",
        write_markdown_table(
            bottleneck_rows,
            [
                "bottleneck",
                "category",
                "affected_unique_windows",
                "raw_profit_at_stake",
                "conservative_recoverable_profit",
                "confidence",
                "suggested_fix",
                "risk_or_tradeoff",
            ],
            max_rows=20,
        ),
        "",
        "## Opportunity Leakage",
        "",
        write_markdown_table(leakage_rows, max_rows=20),
        "",
        "## Freshness Sweep",
        "",
        write_markdown_table(freshness_rows, max_rows=20),
        "",
        "## Edge Threshold Sweep",
        "",
        write_markdown_table(edge_rows, max_rows=20),
        "",
        "## Candidate Noise / Quality",
        "",
        write_markdown_table(candidate_rows, max_rows=20),
        "",
        "## Scan Efficiency",
        "",
        write_markdown_table(scan_rows, max_rows=5),
        "",
        "Interpretation: high unchanged-book rates and high rejected-candidate counts are evidence for testing event-driven candidate invalidation and early missing-ask/no-depth pruning.",
        "",
        "## Latency Waterfall",
        "",
        write_markdown_table(latency_rows, max_rows=20),
        "",
        "Important: `/books` latency is reported separately from `/time` probe latency. `decisions.decision_to_ack_ms` is paper-only in this run and is not live exchange acknowledgement latency.",
        "",
        "## Duplicate Guard Replay",
        "",
        "Duplicate-guard rows are not counted as missed profit unless incremental size and residual depth are proven. This conservative replay keeps repeated paper-mode liquidity from being counted repeatedly.",
        "",
        write_markdown_table(duplicate_rows, max_rows=20),
        "",
        "## Recommended Experiments",
        "",
        "- Run without `--allow-stale-websocket-cache`, or add a strict max candidate book-age gate and compare `freshness_sweep.csv` against captured/true-miss counts.",
        "- Test event-driven candidate invalidation instead of fixed 50 ms full-universe rescans.",
        "- Add missing-ask/no-depth prefilters before optimizer work and before verbose audit logging.",
        "- Consider duplicate-guard upsize only when residual depth, incremental optimal size, and risk budget are proven.",
        "- Use less aggressive forensic audit settings for live-speed experiments, then compare scan p95/p99 and missed-fill classifications.",
        "",
        "## Output Files",
        "",
    ]
    for name, path in summary["outputs"].items():
        if isinstance(path, list):
            lines.append(f"- `{name}`: {len(path)} files")
        else:
            lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _write_top_window_reports(con: Any, config: AuditProfitReportConfig, inventory: dict[str, Any]) -> list[str]:
    top_rows = _rows(
        con,
        f"""
        SELECT *
        FROM opportunity_leakage
        ORDER BY conservative_recoverable_profit DESC, best_expected_profit_raw DESC, duration_ms DESC
        LIMIT {max(0, int(config.top_windows))}
        """,
    )
    if not top_rows:
        return []

    opportunity_ids = [row.get("opportunity_id") for row in top_rows if row.get("opportunity_id")]
    timeline_by_opportunity: dict[str, list[dict[str, Any]]] = {}
    decisions_by_opportunity: dict[str, list[dict[str, Any]]] = {}
    misses_by_opportunity: dict[str, list[dict[str, Any]]] = {}
    books_by_scan: dict[str, list[dict[str, Any]]] = {}
    levels_by_scan: dict[str, list[dict[str, Any]]] = {}

    if opportunity_ids:
        placeholders = ", ".join("?" for _ in opportunity_ids)
        timeline_rows = _rows(
            con,
            f"""
            SELECT *
            FROM (
                SELECT
                    opportunity_id,
                    event_kind,
                    scan_id,
                    decision_id,
                    timestamp_ns,
                    metric_1,
                    metric_2,
                    message,
                    row_number() OVER (PARTITION BY opportunity_id ORDER BY timestamp_ns) AS rn
                FROM timeline_events
                WHERE opportunity_id IN ({placeholders})
            )
            WHERE rn <= 120
            ORDER BY opportunity_id, timestamp_ns
            """,
            opportunity_ids,
        )
        decision_rows = _rows(
            con,
            f"""
            SELECT *
            FROM (
                SELECT
                    opportunity_id,
                    decision_id,
                    scan_id,
                    outcome,
                    skip_reason,
                    action,
                    filled,
                    submitted,
                    book_to_detection_ms,
                    detection_to_decision_ms,
                    decision_to_ack_ms,
                    edge,
                    size,
                    locked_capital,
                    decision_wall_ns,
                    row_number() OVER (PARTITION BY opportunity_id ORDER BY decision_wall_ns) AS rn
                FROM decisions
                WHERE opportunity_id IN ({placeholders})
            )
            WHERE rn <= 40
            ORDER BY opportunity_id, decision_wall_ns
            """,
            opportunity_ids,
        )
        miss_rows = _rows(
            con,
            f"""
            SELECT *
            FROM (
                SELECT
                    opportunity_id,
                    missed_fill_id,
                    decision_id,
                    scan_id,
                    classification,
                    reason,
                    expected_profit,
                    edge,
                    market_activity_score,
                    row_number() OVER (PARTITION BY opportunity_id ORDER BY expected_profit DESC NULLS LAST) AS rn
                FROM missed_fills
                WHERE opportunity_id IN ({placeholders})
            )
            WHERE rn <= 40
            ORDER BY opportunity_id, expected_profit DESC NULLS LAST
            """,
            opportunity_ids,
        )
        timeline_by_opportunity = _group_rows(timeline_rows, "opportunity_id")
        decisions_by_opportunity = _group_rows(decision_rows, "opportunity_id")
        misses_by_opportunity = _group_rows(miss_rows, "opportunity_id")

    scan_ids: list[str] = []
    for row in top_rows:
        opportunity_id = row.get("opportunity_id")
        for decision in decisions_by_opportunity.get(str(opportunity_id), [])[:5]:
            if decision.get("scan_id"):
                scan_ids.append(str(decision["scan_id"]))
        for scan_key in ("first_seen_scan_id", "last_seen_scan_id"):
            if row.get(scan_key):
                scan_ids.append(str(row[scan_key]))
    scan_ids = list(dict.fromkeys(scan_ids))
    if scan_ids:
        placeholders = ", ".join("?" for _ in scan_ids)
        book_rows = _rows(
            con,
            f"""
            SELECT *
            FROM (
                SELECT
                    scan_id,
                    token_id,
                    source,
                    source_event_id,
                    book_age_ms,
                    best_bid,
                    best_ask,
                    spread,
                    bid_depth,
                    ask_depth,
                    depth_untrusted,
                    row_number() OVER (PARTITION BY scan_id ORDER BY book_age_ms DESC NULLS LAST) AS rn
                FROM book_snapshots
                WHERE scan_id IN ({placeholders})
            )
            WHERE rn <= 80
            ORDER BY scan_id, book_age_ms DESC NULLS LAST
            """,
            scan_ids,
        )
        books_by_scan = _group_rows(book_rows, "scan_id")

    written: list[str] = []
    book_level_files = inventory["tables"].get("book_levels", {}).get("part_files", 0)
    can_query_book_levels = 0 < book_level_files <= config.book_level_drilldown_file_limit
    if can_query_book_levels and scan_ids:
        glob = str((config.audit_dir / "book_levels.parquet" / "*.parquet").as_posix())
        placeholders = ", ".join("?" for _ in scan_ids)
        level_rows = _rows(
            con,
            f"""
            SELECT *
            FROM (
                SELECT
                    scan_id,
                    token_id,
                    side,
                    level_index,
                    price,
                    size,
                    cumulative_size,
                    row_number() OVER (PARTITION BY scan_id ORDER BY token_id, side, level_index) AS rn
                FROM read_parquet({_sql_string(glob)}, union_by_name=true)
                WHERE scan_id IN ({placeholders})
            )
            WHERE rn <= 120
            ORDER BY scan_id, token_id, side, level_index
            """,
            scan_ids,
        )
        levels_by_scan = _group_rows(level_rows, "scan_id")

    for index, row in enumerate(top_rows, start=1):
        slug = _slug(row.get("candidate_name") or row.get("opportunity_id") or row.get("window_id") or f"window-{index}")
        path = config.out_dir / "top_windows" / f"{index:03d}_{slug}.md"
        opportunity_id = row.get("opportunity_id")
        opportunity_key = str(opportunity_id)
        timeline = timeline_by_opportunity.get(opportunity_key, [])
        decisions = decisions_by_opportunity.get(opportunity_key, [])
        misses = misses_by_opportunity.get(opportunity_key, [])
        row_scan_ids = []
        for decision in decisions[:5]:
            if decision.get("scan_id"):
                row_scan_ids.append(str(decision["scan_id"]))
        for scan_key in ("first_seen_scan_id", "last_seen_scan_id"):
            if row.get(scan_key):
                row_scan_ids.append(str(row[scan_key]))
        row_scan_ids = list(dict.fromkeys(row_scan_ids))
        books = [book for scan_id in row_scan_ids for book in books_by_scan.get(scan_id, [])]
        book_levels_note = "Skipped because global book_levels drilldown is intentionally disabled for large fragmented tables."
        book_levels = [level for scan_id in row_scan_ids for level in levels_by_scan.get(scan_id, [])]
        if can_query_book_levels:
            book_levels_note = "Queried because book_levels part-file count was below the drilldown safety limit."

        lines = [
            f"# Top Window {index}: {row.get('candidate_name') or row.get('opportunity_id')}",
            "",
            "## Diagnosis",
            "",
            f"- Window id: `{row.get('window_id')}`",
            f"- Opportunity id: `{row.get('opportunity_id')}`",
            f"- Candidate type: `{row.get('candidate_type')}`",
            f"- Duration ms: `{_fmt(row.get('duration_ms'))}`",
            f"- Scan count: `{row.get('scan_count')}`",
            f"- Best edge: `{_fmt(row.get('best_edge'))}`",
            f"- Raw expected profit: `{_fmt(row.get('best_expected_profit_raw'))}`",
            f"- Conservative recoverable profit: `{_fmt(row.get('conservative_recoverable_profit'))}`",
            f"- Dominant miss classification: `{row.get('dominant_miss_classification')}`",
            f"- True competitive miss: `{row.get('true_competitive_miss')}`",
            f"- Stale/phantom: `{row.get('stale_or_phantom')}`",
            f"- Policy limited: `{row.get('policy_limited')}`",
            f"- Evidence: {row.get('evidence_summary')}",
            "",
            "## Decisions",
            "",
            write_markdown_table(decisions, max_rows=40),
            "",
            "## Missed Fills",
            "",
            write_markdown_table(misses, max_rows=40),
            "",
            "## Timeline Events",
            "",
            write_markdown_table(timeline, max_rows=80),
            "",
            "## Book Snapshots Near Decisions",
            "",
            write_markdown_table(books, max_rows=80),
            "",
            "## Book Levels",
            "",
            book_levels_note,
            "",
            write_markdown_table(book_levels, max_rows=40) if book_levels else "_No targeted book-level rows included._",
            "",
            "## Suggested Experiment",
            "",
            _top_window_experiment(row),
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(str(path))
    return written


def _top_window_experiment(row: dict[str, Any]) -> str:
    if row.get("true_competitive_miss") and not row.get("stale_or_phantom"):
        return "Replay this window with event-driven candidate invalidation and compare whether decision time lands before depth depletion."
    if row.get("stale_or_phantom"):
        return "Replay with a max book-age gate and verify that this candidate disappears before REST recheck."
    if row.get("duplicate_guard"):
        return "Only test duplicate upsize if residual depth and incremental optimal size can be proven."
    if row.get("capital_limited"):
        return "Replay with alternative capital limits to estimate policy-limited upside without changing feed/scan code."
    return "Use this window as a control case; no strong recoverable-profit signal is visible."


def _group_rows(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        grouped.setdefault(str(value), []).append(row)
    return grouped


def _describe_parquet_columns(con: Any, glob: str) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({glob}, union_by_name=true)").fetchall()
    return {str(row[0]) for row in rows}


def _metric_row(con: Any, metric: str) -> dict[str, Any]:
    return _scalar_row(con, "SELECT * FROM latency_waterfall WHERE metric = ?", [metric])


def _profit_counts(con: Any, where: str) -> dict[str, Any]:
    row = _scalar_row(
        con,
        f"""
        SELECT
            count(*) AS count,
            coalesce(sum(best_expected_profit_raw), 0) AS raw_profit,
            coalesce(sum(conservative_recoverable_profit), 0) AS conservative_profit
        FROM opportunity_leakage
        WHERE {where}
        """,
    )
    return row or {"count": 0, "raw_profit": 0.0, "conservative_profit": 0.0}


def _candidate_noise(con: Any) -> dict[str, Any]:
    return _scalar_row(
        con,
        """
        SELECT
            coalesce(sum(rows_evaluated), 0) AS rows_evaluated,
            coalesce(sum(missing_ask_rows), 0) AS missing_ask_rows,
            coalesce(sum(no_ask_depth_rows), 0) AS no_ask_depth_rows,
            coalesce(sum(too_expensive_rows), 0) AS too_expensive_rows,
            coalesce(sum(spread_too_wide_rows), 0) AS spread_too_wide_rows,
            coalesce(sum(not_positive_edge_rows), 0) AS not_positive_edge_rows
        FROM candidate_quality
        """,
    )


def _scan_efficiency_summary(con: Any) -> dict[str, Any]:
    return _scalar_row(
        con,
        """
        SELECT
            avg(unchanged_book_rate) AS avg_unchanged_book_rate,
            quantile_cont(n_leg_candidate_count + pair_observation_count, 0.95) AS p95_total_candidate_rows
        FROM scan_efficiency
        """,
    )


def _scalar(con: Any, query: str, params: Sequence[Any] | None = None) -> Any:
    row = con.execute(query, params or []).fetchone()
    return row[0] if row else None


def _scalar_row(con: Any, query: str, params: Sequence[Any] | None = None) -> dict[str, Any]:
    cursor = con.execute(query, params or [])
    row = cursor.fetchone()
    if row is None:
        return {}
    names = [item[0] for item in cursor.description]
    return dict(zip(names, row))


def _rows(con: Any, query: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    cursor = con.execute(query, params or [])
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _fmt(value)
    text = str(value).replace("\n", " ").replace("\r", " ")
    if len(text) > 180:
        text = text[:177] + "..."
    return text.replace("|", "\\|")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.6f}"


def _slug(value: Any) -> str:
    text = str(value or "window").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:80] or "window"


def parse_float_csv(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise AuditProfitReportError(f"Expected comma-separated numbers, got {value!r}") from exc
    if not parsed:
        raise AuditProfitReportError("Expected at least one numeric value.")
    return parsed
