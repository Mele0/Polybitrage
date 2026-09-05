from __future__ import annotations

import contextlib
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_AUDIT_CONTEXT: ContextVar[dict[str, str | None]] = ContextVar(
    "polymarket_audit_context",
    default={"scan_id": None, "decision_id": None, "opportunity_id": None, "order_id": None},
)


@dataclass(frozen=True)
class AuditConfig:
    mode: str = "off"
    audit_dir: Path = Path("reports/audit")
    run_id: str | None = None
    raw_ws: bool = True
    flush_seconds: float = 1.0
    clock_sync_seconds: float = 30.0
    network_probe_seconds: float = 5.0
    compression: str = "zstd"

    @property
    def enabled(self) -> bool:
        return self.mode == "forensic"


_TYPE_TO_PA = {
    "str": "string",
    "int": "int64",
    "float": "float64",
    "bool": "bool_",
}


_TABLES: dict[str, list[tuple[str, str]]] = {
    "clock_sync": [
        ("sample_id", "str"),
        ("wall_send_ns", "int"),
        ("wall_recv_ns", "int"),
        ("perf_send_ns", "int"),
        ("perf_recv_ns", "int"),
        ("server_time_ns", "int"),
        ("rtt_ms", "float"),
        ("estimated_skew_ms", "float"),
        ("uncertainty_ms", "float"),
        ("source", "str"),
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
        ("raw_json", "str"),
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
    "pair_observations": [
        ("observation_id", "str"),
        ("scan_id", "str"),
        ("opportunity_id", "str"),
        ("timestamp", "str"),
        ("pair_name", "str"),
        ("candidate_type", "str"),
        ("parent_token_id", "str"),
        ("child_token_id", "str"),
        ("parent_yes_ask", "float"),
        ("child_no_ask", "float"),
        ("parent_yes_bid", "float"),
        ("child_no_bid", "float"),
        ("gross_total_cost", "float"),
        ("estimated_fee_total_per_unit", "float"),
        ("slippage_buffer", "float"),
        ("net_total_cost", "float"),
        ("entry_threshold", "float"),
        ("distance_to_entry", "float"),
        ("worst_case_profit_per_unit", "float"),
        ("best_case_profit_per_unit", "float"),
        ("max_executable_size", "float"),
        ("classification", "str"),
        ("rejection_reason", "str"),
        ("optimal_size", "float"),
        ("optimal_required_capital", "float"),
        ("optimal_guaranteed_profit", "float"),
        ("optimal_net_cost_per_unit", "float"),
        ("spread_check_passed", "bool"),
        ("depth_check_passed", "bool"),
        ("edge_check_passed", "bool"),
    ],
    "n_leg_candidates": [
        ("n_leg_candidate_id", "str"),
        ("scan_id", "str"),
        ("opportunity_id", "str"),
        ("name", "str"),
        ("event_date", "str"),
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
}


_COMMON_COLUMNS: list[tuple[str, str]] = [("run_id", "str"), ("details_json", "str")]


class AuditDependencyError(RuntimeError):
    pass


class ForensicAudit:
    def __init__(self, config: AuditConfig, *, command: list[str] | None = None) -> None:
        self.config = config
        self.run_id = config.run_id or _default_run_id()
        self.run_dir = config.audit_dir / self.run_id
        self.command = command or sys.argv
        self.status = "unknown"
        self.started_wall_ns: int | None = None
        self.ended_wall_ns: int | None = None
        self._pa: Any = None
        self._pq: Any = None
        self._schemas: dict[str, Any] = {}
        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._parts: dict[str, int] = defaultdict(int)
        self._counters: dict[str, int] = defaultdict(int)
        self._last_flush_perf = time.perf_counter()
        self._activity_by_token: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        self._last_activity_ns_by_token: dict[str, int] = {}
        self._last_error: str | None = None
        self._background_tasks: list[Any] = []
        self._write_queue: queue.Queue[tuple[str, list[dict[str, Any]]] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_errors: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self, *, settings: Any | None = None) -> None:
        if not self.enabled:
            return
        self._load_pyarrow()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for table in _TABLES:
            (self.run_dir / f"{table}.parquet").mkdir(exist_ok=True)
        self._start_writer()
        self.status = "running"
        self.started_wall_ns = time.time_ns()
        self._write_manifest(settings=settings)

    def close(self, *, status: str = "completed", error: BaseException | str | None = None, settings: Any | None = None) -> None:
        if not self.enabled:
            return
        if error is not None:
            self._last_error = str(error)
        self.status = status
        self.ended_wall_ns = time.time_ns()
        self.flush(force=True)
        self._stop_writer()
        self._write_manifest(settings=settings)

    def _start_writer(self) -> None:
        if self._writer_thread is not None:
            return
        self._write_queue = queue.Queue(maxsize=32)
        self._writer_thread = threading.Thread(target=self._writer_loop, name="polymarket-audit-writer", daemon=True)
        self._writer_thread.start()

    def _stop_writer(self) -> None:
        if self._write_queue is not None:
            self._write_queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10)
        self._writer_thread = None
        self._write_queue = None
        if self._writer_errors and self._last_error is None:
            self._last_error = "; ".join(self._writer_errors[-3:])

    def _writer_loop(self) -> None:
        assert self._write_queue is not None
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                table, rows = item
                self._write_part_now(table, rows)
            except Exception as exc:  # noqa: BLE001 - audit errors should surface in manifest, not kill trading.
                self._writer_errors.append(str(exc))
            finally:
                self._write_queue.task_done()

    def start_background_tasks(self, clob_client: Any) -> None:
        if not self.enabled:
            return
        import asyncio

        if self.config.clock_sync_seconds > 0:
            self._background_tasks.append(asyncio.create_task(self._clock_sync_loop(clob_client)))
        if self.config.network_probe_seconds > 0:
            self._background_tasks.append(asyncio.create_task(self._network_probe_loop(clob_client)))

    async def stop_background_tasks(self) -> None:
        if not self._background_tasks:
            return
        import asyncio

        tasks = list(self._background_tasks)
        self._background_tasks = []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _clock_sync_loop(self, clob_client: Any) -> None:
        import asyncio

        while self.status == "running":
            try:
                await clob_client.get_server_time(audit_source="clock_sync")
            except Exception as exc:  # noqa: BLE001 - audit must not kill trading.
                self.record("system_metrics", {"metric_id": self.next_id("metric"), "details_json": json.dumps({"clock_sync_error": str(exc)})})
            await asyncio.sleep(max(1.0, self.config.clock_sync_seconds))

    async def _network_probe_loop(self, clob_client: Any) -> None:
        import asyncio

        while self.status == "running":
            try:
                await clob_client.get_server_time(audit_source="network_probe")
            except Exception as exc:  # noqa: BLE001
                self.record("system_metrics", {"metric_id": self.next_id("metric"), "details_json": json.dumps({"network_probe_error": str(exc)})})
            await asyncio.sleep(max(1.0, self.config.network_probe_seconds))

    def next_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{self.run_id}:{prefix}:{self._counters[prefix]:012d}"

    @contextlib.contextmanager
    def context(
        self,
        *,
        scan_id: str | None = None,
        decision_id: str | None = None,
        opportunity_id: str | None = None,
        order_id: str | None = None,
    ) -> Any:
        if not self.enabled:
            yield
            return
        current = dict(_AUDIT_CONTEXT.get())
        next_context = {
            "scan_id": scan_id if scan_id is not None else current.get("scan_id"),
            "decision_id": decision_id if decision_id is not None else current.get("decision_id"),
            "opportunity_id": opportunity_id if opportunity_id is not None else current.get("opportunity_id"),
            "order_id": order_id if order_id is not None else current.get("order_id"),
        }
        token = _AUDIT_CONTEXT.set(next_context)
        try:
            yield
        finally:
            _AUDIT_CONTEXT.reset(token)

    def current_context(self) -> dict[str, str | None]:
        return dict(_AUDIT_CONTEXT.get()) if self.enabled else {}

    def recent_activity_score(self, token_ids: list[str]) -> float | None:
        if not self.enabled or not token_ids:
            return None
        scores: list[float] = []
        now_ns = time.time_ns()
        cutoff = now_ns - 1_000_000_000
        for token_id in token_ids:
            window = self._activity_by_token.get(str(token_id))
            if not window:
                continue
            recent_events = [kind for event_ns, kind in window if event_ns >= cutoff]
            recent_trades = sum(1 for kind in recent_events if kind == "last_trade_price")
            scores.append(min(1.0, (len(recent_events) / 20.0) + (recent_trades / 10.0)))
        return max(scores) if scores else None

    def record(self, table: str, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if table not in _TABLES:
            raise KeyError(f"Unknown audit table: {table}")
        normalized = self._normalize(table, row)
        self._buffers[table].append(normalized)
        if time.perf_counter() - self._last_flush_perf >= max(0.1, self.config.flush_seconds):
            self.flush()

    def record_network(
        self,
        *,
        method: str,
        endpoint: str,
        status_code: int | None,
        wall_start_ns: int,
        wall_end_ns: int,
        perf_start_ns: int,
        perf_end_ns: int,
        request_bytes: int | None = None,
        response_bytes: int | None = None,
        queue_wait_ms: float | None = None,
        semaphore_wait_ms: float | None = None,
        backoff_ms: float | None = None,
        exception: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        ctx = self.current_context()
        code = int(status_code or 0)
        self.record(
            "network",
            {
                "network_id": self.next_id("network"),
                "scan_id": ctx.get("scan_id"),
                "decision_id": ctx.get("decision_id"),
                "opportunity_id": ctx.get("opportunity_id"),
                "order_id": ctx.get("order_id"),
                "method": method.upper(),
                "endpoint": endpoint,
                "status_code": code,
                "is_429": code == 429,
                "is_425": code == 425,
                "is_5xx": code >= 500,
                "queue_wait_ms": queue_wait_ms,
                "semaphore_wait_ms": semaphore_wait_ms,
                "backoff_ms": backoff_ms,
                "latency_ms": (perf_end_ns - perf_start_ns) / 1_000_000,
                "request_bytes": request_bytes,
                "response_bytes": response_bytes,
                "exception": exception,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "perf_start_ns": perf_start_ns,
                "perf_end_ns": perf_end_ns,
            },
        )

    def record_clock_sync(
        self,
        *,
        wall_send_ns: int,
        wall_recv_ns: int,
        perf_send_ns: int,
        perf_recv_ns: int,
        server_time_seconds: int | float,
        source: str,
    ) -> None:
        if not self.enabled:
            return
        server_time_ns = int(float(server_time_seconds) * 1_000_000_000)
        midpoint_wall_ns = wall_send_ns + ((wall_recv_ns - wall_send_ns) // 2)
        rtt_ms = (perf_recv_ns - perf_send_ns) / 1_000_000
        self.record(
            "clock_sync",
            {
                "sample_id": self.next_id("clock"),
                "wall_send_ns": wall_send_ns,
                "wall_recv_ns": wall_recv_ns,
                "perf_send_ns": perf_send_ns,
                "perf_recv_ns": perf_recv_ns,
                "server_time_ns": server_time_ns,
                "rtt_ms": rtt_ms,
                "estimated_skew_ms": (server_time_ns - midpoint_wall_ns) / 1_000_000,
                "uncertainty_ms": rtt_ms / 2,
                "source": source,
            },
        )

    def record_ws_event(
        self,
        *,
        payload: dict[str, Any],
        raw_message: str | bytes | None,
        local_receipt_ts_ns: int,
        receipt_perf_ns: int,
        processing_ts_ns: int,
        processing_perf_ns: int,
    ) -> str | None:
        if not self.enabled:
            return None
        event_id = self.next_id("ws")
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        token_ids = _payload_token_ids(payload)
        exchange_ts_ns = _timestamp_to_ns(payload.get("timestamp"))
        latency_ms = (
            (local_receipt_ts_ns - exchange_ts_ns) / 1_000_000
            if exchange_ts_ns is not None
            else None
        )
        raw = _safe_json(payload) if self.config.raw_ws else None
        hash_value = _first_present(payload, "hash", "event_hash", "id")
        sequence = _first_present(payload, "sequence", "seq", "event_id", "eventId")
        payload_bytes = len(raw_message) if isinstance(raw_message, (bytes, str)) else len(raw or "")
        event_index = self._counters["ws"]
        self.record(
            "ws_events",
            {
                "event_id": event_id,
                "event_index": event_index,
                "event_type": event_type,
                "market": payload.get("market"),
                "token_ids_json": json.dumps(token_ids),
                "exchange_ts_ns": exchange_ts_ns,
                "local_receipt_ts_ns": local_receipt_ts_ns,
                "processing_ts_ns": processing_ts_ns,
                "receipt_perf_ns": receipt_perf_ns,
                "processing_perf_ns": processing_perf_ns,
                "latency_ms": latency_ms,
                "hash": hash_value,
                "sequence": sequence,
                "payload_bytes": payload_bytes,
                "raw_json": raw,
            },
        )
        self._record_market_activity(event_id, event_type, payload, token_ids, exchange_ts_ns, local_receipt_ts_ns)
        return event_id

    def _record_market_activity(
        self,
        event_id: str,
        event_type: str,
        payload: dict[str, Any],
        token_ids: list[str],
        exchange_ts_ns: int | None,
        local_receipt_ts_ns: int,
    ) -> None:
        if not token_ids:
            token_ids = [""]
        burst_size = max(1, len(token_ids))
        for token_id in token_ids:
            last_ns = self._last_activity_ns_by_token.get(token_id)
            interarrival_ms = (local_receipt_ts_ns - last_ns) / 1_000_000 if last_ns else None
            self._last_activity_ns_by_token[token_id] = local_receipt_ts_ns
            window = self._activity_by_token[token_id]
            window.append((local_receipt_ts_ns, event_type))
            cutoff = local_receipt_ts_ns - 1_000_000_000
            while window and window[0][0] < cutoff:
                window.popleft()
            recent_events = len(window)
            recent_trades = sum(1 for _, kind in window if kind == "last_trade_price")
            price_changes = 1 if event_type == "price_change" else 0
            trade_count = 1 if event_type == "last_trade_price" else 0
            contested_score = min(1.0, (recent_events / 20.0) + (recent_trades / 10.0))
            self.record(
                "market_activity",
                {
                    "activity_id": self.next_id("activity"),
                    "event_id": event_id,
                    "event_type": event_type,
                    "market": payload.get("market"),
                    "token_id": token_id,
                    "exchange_ts_ns": exchange_ts_ns,
                    "local_receipt_ts_ns": local_receipt_ts_ns,
                    "interarrival_ms": interarrival_ms,
                    "recent_event_count_1000ms": recent_events,
                    "recent_trade_count_1000ms": recent_trades,
                    "price_change_count": price_changes,
                    "last_trade_count": trade_count,
                    "update_burst_size": burst_size,
                    "contested_score": contested_score,
                },
            )

    def flush(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        wrote_any = False
        for table, rows in list(self._buffers.items()):
            if not rows:
                continue
            self._enqueue_part(table, rows)
            self._buffers[table] = []
            wrote_any = True
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if wrote_any or force:
            self._last_flush_perf = time.perf_counter()
            self._write_system_metric(flush_ms=elapsed_ms)
        if force:
            self._wait_for_writer()

    def _enqueue_part(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self._write_queue is None:
            self._write_part_now(table, rows)
            return
        self._write_queue.put((table, rows))

    def _wait_for_writer(self) -> None:
        if self._write_queue is not None:
            self._write_queue.join()
        if self._writer_errors and self._last_error is None:
            self._last_error = "; ".join(self._writer_errors[-3:])

    def _write_part_now(self, table: str, rows: list[dict[str, Any]]) -> None:
        self._load_pyarrow()
        schema = self._schemas[table]
        table_obj = self._pa.Table.from_pylist(rows, schema=schema)
        self._parts[table] += 1
        part_path = self.run_dir / f"{table}.parquet" / f"part-{self._parts[table]:06d}.parquet"
        tmp_path = part_path.with_suffix(".tmp")
        self._pq.write_table(table_obj, tmp_path, compression=self.config.compression)
        tmp_path.replace(part_path)

    def _write_system_metric(self, *, flush_ms: float | None = None) -> None:
        if not self.enabled:
            return
        rss_mb = _rss_mb()
        self._enqueue_part(
            "system_metrics",
            [
                self._normalize(
                    "system_metrics",
                    {
                        "metric_id": self.next_id("metric"),
                        "timestamp_ns": time.time_ns(),
                        "perf_ns": time.perf_counter_ns(),
                        "audit_queue_depth": sum(len(rows) for rows in self._buffers.values()),
                        "flush_ms": flush_ms,
                        "event_loop_lag_ms": None,
                        "rss_mb": rss_mb,
                        "open_table_count": len([rows for rows in self._buffers.values() if rows]),
                    },
                )
            ],
        )

    def _normalize(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        columns = _COMMON_COLUMNS + _TABLES[table]
        names = {name for name, _ in columns}
        extra = {key: _json_safe(value) for key, value in row.items() if key not in names}
        if "details_json" in row and row["details_json"]:
            try:
                existing = json.loads(str(row["details_json"]))
                if isinstance(existing, dict):
                    extra = {**existing, **extra}
            except json.JSONDecodeError:
                extra["_details"] = row["details_json"]
        normalized: dict[str, Any] = {"run_id": self.run_id, "details_json": json.dumps(extra, sort_keys=True) if extra else None}
        for name, type_name in _TABLES[table]:
            normalized[name] = _coerce_value(row.get(name), type_name)
        return normalized

    def _load_pyarrow(self) -> None:
        if self._pa is not None and self._pq is not None:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:  # pragma: no cover - depends on optional local extra.
            raise AuditDependencyError("Install forensic audit dependencies with: python -m pip install -e '.[audit]'") from exc
        self._pa = pa
        self._pq = pq
        self._schemas = {
            table: pa.schema([(name, getattr(pa, _TYPE_TO_PA[type_name])()) for name, type_name in _COMMON_COLUMNS + fields])
            for table, fields in _TABLES.items()
        }

    def _write_manifest(self, *, settings: Any | None = None) -> None:
        payload = {
            "run_id": self.run_id,
            "status": self.status,
            "started_wall_ns": self.started_wall_ns,
            "ended_wall_ns": self.ended_wall_ns,
            "started_at": _iso_from_ns(self.started_wall_ns),
            "ended_at": _iso_from_ns(self.ended_wall_ns),
            "command": self.command,
            "audit_config": _json_safe(asdict(self.config)),
            "settings": _json_safe(asdict(settings)) if is_dataclass(settings) else _json_safe(settings),
            "git": _git_state(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "version": platform.version(),
            },
            "pip_freeze": _pip_freeze(),
            "redacted_env": _redacted_env(),
            "tables": sorted(_TABLES),
            "table_format": "partitioned_parquet_directories",
            "last_error": self._last_error,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.run_dir / "manifest.json.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.run_dir / "manifest.json")


def _default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{os.getpid()}"


def _coerce_value(value: Any, type_name: str) -> Any:
    if value is None:
        return None
    if type_name == "str":
        if isinstance(value, str):
            return value
        return json.dumps(_json_safe(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
    if type_name == "int":
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if type_name == "float":
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if type_name == "bool":
        return bool(value)
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not _looks_secret(str(key))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _safe_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in ["key", "secret", "passphrase", "private", "signature", "token"])


def _redacted_env() -> dict[str, str]:
    prefixes = ("POLYMARKET_", "KALSHI_")
    return {key: "<redacted>" for key in sorted(os.environ) if key.startswith(prefixes)}


def _git_state() -> dict[str, Any]:
    cwd = Path.cwd()
    return {
        "cwd": str(cwd),
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "branch": _run_text(["git", "branch", "--show-current"]),
        "dirty": bool(_run_text(["git", "status", "--porcelain"])),
        "status_short": _run_text(["git", "status", "--short"], limit=20_000),
    }


def _pip_freeze() -> list[str]:
    output = _run_text([sys.executable, "-m", "pip", "freeze"], limit=200_000)
    return [line for line in output.splitlines() if line.strip()]


def _run_text(cmd: list[str], *, limit: int = 10_000) -> str:
    try:
        result = subprocess.run(cmd, cwd=Path.cwd(), check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return ""
    return (result.stdout or result.stderr or "")[:limit].strip()


def _iso_from_ns(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC).isoformat()


def _timestamp_to_ns(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = int(str(value))
    except (TypeError, ValueError):
        return None
    if numeric > 10_000_000_000:
        return numeric * 1_000_000
    return numeric * 1_000_000_000


def _payload_token_ids(payload: dict[str, Any]) -> list[str]:
    token_ids: list[str] = []
    for key in ["asset_id", "assetId", "token_id", "tokenId"]:
        value = payload.get(key)
        if value not in (None, ""):
            token_ids.append(str(value))
    for change in payload.get("price_changes") or []:
        if isinstance(change, dict):
            token_id = change.get("asset_id") or change.get("assetId") or change.get("token_id") or change.get("tokenId")
            if token_id not in (None, ""):
                token_ids.append(str(token_id))
    for key in ["assets_ids", "asset_ids", "clob_token_ids"]:
        value = payload.get(key)
        if isinstance(value, list):
            token_ids.extend(str(item) for item in value if item not in (None, ""))
    return list(dict.fromkeys(token_ids))


def _first_present(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    for change in payload.get("price_changes") or []:
        if isinstance(change, dict):
            for key in keys:
                value = change.get(key)
                if value not in (None, ""):
                    return str(value)
    return None


def _rss_mb() -> float | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KiB.
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return None
