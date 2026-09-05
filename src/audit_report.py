from __future__ import annotations

import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class AuditReportError(RuntimeError):
    pass


def generate_audit_report(
    run_dir: Path,
    *,
    compare_dir: Path | None = None,
    top_missed: int = 25,
    formats: set[str] | None = None,
) -> dict[str, Any]:
    formats = formats or {"html", "markdown", "json"}
    primary = _load_run(run_dir)
    comparison = _load_run(compare_dir) if compare_dir is not None else None
    summary = _summarize(primary, top_missed=top_missed)
    if comparison is not None:
        summary["compare"] = _compare(summary, _summarize(comparison, top_missed=top_missed))
        summary["compare"]["other_run_id"] = comparison["manifest"].get("run_id")

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    if "json" in formats:
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if "markdown" in formats:
        (report_dir / "summary.md").write_text(_markdown_report(summary), encoding="utf-8")
    if "html" in formats:
        (report_dir / "index.html").write_text(_html_report(summary), encoding="utf-8")
    return summary


def _load_run(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        raise AuditReportError("Missing audit run directory.")
    if not run_dir.exists():
        raise AuditReportError(f"Audit run directory does not exist: {run_dir}")
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise AuditReportError("Install forensic audit dependencies with: python -m pip install -e '.[audit]'") from exc

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"status": "unknown", "run_id": run_dir.name}
    tables: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for table in [
        "clock_sync",
        "network",
        "ws_events",
        "book_snapshots",
        "book_levels",
        "market_activity",
        "pair_observations",
        "n_leg_candidates",
        "portfolio_snapshots",
        "decisions",
        "orders",
        "missed_fills",
        "opportunity_windows",
        "timeline_events",
        "system_metrics",
    ]:
        path = run_dir / f"{table}.parquet"
        rows = _read_parquet_rows(pq, path)
        if not rows:
            missing.append(table)
        tables[table] = rows
    return {"run_dir": run_dir, "manifest": manifest, "tables": tables, "missing_tables": missing}


def _read_parquet_rows(pq: Any, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_dir() and not any(path.glob("*.parquet")):
        return []
    try:
        return pq.read_table(path).to_pylist()
    except Exception:
        if path.is_dir():
            rows: list[dict[str, Any]] = []
            for part in sorted(path.glob("*.parquet")):
                try:
                    rows.extend(pq.read_table(part).to_pylist())
                except Exception:
                    continue
            return rows
        return []


def _summarize(run: dict[str, Any], *, top_missed: int) -> dict[str, Any]:
    tables = run["tables"]
    manifest = run["manifest"]
    decisions = tables["decisions"]
    observations = tables["pair_observations"]
    n_leg = tables["n_leg_candidates"]
    network = tables["network"]
    ws_events = tables["ws_events"]
    missed = tables["missed_fills"]
    activity = tables["market_activity"]
    timelines = tables["timeline_events"]

    top_missed_rows = sorted(
        missed,
        key=lambda row: (
            -(row.get("expected_profit") or 0.0),
            -(row.get("edge") or 0.0),
        ),
    )[:top_missed]
    timeline_by_opportunity = defaultdict(list)
    for row in timelines:
        opportunity_id = row.get("opportunity_id")
        if opportunity_id:
            timeline_by_opportunity[opportunity_id].append(row)
    top_timelines = {
        str(row.get("opportunity_id")): sorted(
            timeline_by_opportunity.get(row.get("opportunity_id"), []),
            key=lambda item: item.get("timestamp_ns") or 0,
        )[:250]
        for row in top_missed_rows
        if row.get("opportunity_id")
    }

    summary = {
        "run_id": manifest.get("run_id") or run["run_dir"].name,
        "status": manifest.get("status", "unknown"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "run_dir": str(run["run_dir"]),
        "missing_tables": run["missing_tables"],
        "row_counts": {table: len(rows) for table, rows in tables.items()},
        "opportunity_funnel": _opportunity_funnel(observations, n_leg, decisions),
        "latency": {
            "network_ms": _percentiles([row.get("latency_ms") for row in network]),
            "websocket_event_lag_ms": _percentiles([row.get("latency_ms") for row in ws_events]),
            "book_to_detection_ms": _percentiles([row.get("book_to_detection_ms") for row in decisions]),
            "detection_to_decision_ms": _percentiles([row.get("detection_to_decision_ms") for row in decisions]),
            "decision_to_ack_ms": _percentiles([row.get("decision_to_ack_ms") for row in decisions]),
        },
        "rate_limits": {
            "http_429_count": sum(1 for row in network if row.get("is_429")),
            "http_425_count": sum(1 for row in network if row.get("is_425")),
            "http_5xx_count": sum(1 for row in network if row.get("is_5xx")),
            "slowest_requests": sorted(network, key=lambda row: row.get("latency_ms") or 0, reverse=True)[:10],
        },
        "miss_reasons": dict(Counter(str(row.get("classification") or row.get("reason") or "unknown") for row in missed)),
        "decision_outcomes": dict(Counter(str(row.get("outcome") or "unknown") for row in decisions)),
        "market_activity": _market_activity_summary(activity),
        "top_missed": top_missed_rows,
        "top_timelines": top_timelines,
        "competitive_diagnosis": _diagnose(tables),
    }
    return summary


def _opportunity_funnel(
    observations: list[dict[str, Any]],
    n_leg: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, int]:
    found = len(observations) + len(n_leg)
    spread_ok = sum(1 for row in observations if row.get("spread_check_passed")) + len(n_leg)
    depth_ok = sum(1 for row in observations if row.get("depth_check_passed")) + sum(
        1 for row in n_leg if row.get("classification") in {"EXECUTABLE_ARBITRAGE_CANDIDATE", "NEAR_ARBITRAGE"}
    )
    edge_ok = sum(1 for row in observations if row.get("edge_check_passed")) + sum(
        1 for row in n_leg if (row.get("gross_edge") or 0) > 0
    )
    capital_ok = sum(1 for row in decisions if row.get("passed_capital_check"))
    submitted = sum(1 for row in decisions if row.get("submitted"))
    filled = sum(1 for row in decisions if row.get("filled"))
    return {
        "found": found,
        "spread_ok": spread_ok,
        "depth_ok": depth_ok,
        "edge_ok": edge_ok,
        "capital_ok": capital_ok,
        "submitted": submitted,
        "filled": filled,
    }


def _market_activity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_token[str(row.get("token_id") or "")].append(row)
    token_summaries = []
    for token_id, token_rows in by_token.items():
        scores = [row.get("contested_score") for row in token_rows if row.get("contested_score") is not None]
        interarrivals = [row.get("interarrival_ms") for row in token_rows if row.get("interarrival_ms") is not None]
        token_summaries.append(
            {
                "token_id": token_id,
                "events": len(token_rows),
                "last_trade_events": sum(1 for row in token_rows if row.get("event_type") == "last_trade_price"),
                "price_change_events": sum(1 for row in token_rows if row.get("event_type") == "price_change"),
                "max_recent_event_count_1000ms": max((row.get("recent_event_count_1000ms") or 0 for row in token_rows), default=0),
                "avg_contested_score": statistics.mean(scores) if scores else None,
                "p50_interarrival_ms": _percentiles(interarrivals).get("p50"),
            }
        )
    token_summaries.sort(key=lambda row: (row.get("avg_contested_score") or 0, row.get("events") or 0), reverse=True)
    return {
        "tokens": token_summaries[:25],
        "event_count": len(rows),
        "contested_token_count": sum(1 for row in token_summaries if (row.get("avg_contested_score") or 0) >= 0.5),
    }


def _diagnose(tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    network = tables["network"]
    decisions = tables["decisions"]
    missed = tables["missed_fills"]
    ws_events = tables["ws_events"]
    notes: list[str] = []
    net_p95 = _percentiles([row.get("latency_ms") for row in network]).get("p95")
    ws_p95 = _percentiles([row.get("latency_ms") for row in ws_events]).get("p95")
    if net_p95 and net_p95 > 500:
        notes.append(f"REST p95 latency is high at {net_p95:.1f} ms.")
    if ws_p95 and ws_p95 > 250:
        notes.append(f"WebSocket event lag p95 is high at {ws_p95:.1f} ms.")
    if any(row.get("is_429") for row in network):
        notes.append("HTTP 429s were observed; rate limiting is a direct competitive disadvantage.")
    capital_blocks = sum(1 for row in decisions if "capital" in str(row.get("skip_reason") or "").lower() or "cash" in str(row.get("skip_reason") or "").lower())
    if capital_blocks:
        notes.append(f"{capital_blocks} decisions were blocked by cash/capital limits.")
    contested_misses = sum(1 for row in missed if (row.get("market_activity_score") or 0) >= 0.5)
    if contested_misses:
        notes.append(f"{contested_misses} missed opportunities happened during contested market bursts.")
    if not notes:
        notes.append("No dominant competitive bottleneck is obvious from the available audit tables.")
    return notes


def _compare(primary: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    keys = ["network_ms", "websocket_event_lag_ms", "book_to_detection_ms", "detection_to_decision_ms", "decision_to_ack_ms"]
    latency_delta = {}
    for key in keys:
        latency_delta[key] = {
            "primary_p95": primary["latency"].get(key, {}).get("p95"),
            "other_p95": other["latency"].get(key, {}).get("p95"),
            "delta_p95": _delta(primary["latency"].get(key, {}).get("p95"), other["latency"].get(key, {}).get("p95")),
        }
    funnel_delta = {
        key: primary["opportunity_funnel"].get(key, 0) - other["opportunity_funnel"].get(key, 0)
        for key in sorted(set(primary["opportunity_funnel"]) | set(other["opportunity_funnel"]))
    }
    return {
        "latency_delta": latency_delta,
        "funnel_delta": funnel_delta,
        "miss_reason_delta": _counter_delta(primary.get("miss_reasons", {}), other.get("miss_reasons", {})),
        "rate_limit_delta": {
            "http_429_count": primary["rate_limits"]["http_429_count"] - other["rate_limits"]["http_429_count"],
            "http_425_count": primary["rate_limits"]["http_425_count"] - other["rate_limits"]["http_425_count"],
            "http_5xx_count": primary["rate_limits"]["http_5xx_count"] - other["rate_limits"]["http_5xx_count"],
        },
    }


def _counter_delta(primary: dict[str, int], other: dict[str, int]) -> dict[str, int]:
    return {key: primary.get(key, 0) - other.get(key, 0) for key in sorted(set(primary) | set(other))}


def _delta(primary: float | None, other: float | None) -> float | None:
    if primary is None or other is None:
        return None
    return primary - other


def _percentiles(values: list[Any]) -> dict[str, float | None]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(clean),
        "p50": _percentile(clean, 50),
        "p95": _percentile(clean, 95),
        "p99": _percentile(clean, 99),
        "max": max(clean),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Competitive Forensic Audit Report",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Started: {summary.get('started_at') or 'unknown'}",
        f"- Ended: {summary.get('ended_at') or 'unknown'}",
        "",
        "## Opportunity Funnel",
        "",
        _markdown_kv(summary["opportunity_funnel"]),
        "",
        "## Latency",
        "",
        _markdown_nested(summary["latency"]),
        "",
        "## Miss Reasons",
        "",
        _markdown_kv(summary["miss_reasons"]),
        "",
        "## Competitive Diagnosis",
        "",
        "\n".join(f"- {item}" for item in summary["competitive_diagnosis"]),
        "",
        "## Top Missed Opportunities",
        "",
        _markdown_rows(summary["top_missed"], ["candidate_name", "classification", "reason", "edge", "expected_profit", "market_activity_score"]),
    ]
    if summary.get("compare"):
        lines += ["", "## Run Comparison", "", _markdown_nested(summary["compare"])]
    return "\n".join(lines) + "\n"


def _html_report(summary: dict[str, Any]) -> str:
    funnel_svg = _bar_svg(summary["opportunity_funnel"], title="Opportunity funnel")
    miss_svg = _bar_svg(summary["miss_reasons"], title="Miss reasons")
    latency_rows = _html_nested(summary["latency"])
    timeline_html = _html_timelines(summary)
    compare_html = f"<h2>Run Comparison</h2>{_html_nested(summary['compare'])}" if summary.get("compare") else ""
    missing = ", ".join(summary["missing_tables"]) if summary["missing_tables"] else "none"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Forensic Audit {html.escape(str(summary['run_id']))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #172026; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde2; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f8; }}
    code {{ background: #f4f6f8; padding: 2px 4px; border-radius: 3px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
    .note {{ background: #fff8db; padding: 10px 12px; border: 1px solid #ead98c; }}
    details {{ margin: 12px 0; }}
    svg {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Competitive Forensic Audit Report</h1>
  <p><b>Run:</b> <code>{html.escape(str(summary['run_id']))}</code> <b>Status:</b> <code>{html.escape(str(summary['status']))}</code></p>
  <p><b>Started:</b> {html.escape(str(summary.get('started_at') or 'unknown'))} <b>Ended:</b> {html.escape(str(summary.get('ended_at') or 'unknown'))}</p>
  <p class="note"><b>Missing/empty tables:</b> {html.escape(missing)}. Incomplete runs are expected to have gaps; available data is still analyzed.</p>
  <div class="grid"><section>{funnel_svg}</section><section>{miss_svg}</section></div>
  <h2>Latency Waterfall Metrics</h2>
  {latency_rows}
  <h2>Competitive Diagnosis</h2>
  <ul>{''.join(f'<li>{html.escape(item)}</li>' for item in summary['competitive_diagnosis'])}</ul>
  <h2>Ambient Market Activity</h2>
  {_html_rows(summary['market_activity'].get('tokens', []), ['token_id', 'events', 'last_trade_events', 'price_change_events', 'max_recent_event_count_1000ms', 'avg_contested_score', 'p50_interarrival_ms'])}
  <h2>Top Missed Opportunities</h2>
  {_html_rows(summary['top_missed'], ['candidate_name', 'classification', 'reason', 'edge', 'expected_profit', 'market_activity_score'])}
  <h2>Chronological Competitive Timelines</h2>
  {timeline_html}
  {compare_html}
</body>
</html>
"""


def _bar_svg(values: dict[str, Any], *, title: str) -> str:
    items = [(str(key), float(value or 0)) for key, value in values.items()]
    if not items:
        return f"<h2>{html.escape(title)}</h2><p>No data.</p>"
    max_value = max(value for _, value in items) or 1.0
    width = 520
    row_h = 26
    height = 40 + len(items) * row_h
    rows = [f"<text x='0' y='18' font-size='16' font-weight='600'>{html.escape(title)}</text>"]
    for idx, (label, value) in enumerate(items):
        y = 36 + idx * row_h
        bar_w = int((width - 190) * value / max_value)
        rows.append(f"<text x='0' y='{y + 15}' font-size='12'>{html.escape(label)}</text>")
        rows.append(f"<rect x='170' y='{y}' width='{bar_w}' height='18' fill='#3777b8'></rect>")
        rows.append(f"<text x='{180 + bar_w}' y='{y + 14}' font-size='12'>{value:g}</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(rows)}</svg>"


def _html_timelines(summary: dict[str, Any]) -> str:
    if not summary.get("top_timelines"):
        return "<p>No timeline events available for missed opportunities.</p>"
    parts = []
    for opportunity_id, rows in summary["top_timelines"].items():
        parts.append(
            "<details><summary><code>"
            + html.escape(str(opportunity_id))
            + "</code></summary>"
            + _html_rows(rows, ["event_kind", "candidate_name", "timestamp_ns", "metric_1", "metric_2", "message"])
            + "</details>"
        )
    return "\n".join(parts)


def _html_nested(payload: dict[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        rows.append({"metric": key, "value": json.dumps(value, sort_keys=True)})
    return _html_rows(rows, ["metric", "value"])


def _html_rows(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No data.</p>"
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(_display(row.get(column)))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _markdown_kv(payload: dict[str, Any]) -> str:
    return "\n".join(f"- `{key}`: {value}" for key, value in payload.items()) or "_No data._"


def _markdown_nested(payload: dict[str, Any]) -> str:
    return "\n".join(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`" for key, value in payload.items()) or "_No data._"


def _markdown_rows(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data._"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_display(row.get(column)).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(out)


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)

