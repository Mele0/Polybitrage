"""Offline analyzer for the latency observatory (scan stages + full pipeline).

Reads the artifacts written by ``src.profiling`` and prints, in order:
  1. the latest per-stage **scan** percentile table (from ``stages.csv``),
  2. the **pipeline** latency table — the tick-to-trade *ends* (wire / queue /
     wakeup / loop_lag / order_rtt / book_age / gc_pause),
  3. a **tick-to-trade budget**: where the milliseconds actually go, end to end,
     and which leg dominates (with the concrete fix to consider),
  4. **counters** + the **missed-opportunity reason histogram** (from
     ``counters.json``) — *why* near-arbs failed to convert, and
  5. the slowest scans with full per-stage attribution (from ``slow_scans.jsonl``).

Usage:
    python -m src.profile_report --profile-dir reports/profile [--top 20]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Top-level scan stages (s2_get_books already contains s2a + s2b).
TOP_STAGES: tuple[str, ...] = (
    "s1_dirtyset",
    "s2_get_books",
    "s3_fast_observe_all",
    "s4_nleg",
    "s5_filter",
    "s6_try_enter",
)

# Pipeline series → one-line meaning (display order).
PIPELINE_SERIES: dict[str, str] = {
    "wire_latency": "exchange event → local receipt (network + exchange egress)",
    "queue_wait": "WS update enqueued → drained by the loop (firehose backlog)",
    "wakeup_latency": "data-ready → scan resumes (executor / OS scheduler handoff)",
    "loop_lag": "event-loop scheduling lag (blocking call / GC / OS preemption)",
    "scan_interval": "wall gap between consecutive scans (cadence)",
    "book_age_at_decision": "staleness of the books we acted on",
    "decision_to_submit": "candidate identified → order submitted",
    "order_rtt": "order submitted → exchange ack (live mode)",
    "gc_pause": "individual GC pause durations",
}

# Pipeline series → what a high tail implies and the lever to pull.
PIPELINE_HINTS: dict[str, str] = {
    "wire_latency": "Network-bound. Co-locate / move VPS region closer — quantify with scripts/latency_probe.py.",
    "queue_wait": "Loop is falling behind the WS firehose. Score fewer pairs / speed up the scan; correlate with loop_lag.",
    "wakeup_latency": "Executor↔loop handoff. Check CPU contention, default thread-pool saturation, and the CPU freq governor.",
    "loop_lag": "Event loop starved. Hunt the blocking call (inline REST? large numpy?) or GC — correlate spikes with gc_pause.",
    "scan_interval": "Cadence. Long+bimodal is fine if idle; long while busy means you cannot keep up with updates.",
    "book_age_at_decision": "Acting on stale books. Tighten WS freshness (priority seeding) or lower the stale thresholds.",
    "order_rtt": "Order round-trip. Network + exchange matching — co-locate, keep TLS warm, keep pre-signing on.",
    "gc_pause": "GC spikes. Consider gc.freeze() after warmup, fewer per-scan allocations, or gc.disable() on the hot path.",
}

# The ordered legs that compose the end-to-end tick-to-trade path, with the
# stage/series each is read from and whether it is network (uncontrollable in
# software) or compute (controllable).
BUDGET_LEGS: tuple[tuple[str, str, str], ...] = (
    ("wire_latency", "wire_latency", "network"),
    ("queue_wait", "queue_wait", "compute"),
    ("wakeup_latency", "wakeup_latency", "compute"),
    ("total_scan", "total_scan", "compute"),
    ("order_rtt", "order_rtt", "network"),
)

# Stage → what a slow scan dominated by it implies (from the review decision tree).
HINTS: dict[str, str] = {
    "s2_get_books": "Rust→Python boundary dominates — see s2a vs s2b below.",
    "s3_fast_observe_all": "scan kernel / array-fill dominates — optimize numpy/numba or score fewer pairs.",
    "s1_dirtyset": "dirty-set build dominates — large WS update fan-out; check token→pair mapping size.",
    "s4_nleg": "N-leg scan dominates — vectorize further or cap n_leg ranges.",
    "s5_filter": "filter/list-comp dominates — unusual; check show_top / near-arb set size.",
    "s6_try_enter": "entry/decision region dominates — ensure REST recheck is backgrounded, not inline.",
    "__unaccounted__": "total high but stage-sum low → event-loop lag / GIL / scheduler. Check the loop_lag + gc_pause series above.",
}


def _read_stages(path: Path) -> dict[str, dict[str, Any]]:
    """Return the most recent percentile row per stage."""
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            latest[row["stage"]] = row  # later rows overwrite → keeps the last snapshot
    return latest


def _read_counters(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_slow(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _print_stage_table(stages: dict[str, dict[str, Any]]) -> None:
    print("=== Per-stage scan latency (latest snapshot, ms) ===")
    if not stages:
        print("  (no stages.csv data)\n")
        return
    hdr = f"{'stage':<22}{'count':>10}{'mean':>11}{'p50':>11}{'p90':>11}{'p99':>11}{'p99.9':>11}{'max':>11}"
    print(hdr)
    order = list(TOP_STAGES) + ["s2a_rust_snapshot", "s2b_python_rebuild", "total_scan"]
    for stage in order:
        r = stages.get(stage)
        if not r:
            continue
        print(
            f"{stage:<22}{int(_f(r, 'count')):>10}{_f(r, 'mean_ms'):>11.3f}{_f(r, 'p50_ms'):>11.3f}"
            f"{_f(r, 'p90_ms'):>11.3f}{_f(r, 'p99_ms'):>11.3f}{_f(r, 'p999_ms'):>11.3f}{_f(r, 'max_ms'):>11.3f}"
        )
    print()


def _print_pipeline_table(stages: dict[str, dict[str, Any]]) -> None:
    present = [(s, stages[s]) for s in PIPELINE_SERIES if s in stages]
    print("=== Pipeline latency — tick-to-trade ends (latest snapshot, ms) ===")
    if not present:
        print("  (no pipeline series yet — needs a WS run with --profile)\n")
        return
    hdr = f"{'series':<22}{'count':>10}{'mean':>11}{'p50':>11}{'p90':>11}{'p99':>11}{'p99.9':>11}{'max':>11}"
    print(hdr)
    for series, r in present:
        print(
            f"{series:<22}{int(_f(r, 'count')):>10}{_f(r, 'mean_ms'):>11.3f}{_f(r, 'p50_ms'):>11.3f}"
            f"{_f(r, 'p90_ms'):>11.3f}{_f(r, 'p99_ms'):>11.3f}{_f(r, 'p999_ms'):>11.3f}{_f(r, 'max_ms'):>11.3f}"
        )
    print()
    # Per-series tail hints: flag any series whose p99 is meaningfully worse than
    # its p50 (a fat tail is where opportunities die), plus always-relevant ones.
    print("  Interpretation (high p99 ⇒ the lever to pull):")
    for series, r in present:
        p50, p99 = _f(r, "p50_ms"), _f(r, "p99_ms")
        tail = f"  [p99/p50 ×{p99 / p50:.1f}]" if p50 > 0 and p99 / p50 >= 3 else ""
        print(f"    {series:<22}{PIPELINE_HINTS.get(series, '')}{tail}")
    print()


def _print_budget(stages: dict[str, dict[str, Any]]) -> None:
    """Approximate end-to-end tick-to-trade budget from independent p50s."""
    legs = [(label, stages[src], kind) for label, src, kind in BUDGET_LEGS if src in stages]
    if not legs:
        return
    total = sum(_f(r, "p50_ms") for _, r, _ in legs)
    if total <= 0:
        return
    print("=== Tick-to-trade budget (p50, approximate end-to-end) ===")
    print("  (legs measured independently; sum is indicative, not a single-event trace)")
    net_ms = 0.0
    dom_label, dom_v = "", -1.0
    for label, r, kind in legs:
        v = _f(r, "p50_ms")
        if kind == "network":
            net_ms += v
        if v > dom_v:
            dom_label, dom_v = label, v
        bar = "█" * max(0, round(20 * v / total))
        print(f"  {label:<18}{v:>8.3f} ms  ({100 * v / total:>4.0f}%)  {kind:<8} {bar}")
    print(f"  {'─' * 46}")
    print(f"  {'end-to-end (sum)':<18}{total:>8.3f} ms")
    ctrl = total - net_ms
    print(
        f"  network legs {net_ms:.3f} ms ({100 * net_ms / total:.0f}%, fix by co-location)  |  "
        f"software legs {ctrl:.3f} ms ({100 * ctrl / total:.0f}%, fix in code)"
    )
    print(f"  dominant leg: {dom_label} — {PIPELINE_HINTS.get(dom_label, '')}\n")


def _print_counters(counters: dict[str, Any]) -> None:
    if not counters:
        return
    ev = counters.get("counters") or {}
    missed = counters.get("missed_reasons") or {}
    if ev:
        print("=== Event counters ===")
        scans = float(ev.get("scans") or 0)
        for name, cnt in sorted(ev.items()):
            extra = ""
            if name in ("ws_event_wakeups", "empty_wakeups") and scans:
                extra = f"  ({100 * float(cnt) / scans:.0f}% of scans)"
            print(f"  {name:<26}{int(cnt):>12}{extra}")
        filled = float(ev.get("entries_filled") or 0)
        missed_n = float(ev.get("entries_missed") or 0)
        denom = filled + missed_n
        if denom:
            print(f"  {'fill rate':<26}{100 * filled / denom:>11.1f}%  (filled {int(filled)} / executable {int(denom)})")
        print()
    if missed:
        total = sum(missed.values())
        print(f"=== Missed-opportunity reasons (total {total}) — where executable arbs died ===")
        for reason, cnt in Counter(missed).most_common():
            pct = 100 * cnt / total if total else 0
            print(f"  {reason:<28}{cnt:>8}  ({pct:>4.0f}%)   {_MISS_HINT.get(reason, '')}")
        print()


_MISS_HINT: dict[str, str] = {
    "rest_recheck_lost": "edge gone by REST recheck → too slow / price moved. Cut wire+queue+scan latency.",
    "rest_recheck_missing": "REST returned no quote at recheck → thin/closing market.",
    "depth_depleted": "size eaten before we acted → faster path or smaller target size.",
    "capital_blocked": "out of cash / locked-capital cap → raise budget or free capital faster.",
    "duplicate_guard_blocked": "already holding this pair (dedup) — usually benign.",
    "cooldown_guard_blocked": "per-pair cooldown active — loosen cooldown if leaving edge on the table.",
    "order_failed_or_rejected": "exchange rejected the order (live) — check FOK/size/tick/balance.",
    "edge_or_threshold_lost": "edge slipped below threshold between scan and entry → latency-sensitive.",
    "unknown": "uncategorized — inspect raw action strings.",
}


def _dominant_stage(rec: dict[str, Any]) -> str:
    total = _f(rec, "total_scan_ms")
    stage_sum = sum(_f(rec, s + "_ms") for s in TOP_STAGES)
    if total > 0 and stage_sum < 0.6 * total:
        return "__unaccounted__"
    best, best_v = "__unaccounted__", -1.0
    for s in TOP_STAGES:
        v = _f(rec, s + "_ms")
        if v > best_v:
            best, best_v = s, v
    return best


def _print_slow(slow: list[dict[str, Any]], top: int) -> None:
    print("=== Slowest scans ===")
    if not slow:
        print("  (no slow_scans.jsonl data — no scan exceeded --profile-slow-scan-ms)\n")
        return
    ranked = sorted(slow, key=lambda r: _f(r, "total_scan_ms"), reverse=True)
    shown = ranked[:top]
    cols = ["s1_dirtyset", "s2_get_books", "s2a_rust_snapshot", "s2b_python_rebuild",
            "s3_fast_observe_all", "s4_nleg", "s5_filter", "s6_try_enter"]
    short = {"s1_dirtyset": "s1", "s2_get_books": "s2", "s2a_rust_snapshot": "s2a",
             "s2b_python_rebuild": "s2b", "s3_fast_observe_all": "s3", "s4_nleg": "s4",
             "s5_filter": "s5", "s6_try_enter": "s6"}
    hdr = f"{'total':>7} " + " ".join(f"{short[c]:>5}" for c in cols) + f" | {'dirty':>5}{'uniq':>5} {'nleg':>5} {'enter':>6} {'gc':>3}"
    print(f"(showing {len(shown)} of {len(ranked)} slow scans)")
    print(hdr)
    for r in shown:
        line = f"{_f(r, 'total_scan_ms'):>7.2f} " + " ".join(f"{_f(r, c + '_ms'):>5.1f}" for c in cols)
        line += (
            f" | {int(_f(r, 'dirty_tokens')):>5}{int(_f(r, 'unique_tokens')):>5} "
            f"{str(r.get('n_leg_dirty')):>5} {str(r.get('entered')):>6} "
            f"{'yes' if r.get('gc_active') else 'no':>3}"
        )
        print(line)
    print()

    # ── Spike attribution ──────────────────────────────────────────────────────
    print("=== Spike attribution ===")
    dom = Counter(_dominant_stage(r) for r in ranked)
    n = len(ranked)
    for stage, cnt in dom.most_common():
        label = "total>stages (unaccounted)" if stage == "__unaccounted__" else stage
        print(f"  {label:<28}{cnt:>4}  ({100 * cnt / n:.0f}%)   → {HINTS.get(stage, '')}")
    gc_n = sum(1 for r in ranked if r.get("gc_active"))
    print(f"  GC overlapped {gc_n}/{n} slow scans ({100 * gc_n / n:.0f}%).")

    # s2 split guidance when the boundary is the leader.
    if dom.get("s2_get_books"):
        s2a = sum(_f(r, "s2a_rust_snapshot_ms") for r in ranked)
        s2b = sum(_f(r, "s2b_python_rebuild_ms") for r in ranked)
        if s2b > s2a:
            print("  → within s2: Python rebuild (s2b) > Rust snapshot (s2a). Strong case for get_bba_batch.")
        elif s2a > 0:
            print("  → within s2: Rust snapshot (s2a) ≥ Python rebuild (s2b). Suspect REST fallback/seeding or lock contention, not from_float_data.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze latency-observatory profiler output.")
    ap.add_argument("--profile-dir", type=Path, default=Path("reports/profile"))
    ap.add_argument("--top", type=int, default=20, help="How many slowest scans to list.")
    args = ap.parse_args()

    stages = _read_stages(args.profile_dir / "stages.csv")
    counters = _read_counters(args.profile_dir / "counters.json")
    slow = _read_slow(args.profile_dir / "slow_scans.jsonl")
    print(f"profile dir: {args.profile_dir}\n")
    _print_stage_table(stages)
    _print_pipeline_table(stages)
    _print_budget(stages)
    _print_counters(counters)
    _print_slow(slow, args.top)


if __name__ == "__main__":
    main()
