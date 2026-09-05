"""Low-overhead latency observatory for the arb bot (scan stages + full pipeline).

Answers two questions a live run cannot otherwise answer:

  1. **Where does a slow *scan* spend its time?**  Per-stage latencies for
     ``_scan_once`` (plus the Rust→Python ``get_books`` split) go into preallocated
     ring buffers; percentiles are snapshotted to ``stages.csv`` every few seconds
     and any scan over a threshold is dumped to ``slow_scans.jsonl``.

  2. **Where along the *tick-to-trade* pipeline do we lose opportunities?**  The
     scan kernel is only the middle of the pipeline.  We also record the ends:

       * ``wire_latency``        exchange event → local receipt (network + exchange)
       * ``queue_wait``          WS update enqueued → drained by the main loop
                                 (rises when we are CPU-bound and falling behind)
       * ``wakeup_latency``      data-ready → scan actually resumes (executor / OS
                                 scheduler handoff)
       * ``loop_lag``            event-loop scheduling lag (blocking call / GC / OS
                                 preemption starving every task at once)
       * ``scan_interval``       wall time between consecutive scans (cadence)
       * ``book_age_at_decision`` staleness of the books we actually act on
       * ``decision_to_submit``  candidate identified → order submitted
       * ``order_rtt``           order submitted → exchange ack (live mode)
       * ``gc_pause``            individual GC pause durations

     Plus **counters** (scans, wakeups, entries…) and a **missed-opportunity reason
     histogram** that attributes every near-arb that did *not* convert to a bucket
     (price_moved / insufficient_depth / rest_recheck / insufficient_cash /
     max_positions / inflight / …) — the direct answer to "where am I lacking?".

Design constraints (so a live run can leave it on):
  * Gated by the module-level :data:`ENABLED` flag.  Every hot-path call site
    guards on ``if profiling.ENABLED:`` first, so a non-profiled run pays one bool
    check and nothing else.
  * No per-sample allocation, string-formatting, or logging on the hot path —
    ``record`` / ``record_series`` write one float into a preallocated numpy ring
    buffer; counters are a single dict increment.
  * Percentile computation and all file I/O happen off the hot path in a
    background dumper task (see ``PaperArbSimulator._bg_profile_dump``), typically
    via ``asyncio.to_thread`` so the event loop is never blocked.

Threading note: ``record`` runs on the event-loop thread.  ``record_series`` may
additionally be called from WS I/O threads (``wire_latency``) and from the GC
callback (``gc_pause``); the dumper reads the buffers from a worker thread.  Reads
are statistical (percentiles over many samples) so a benign torn read/write of an
in-flight sample is irrelevant; no lock is taken on the hot path by design.
"""
from __future__ import annotations

import asyncio
import csv
import gc
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

# ── Module-level state (read on the hot path) ────────────────────────────────
ENABLED: bool = False
RECORDER: "LatencyRecorder | None" = None

# Stages recorded once per scan, in pipeline order (pre-registered to avoid a
# first-touch allocation on the hot path).  s2a/s2b are recorded once per token
# inside the Rust WS client and accumulate within a scan.  These feed the
# per-scan accumulator and the slow-scan breakdown.
SCAN_STAGES: tuple[str, ...] = (
    "s1_dirtyset",
    "s2_get_books",
    "s2a_rust_snapshot",
    "s2b_python_rebuild",
    "s3_fast_observe_all",
    "s4_nleg",
    "s5_filter",
    "s6_try_enter",
    "total_scan",
)

# End-to-end pipeline series recorded *outside* the scan-stage accounting (no
# per-scan accumulation).  Pre-registered so their ring buffers exist before the
# first sample.  They share the ring/percentile machinery and are auto-dumped to
# stages.csv, but are kept out of ``_scan_acc`` so they never pollute the
# slow-scan per-stage breakdown or the total_scan stage-sum.
PIPELINE_SERIES: tuple[str, ...] = (
    # ── tick-to-trade pipeline ──
    "wire_latency",
    "queue_wait",
    "wakeup_latency",
    "loop_lag",
    "scan_interval",
    "book_age_at_decision",
    "signing_latency",
    "rest_call_latency",
    "order_rtt",
    "gc_pause",
    # ── alpha / opportunity decay (a duration → fits the ns ring) ──
    "opportunity_lifetime",
)

# Canonical missed-opportunity reasons (mirrors _audit_miss_classification in
# simulator.py).  Pre-seeded to 0 so the Grafana miss panel always has series
# even before any executable opportunity has failed.
KNOWN_MISS_REASONS: tuple[str, ...] = (
    "rest_recheck_lost",
    "rest_recheck_missing",
    "depth_depleted",
    "capital_blocked",
    "duplicate_guard_blocked",
    "cooldown_guard_blocked",
    "order_failed_or_rejected",
    "edge_or_threshold_lost",
    "unknown",
)

# Counters pre-seeded to 0 so their Prometheus series exist from t=0.
_PRESEED_COUNTERS: tuple[str, ...] = (
    "scans", "ws_event_wakeups", "empty_wakeups",
    "entries_filled", "entries_missed", "near_arb_appearances",
)

# Series that also get fixed-bucket histograms (for Grafana latency heatmaps).
# Curated to the high-signal legs so /metrics stays compact.
HEATMAP_SERIES: frozenset[str] = frozenset({
    "wire_latency", "queue_wait", "total_scan", "loop_lag",
    "order_rtt", "rest_call_latency", "scan_interval",
})
# Cumulative bucket upper bounds in ms (Prometheus `le`).  Log-ish spacing from
# sub-100µs to 1s covers everything from a hot scan to a stalled REST call.
_HIST_EDGES_MS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000,
)

_RING_DEFAULT = 200_000  # samples retained per stage (sliding window)

# ── GC activity tracking (event count + per-pause duration) ──────────────────
_GC_EVENTS: int = 0
_GC_START_NS: int = 0
_GC_HOOK_INSTALLED: bool = False


def _gc_callback(phase: str, info: dict[str, Any]) -> None:
    """gc.callbacks hook: count collections and time each pause.

    ``start``/``stop`` bracket one collection.  We stamp the start and, on stop,
    record the elapsed nanoseconds into the ``gc_pause`` series so the report can
    show GC pause percentiles (a classic source of Python latency spikes), not
    just the per-scan ``gc_active`` boolean.
    """
    global _GC_EVENTS, _GC_START_NS
    if phase == "start":
        _GC_START_NS = time.perf_counter_ns()
    elif phase == "stop":
        _GC_EVENTS += 1
        rec = RECORDER
        if ENABLED and rec is not None and _GC_START_NS:
            rec.record_series("gc_pause", time.perf_counter_ns() - _GC_START_NS)


def _install_gc_hook() -> None:
    global _GC_HOOK_INSTALLED
    if not _GC_HOOK_INSTALLED:
        gc.callbacks.append(_gc_callback)
        _GC_HOOK_INSTALLED = True


class LatencyRecorder:
    """Per-stage ring buffers + per-scan accumulators + counters."""

    def __init__(
        self,
        profile_dir: str | Path,
        *,
        slow_scan_ms: float = 5.0,
        ring: int = _RING_DEFAULT,
    ) -> None:
        self.dir = Path(profile_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.slow_scan_ns: float = float(slow_scan_ms) * 1e6
        self._ring = int(ring)
        self._buf: dict[str, np.ndarray] = {}
        self._idx: dict[str, int] = {}       # next write position (wraps)
        self._count: dict[str, int] = {}      # total samples ever recorded
        self._scan_acc: dict[str, int] = {}   # ns accumulated in the current scan
        self._slow: deque[dict[str, Any]] = deque(maxlen=10_000)
        self._gc_at_scan_start: int = 0
        # Cumulative event counters and missed-opportunity reason histogram,
        # pre-seeded to 0 so their Prometheus series exist before the first event.
        self._counters: Counter[str] = Counter({k: 0 for k in _PRESEED_COUNTERS})
        self._missed_reasons: Counter[str] = Counter({k: 0 for k in KNOWN_MISS_REASONS})
        self._stages_csv = self.dir / "stages.csv"
        self._slow_path = self.dir / "slow_scans.jsonl"
        self._counters_path = self.dir / "counters.json"
        self._started_wall = time.time()
        # Plain-dict snapshot rebuilt off the hot path by dump() so the Prometheus
        # /metrics scrape can serialize it with zero numpy work on the event loop.
        # A single reference swap → readers never see a half-built snapshot.
        self._metrics_snapshot: dict[str, Any] | None = None
        for stage in SCAN_STAGES:
            self._ensure(stage, accumulate=True)
        for series in PIPELINE_SERIES:
            self._ensure(series, accumulate=False)

    # ── hot path ──────────────────────────────────────────────────────────────
    def _ensure(self, stage: str, *, accumulate: bool = True) -> np.ndarray:
        b = self._buf.get(stage)
        if b is None:
            b = np.empty(self._ring, dtype=np.float64)  # nanoseconds
            self._buf[stage] = b
            self._idx[stage] = 0
            self._count[stage] = 0
            if accumulate:
                self._scan_acc[stage] = 0
        return b

    def begin_scan(self) -> None:
        """Reset per-scan accumulators and snapshot the GC counter."""
        acc = self._scan_acc
        for k in acc:
            acc[k] = 0
        self._gc_at_scan_start = _GC_EVENTS

    def record(self, stage: str, dt_ns: int) -> None:
        """Append one scan-stage sample (ns) and add it to the scan accumulator."""
        b = self._buf.get(stage)
        if b is None:
            b = self._ensure(stage, accumulate=True)
        i = self._idx[stage]
        b[i] = dt_ns
        i += 1
        self._idx[stage] = 0 if i >= self._ring else i
        self._count[stage] += 1
        # Only stages tracked in the per-scan accumulator participate in the
        # slow-scan breakdown; pipeline series (registered without accumulate)
        # are skipped here so begin_scan() never has to reset them.
        if stage in self._scan_acc:
            self._scan_acc[stage] += dt_ns

    def record_series(self, series: str, dt_ns: float) -> None:
        """Append one pipeline-series sample (ns) — no per-scan accumulation.

        Safe to call from WS I/O threads and the GC callback; see module docstring.
        """
        b = self._buf.get(series)
        if b is None:
            b = self._ensure(series, accumulate=False)
        i = self._idx[series]
        b[i] = dt_ns
        i += 1
        self._idx[series] = 0 if i >= self._ring else i
        self._count[series] += 1

    def incr(self, name: str, n: int = 1) -> None:
        """Increment a cumulative event counter (e.g. scans, wakeups, entries)."""
        self._counters[name] += n

    def incr_missed(self, reason: str, n: int = 1) -> None:
        """Increment the missed-opportunity reason histogram."""
        self._missed_reasons[reason] += n

    def scan_ms(self, stage: str) -> float:
        """Accumulated milliseconds for ``stage`` in the current scan."""
        return self._scan_acc.get(stage, 0) / 1e6

    def gc_active(self) -> bool:
        return _GC_EVENTS > self._gc_at_scan_start

    def maybe_record_slow_scan(self, total_ns: int, context: dict[str, Any]) -> None:
        """Buffer a full per-stage breakdown when the scan exceeds the threshold."""
        if total_ns < self.slow_scan_ns:
            return
        rec: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "total_scan_ms": round(total_ns / 1e6, 4),
        }
        for stage, ns in self._scan_acc.items():
            rec[stage + "_ms"] = round(ns / 1e6, 4)
        rec["gc_active"] = self.gc_active()
        rec.update(context)
        self._slow.append(rec)

    # ── off-hot-path: percentiles, dump, summary ────────────────────────────────
    def _percentiles(self, stage: str) -> dict[str, Any] | None:
        n = self._count.get(stage, 0)
        if n == 0:
            return None
        buf = self._buf[stage]
        data = buf if n >= self._ring else buf[:n]
        p = np.percentile(data, [50, 90, 99, 99.9, 100])
        return {
            "count": int(n),
            "mean_ms": round(float(np.mean(data)) / 1e6, 5),
            "p50_ms": round(float(p[0]) / 1e6, 5),
            "p90_ms": round(float(p[1]) / 1e6, 5),
            "p99_ms": round(float(p[2]) / 1e6, 5),
            "p999_ms": round(float(p[3]) / 1e6, 5),
            "max_ms": round(float(p[4]) / 1e6, 5),
        }

    def _histogram(self, stage: str) -> dict[str, int] | None:
        """Cumulative ≤ bucket counts (ms edges) for a Grafana latency heatmap.

        Off the hot path (called from dump()).  One O(n) pass via np.histogram,
        then cumsum — cheap even over the full ring.
        """
        n = self._count.get(stage, 0)
        if n == 0:
            return None
        buf = self._buf[stage]
        data_ms = (buf if n >= self._ring else buf[:n]) / 1e6  # ns → ms (copy)
        edges = np.array((0.0,) + _HIST_EDGES_MS + (np.inf,))
        counts, _ = np.histogram(data_ms, bins=edges)
        cum = np.cumsum(counts)
        out = {str(_HIST_EDGES_MS[i]): int(cum[i]) for i in range(len(_HIST_EDGES_MS))}
        out["+Inf"] = int(cum[-1])
        return out

    def _build_snapshot(self) -> dict[str, Any]:
        """Materialize all percentiles + counters into a plain dict (off hot path)."""
        latency: dict[str, dict[str, Any]] = {}
        for stage in SCAN_STAGES + PIPELINE_SERIES + tuple(
            s for s in self._buf if s not in SCAN_STAGES and s not in PIPELINE_SERIES
        ):
            st = self._percentiles(stage)
            if st:
                if stage in HEATMAP_SERIES:
                    hist = self._histogram(stage)
                    if hist:
                        st = {**st, "buckets": hist}
                latency[stage] = st
        return {
            "ts": round(time.time(), 3),
            "elapsed_s": round(time.time() - self._started_wall, 1),
            "latency": latency,
            "counters": dict(self._counters),
            "missed_reasons": dict(self._missed_reasons),
        }

    def snapshot(self) -> dict[str, Any] | None:
        """Return the most recent cached metrics snapshot (or None before first dump).

        Cheap: returns the cached dict reference.  Safe to call from the event
        loop (e.g. the /metrics scrape handler) — no percentile computation here.
        """
        return self._metrics_snapshot

    def _dump_counters(self) -> None:
        """Overwrite counters.json with the current cumulative snapshot."""
        if not self._counters and not self._missed_reasons:
            return
        snapshot = {
            "ts": round(time.time(), 3),
            "elapsed_s": round(time.time() - self._started_wall, 1),
            "counters": dict(self._counters),
            "missed_reasons": dict(self._missed_reasons),
        }
        tmp = self._counters_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(snapshot, indent=2))
            tmp.replace(self._counters_path)  # atomic
        except OSError:
            pass

    def dump(self) -> None:
        """Append a percentile snapshot per stage to stages.csv, flush slow scans,
        and overwrite the cumulative counters snapshot.

        Safe to call from a worker thread (see module docstring).
        """
        now = round(time.time(), 3)
        rows: list[dict[str, Any]] = []
        for stage in SCAN_STAGES + tuple(s for s in self._buf if s not in SCAN_STAGES):
            stats = self._percentiles(stage)
            if stats:
                rows.append({"ts": now, "stage": stage, **stats})
        if rows:
            write_header = not self._stages_csv.exists()
            with self._stages_csv.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
        if self._slow:
            with self._slow_path.open("a") as f:
                while self._slow:
                    f.write(json.dumps(self._slow.popleft()) + "\n")
        self._dump_counters()
        # Refresh the cached snapshot for the /metrics scrape (single ref swap).
        self._metrics_snapshot = self._build_snapshot()

    def write_summary(self, quiet: bool = False) -> None:
        """Persist final per-stage + pipeline tables and counters to summary.txt.

        Prints them too unless ``quiet`` (headless mode keeps the console silent).
        """
        self.dump()  # flush any residual slow-scan/counter records first
        elapsed = time.time() - self._started_wall
        lines = [
            f"profiling summary — {self._stages_csv.parent} (elapsed {elapsed:.0f}s)",
            "",
            "── Scan stages (compute inside _scan_once) ──",
            f"{'stage':<22}{'count':>10}{'mean':>11}{'p50':>11}{'p90':>11}{'p99':>11}{'p99.9':>11}{'max':>11}",
        ]
        for stage in SCAN_STAGES:
            st = self._percentiles(stage)
            if not st:
                continue
            lines.append(self._fmt_row(stage, st))

        pipeline_rows = [(s, self._percentiles(s)) for s in PIPELINE_SERIES]
        pipeline_rows = [(s, st) for s, st in pipeline_rows if st]
        if pipeline_rows:
            lines += [
                "",
                "── Pipeline latency (tick-to-trade ends, all ms) ──",
                f"{'series':<22}{'count':>10}{'mean':>11}{'p50':>11}{'p90':>11}{'p99':>11}{'p99.9':>11}{'max':>11}",
            ]
            for series, st in pipeline_rows:
                lines.append(self._fmt_row(series, st))

        if self._counters:
            lines += ["", "── Event counters ──"]
            for name, cnt in sorted(self._counters.items()):
                lines.append(f"{name:<30}{cnt:>12}")

        nonzero_reasons = {r: c for r, c in self._missed_reasons.items() if c}
        if nonzero_reasons:
            total = sum(nonzero_reasons.values())
            lines += ["", f"── Missed-opportunity reasons (total {total}) ──"]
            for reason, cnt in sorted(nonzero_reasons.items(), key=lambda kv: -kv[1]):
                pct = 100 * cnt / total if total else 0
                lines.append(f"{reason:<30}{cnt:>10}  ({pct:>4.0f}%)")

        text = "\n".join(lines)
        try:
            (self.dir / "summary.txt").write_text(text + "\n")
        except OSError:
            pass
        if not quiet:
            print("\n" + text, flush=True)

    @staticmethod
    def _fmt_row(name: str, st: dict[str, Any]) -> str:
        return (
            f"{name:<22}{st['count']:>10}{st['mean_ms']:>11.3f}{st['p50_ms']:>11.3f}"
            f"{st['p90_ms']:>11.3f}{st['p99_ms']:>11.3f}{st['p999_ms']:>11.3f}{st['max_ms']:>11.3f}"
        )


async def loop_lag_monitor(interval_s: float = 0.005) -> None:
    """Background task: measure event-loop scheduling lag.

    Sleeps ``interval_s`` then records how much *longer* than requested the wakeup
    actually took.  A large overshoot means the loop could not service the timer
    on time — it was blocked in a CPU-bound call, paused for GC, or preempted by
    the OS.  That same starvation delays the scan loop and the WS-queue drain, so
    ``loop_lag`` percentiles directly explain the "total high but stage-sum low"
    (``__unaccounted__``) slow scans.  Runs at ~1/interval Hz; negligible cost.

    Self-terminating: returns when profiling is torn down.  Start it as a task
    and cancel on shutdown (the cancel is the normal stop path).
    """
    interval_ns = int(interval_s * 1e9)
    while ENABLED:
        t0 = time.perf_counter_ns()
        await asyncio.sleep(interval_s)
        lag = time.perf_counter_ns() - t0 - interval_ns
        rec = RECORDER
        if rec is not None:
            rec.record_series("loop_lag", lag if lag > 0 else 0)


def setup(profile_dir: str | Path, *, slow_scan_ms: float = 5.0) -> LatencyRecorder:
    """Install the GC hook, create the recorder, and arm the ENABLED flag."""
    global RECORDER, ENABLED
    _install_gc_hook()
    RECORDER = LatencyRecorder(profile_dir, slow_scan_ms=slow_scan_ms)
    ENABLED = True
    return RECORDER


def teardown() -> None:
    """Disarm profiling.  The recorder object is kept so a final summary can run."""
    global ENABLED
    ENABLED = False
