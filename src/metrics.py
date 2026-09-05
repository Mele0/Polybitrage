"""Zero-dependency Prometheus exposition renderer for the arb bot.

Turns the bot's live state + the profiler's cached snapshot into Prometheus
text-format metrics, served at ``GET /metrics`` by ``health_server.HealthServer``.

Design / overhead
-----------------
* **No third-party dependency** — emits the text exposition format directly.
* **Pull model, near-zero hot-path cost** — nothing runs between scrapes.  On a
  scrape we read:
    - the latest ``ScanRow`` (already computed each scan) for portfolio /
      opportunity / WS-health vitals, and
    - ``profiling.RECORDER.snapshot()`` — a plain dict the background dumper
      already built off the event loop — for latency percentiles + counters.
  So the scrape handler does only cheap dict/attr reads and string building; no
  numpy, no locks, no per-scan work added anywhere.

Everything Grafana needs comes from here: latency (with peaks via p99/max),
throughput, opportunities, fill rate, miss reasons, PnL, positions, WS health.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src import market_profiler, net_profiler, profiling, system_profiler

if TYPE_CHECKING:
    from src.simulator import PaperArbSimulator

# Latency snapshot key (in profiling) → Prometheus quantile label.
_QUANTILES: tuple[tuple[str, str], ...] = (
    ("p50_ms", "0.5"),
    ("p90_ms", "0.9"),
    ("p99_ms", "0.99"),
    ("p999_ms", "0.999"),
    ("max_ms", "1.0"),
)


class _Buf:
    """Accumulates metric families, emitting HELP/TYPE once per metric name."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._declared: set[str] = set()

    def metric(
        self,
        name: str,
        value: float | int | bool | None,
        help_text: str,
        mtype: str = "gauge",
        labels: dict[str, str] | None = None,
    ) -> None:
        if value is None:
            return
        if name not in self._declared:
            self._lines.append(f"# HELP {name} {help_text}")
            self._lines.append(f"# TYPE {name} {mtype}")
            self._declared.add(name)
        if labels:
            lbl = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
            self._lines.append(f"{name}{{{lbl}}} {_num(value)}")
        else:
            self._lines.append(f"{name} {_num(value)}")

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"


def _num(v: float | int | bool) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if v != v:  # NaN
        return "NaN"
    return repr(float(v))


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render_prometheus(sim: "PaperArbSimulator", started_at: float) -> str:
    """Render all bot metrics in Prometheus text-exposition format."""
    b = _Buf()
    b.metric("arb_up", 1, "1 while the bot process is serving metrics.")
    b.metric("arb_uptime_seconds", round(time.time() - started_at, 1), "Seconds since the metrics server started.")
    b.metric("arb_scans_total", int(getattr(sim, "_total_scans", 0)), "Total scan cycles executed.", "counter")

    _render_vitals(b, sim)
    _render_ws_deep(b, sim)
    _render_profiler(b)
    _render_system(b)
    _render_network(b)
    _render_market(b)
    return b.text()


def _render_vitals(b: _Buf, sim: "PaperArbSimulator") -> None:
    """Portfolio / opportunity / WS-health vitals from the latest ScanRow."""
    rows = getattr(sim, "scan_rows", None)
    row = rows[-1] if rows else None  # deque[-1] is O(1) and slice-free
    if row is None:
        return

    # ── Portfolio ──
    b.metric("arb_cash_usd", _f(row.cash_available), "Available (unlocked) cash, USD.")
    b.metric("arb_locked_capital_usd", _f(row.locked_capital), "Capital locked in open positions, USD.")
    b.metric("arb_open_positions", _i(row.open_positions_count), "Open positions count.")
    b.metric("arb_realized_pnl_usd", _f(row.realized_pnl), "Realized PnL of closed positions, USD.")
    b.metric("arb_unrealized_pnl_usd", _f(row.unrealized_pnl), "Mark-to-liquidation PnL of open positions, USD.")
    b.metric("arb_guaranteed_profit_usd", _f(row.guaranteed_profit_if_held), "Worst-case guaranteed profit if all open positions held to resolution, USD.")
    b.metric("arb_best_case_profit_usd", _f(row.best_case_profit_if_held), "Best-case profit if all open positions held to resolution, USD.")

    # ── Opportunities (current scan) ──
    b.metric("arb_executable_candidates", _i(row.executable_candidates_count), "Pairs below the entry threshold this scan.")
    b.metric("arb_near_candidates", _i(row.near_arb_candidates_count), "Pairs within the near-arb threshold this scan.")
    b.metric("arb_rejected_candidates", _i(row.rejected_count), "Pairs rejected (missing book / ask / spread) this scan.")
    b.metric("arb_best_net_cost", _f(row.net_total_cost), "Best (lowest) net total cost seen this scan; <1.0 ⇒ arbitrage.")
    b.metric("arb_distance_to_entry", _f(row.distance_to_entry), "Gap between best net cost and the entry threshold (≤0 ⇒ executable).")
    b.metric("arb_best_optimal_size", _f(row.best_optimal_size), "Optimal size of the best opportunity this scan.")
    b.metric("arb_best_n_leg_gross_edge", _f(row.best_n_leg_gross_edge), "Gross edge of the best N-leg opportunity this scan.")

    # ── Scan / book health (live, available even without --profile) ──
    b.metric("arb_scan_time_ms", _f(row.scan_time_ms), "Wall time of the latest scan, ms.")
    b.metric("arb_scan_tokens_total", _i(row.unique_tokens), "Unique tokens in the scan universe.")
    b.metric("arb_scan_tokens_fetched", _i(row.unique_tokens_fetched), "Tokens re-fetched this scan (dirty-set size).")
    b.metric("arb_max_book_age_ms", _f(row.max_book_age_ms), "Staleness of the oldest book touched this scan, ms.")
    b.metric("arb_wire_latency_ms", _f(row.update_latency_ms), "Exchange-event → local-receipt latency this scan, ms.")
    b.metric("arb_books_missing", _i(row.books_missing_count), "Tokens with no order book this scan.")
    b.metric("arb_asks_missing", _i(row.asks_missing_count), "Tokens missing the ask side this scan.")
    b.metric("arb_cache_hits_total", _i(row.cache_hits), "Cumulative REST cache hits.", "counter")
    b.metric("arb_failed_books_total", _i(row.failed_book_count), "Books that failed to fetch this scan.")

    # ── WebSocket health ──
    b.metric("arb_ws_connected", bool(row.websocket_connected), "1 if at least one WS slot is connected.")
    b.metric("arb_ws_reconnects_total", _i(row.websocket_reconnect_count), "Cumulative WS reconnects.", "counter")
    b.metric("arb_ws_token_updates_total", _i(row.token_update_count), "Cumulative per-token WS updates received.", "counter")
    b.metric("arb_ws_event_recomputes_total", _i(row.event_triggered_recomputes), "Cumulative event-triggered scan wakeups.", "counter")
    b.metric("arb_fallback_to_polling", bool(row.fallback_to_polling_used), "1 if REST polling fallback was used this scan.")


def _render_profiler(b: _Buf) -> None:
    """Latency percentiles + profiler counters + miss histogram (when --profile)."""
    rec = profiling.RECORDER
    if rec is None:
        return
    snap = rec.snapshot()
    if not snap:
        return

    latency = snap.get("latency") or {}
    for series, st in latency.items():
        for key, q in _QUANTILES:
            b.metric(
                "arb_latency_ms", st.get(key),
                "Pipeline/scan latency percentiles, ms (series=stage, quantile).",
                labels={"series": series, "quantile": q},
            )
        b.metric("arb_latency_mean_ms", st.get("mean_ms"), "Mean latency per series, ms.", labels={"series": series})
        b.metric("arb_latency_samples", st.get("count"), "Samples in the current ring window per series.", labels={"series": series})
        # Cumulative histogram buckets (curated series) → Grafana latency heatmaps.
        for le, cnt in (st.get("buckets") or {}).items():
            b.metric("arb_latency_bucket", int(cnt),
                     "Cumulative latency histogram (ms upper bound `le`) for heatmaps.",
                     labels={"series": series, "le": le})

    counters = snap.get("counters") or {}
    # Map known cumulative counters to *_total (Prometheus counter) names.
    _counter_map = {
        "scans": "arb_profiler_scans_total",
        "ws_event_wakeups": "arb_ws_event_wakeups_total",
        "empty_wakeups": "arb_empty_wakeups_total",
        "entries_filled": "arb_entries_filled_total",
        "entries_missed": "arb_entries_missed_total",
        "near_arb_appearances": "arb_near_arb_appearances_total",
    }
    for src_name, prom_name in _counter_map.items():
        if src_name in counters:
            b.metric(prom_name, int(counters[src_name]), f"Cumulative {src_name}.", "counter")

    for reason, cnt in (snap.get("missed_reasons") or {}).items():
        b.metric(
            "arb_missed_total", int(cnt),
            "Executable opportunities that did not convert, by reason.",
            "counter", labels={"reason": reason},
        )


def _render_ws_deep(b: _Buf, sim: "PaperArbSimulator") -> None:
    """Deep WebSocket telemetry from the Rust backend (per-type msgs, bytes, feed)."""
    ws = getattr(getattr(sim, "_provider", None), "ws_client", None)
    fn = getattr(ws, "ws_stats", None)
    if fn is None:
        return
    try:
        s = fn()
    except Exception:  # noqa: BLE001
        return
    b.metric("arb_ws_book_msgs_total", _i(s.get("book_msgs")), "Cumulative 'book' snapshot messages.", "counter")
    b.metric("arb_ws_price_change_msgs_total", _i(s.get("price_change_msgs")), "Cumulative 'price_change' messages.", "counter")
    b.metric("arb_ws_bba_msgs_total", _i(s.get("bba_msgs")), "Cumulative 'best_bid_ask' messages.", "counter")
    b.metric("arb_ws_frames_total", _i(s.get("frames_recv")), "Cumulative WS frames received.", "counter")
    b.metric("arb_ws_bytes_total", _i(s.get("bytes_recv")), "Cumulative WS bytes received.", "counter")
    b.metric("arb_ws_parse_errors_total", _i(s.get("parse_errors")), "Cumulative WS parse errors (data quality).", "counter")
    b.metric("arb_ws_connected_slots", _i(s.get("connected_slots")), "Connected WS slots (0-2).")
    b.metric("arb_ws_feed_staleness_ms", _f(s.get("ms_since_last_message")), "Ms since the last frame on any slot (feed gap, not per-token).")
    b.metric("arb_ws_last_drain_size", _i(s.get("last_drain_size")), "Tokens in the most recent queue drain (tick backpressure depth).")


def _render_system(b: _Buf) -> None:
    """OS / process resource metrics (CPU, memory, ctx-switches, GC, threads)."""
    s = system_profiler.snapshot()
    if not s:
        return
    g = s.get
    b.metric("arb_proc_cpu_percent", g("cpu_percent"), "Process CPU% (all threads incl. Rust), from getrusage diff.")
    b.metric("arb_sys_cpu_percent", g("sys_cpu_percent"), "System-wide CPU%.")
    b.metric("arb_proc_rss_bytes", g("rss_bytes"), "Process resident set size, bytes.")
    b.metric("arb_proc_maxrss_bytes", g("maxrss_bytes"), "Peak resident set size, bytes.")
    b.metric("arb_sys_mem_percent", g("sys_mem_percent"), "System memory used, percent.")
    b.metric("arb_proc_threads", g("threads_os") if "threads_os" in s else g("threads_python"), "OS thread count (incl. Rust tokio + WS threads).")
    b.metric("arb_proc_open_fds", g("open_fds"), "Open file descriptors.")
    b.metric("arb_ctx_switch_involuntary_per_s", g("ctx_switch_involuntary_per_s"), "Involuntary ctx switches/s — OS preemption; the prime latency-jitter signal.")
    b.metric("arb_ctx_switch_voluntary_per_s", g("ctx_switch_voluntary_per_s"), "Voluntary ctx switches/s — blocking waits (IO/locks).")
    b.metric("arb_ctx_switch_involuntary_total", g("ctx_switch_involuntary_total"), "Cumulative involuntary ctx switches.", "counter")
    b.metric("arb_page_faults_per_s", g("page_faults_per_s"), "Page faults/s.")
    b.metric("arb_page_faults_major_total", g("page_faults_major_total"), "Cumulative major (disk-backed) page faults.", "counter")
    b.metric("arb_load_avg_1m", g("load_avg_1m"), "1-minute system load average.")
    for gen in (0, 1, 2):
        b.metric("arb_gc_collections_total", g(f"gc_gen{gen}_collections_total"),
                 "Cumulative GC collections by generation.", "counter", labels={"generation": str(gen)})


def _render_network(b: _Buf) -> None:
    """Active network RTT + clock skew (background probes)."""
    n = net_profiler.snapshot()
    if not n:
        return
    b.metric("arb_net_clock_skew_ms", n.get("clock_skew_ms"),
             "Local minus exchange clock, ms — disambiguates wire_latency (network vs clock).")
    for host in ("clob", "gamma"):
        b.metric("arb_net_tcp_connect_ms", n.get(f"{host}_tcp_connect_ms"), "TCP connect RTT, ms.", labels={"host": host})
        b.metric("arb_net_https_get_ms", n.get(f"{host}_https_get_ms"), "HTTPS GET RTT, ms.", labels={"host": host})
        b.metric("arb_net_reachable", n.get(f"{host}_reachable"), "1 if endpoint reachable on last probe.", labels={"host": host})


def _render_market(b: _Buf) -> None:
    """Order-book microstructure aggregates across recently-updated tokens."""
    m = market_profiler.snapshot()
    if not m:
        return
    g = m.get
    b.metric("arb_book_spread_median", g("spread_median"), "Median bid-ask spread across recently-updated tokens.")
    b.metric("arb_book_spread_p90", g("spread_p90"), "p90 spread.")
    b.metric("arb_book_spread_p99", g("spread_p99"), "p99 spread (the widest, sniping-prone books).")
    b.metric("arb_book_spread_max", g("spread_max"), "Max spread observed in the window.")
    b.metric("arb_book_depth_median", g("depth_median"), "Median two-sided depth, shares.")
    b.metric("arb_book_depth_total", g("depth_total"), "Total observed depth in the window, shares.")
    b.metric("arb_book_imbalance_median", g("imbalance_median"), "Median top-of-book imbalance (+1 all-bid … -1 all-ask).")
    b.metric("arb_book_crossed", g("crossed"), "Crossed books (bid>ask) in the window — data error or true arb.")
    b.metric("arb_book_locked", g("locked"), "Locked books (bid==ask) in the window.")
    b.metric("arb_book_one_sided", g("one_sided"), "One-sided books in the window.")
    b.metric("arb_book_two_sided", g("two_sided"), "Two-sided books observed in the window.")
    b.metric("arb_book_two_sided_frac", g("two_sided_frac"), "Fraction of observed books that were two-sided.")
    b.metric("arb_book_observed", g("observed"), "Books observed in the sampling window.")


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
