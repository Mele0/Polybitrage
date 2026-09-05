# Monitoring — Prometheus + Grafana

Live, low-overhead observability for the arb bot. The bot exposes Prometheus
metrics at **`/metrics`** on its health port; Prometheus scrapes them and Grafana
visualizes everything — **latency (with peaks), scans, opportunities, fills,
misses-by-reason, PnL, positions, and WebSocket health** — on one dashboard.

## Why this is (near) zero-overhead

- **Pull model.** Nothing runs between scrapes. The hot path only maintains the
  lock-free numpy ring buffers / counters it already kept for the profiler.
- **No work on the event loop.** Latency percentiles are computed off-loop by the
  background dumper and cached; the `/metrics` handler just serializes a dict +
  reads the latest `ScanRow`. No numpy, no locks per scrape.
- **No new Python dependency** — the bot emits the exposition format itself.

## 1. Run the bot headless with metrics

```bash
PY=.venv/bin/python   # the project virtualenv (see ../README for setup)

$PY -m src.main paper-arb-sim \
  --candidate-pairs config/generated_crypto_june-13.yaml --scan-all-candidates \
  --relation-safety all --book-source rust-websocket \
  --headless \
  --profile --profile-dump-seconds 5 \
  --health-host 0.0.0.0 --health-port 8765
```

- `--headless` → no console output at all; the bot only trades + serves telemetry.
- `--profile` → populates the latency series (loop_lag, wire, queue_wait, …).
- `--profile-dump-seconds 5` → refresh the latency snapshot every 5s (matches the
  Prometheus scrape interval).
- `--health-host 0.0.0.0` → **required** so Prometheus (in Docker) can reach the
  bot on the host. It binds `127.0.0.1` by default. Only do this on a trusted/
  firewalled network.

Sanity check: `curl -s localhost:8765/metrics | head`.

## 2. Start Prometheus + Grafana

```bash
cd monitoring
docker compose up -d
```

- **Grafana** → http://localhost:3000 — the *Polymarket Arb Bot — Latency &
  Trading* dashboard is auto-provisioned (anonymous admin access, no login).
- **Prometheus** → http://localhost:9090 — check **Status → Targets**: `arb-bot`
  should be **UP**. If it's DOWN, the bot isn't running with `--health-host 0.0.0.0`,
  or the port differs (edit `prometheus.yml`).

Stop with `docker compose down` (add `-v` to wipe stored metrics).

## What's on the dashboard (60 panels, 9 rows)

| Row | Panels |
|-----|--------|
| ⏱ Tick-to-trade latency | pipeline p50 & p99 (wire / queue_wait / wakeup / total_scan / order_rtt / signing / rest_call), `loop_lag` (starvation, thresholds 5/20 ms) |
| 🔥 Latency heatmaps | distribution-over-time heatmaps for wire_latency, total_scan, queue_wait, loop_lag, rest_call_latency, order_rtt |
| 🧩 Scanner module breakdown | per-stage s1–s6 compute, p50 (stacked) & p99 |
| 🚀 Throughput & WebSocket | scan rate, **WS msgs/s by type** (book/price_change/bba), **bytes/s**, dirty-set size, **tick backpressure** (drain size), event vs empty wakeups, connected slots, **feed staleness**, parse errors, reconnects |
| 💰 Opportunities, fills & alpha decay | executable/near/rejected, best net cost & distance, **opportunity lifetime** (how long arbs live), fill rate, near-arb appearances/min, **misses by reason** (donut), filled vs missed |
| 📖 Order-book microstructure | spread median/p90/p99/max, depth & imbalance, **crossed/locked/one-sided** book quality, two-sided fraction |
| 🖥 OS / System | process & system CPU%, **involuntary ctx-switches/s** (preemption jitter, thresholds 500/2000), RSS, threads, FDs, load, page faults/s, GC collections/s by gen |
| 🌐 Network | TCP-connect & HTTPS-GET RTT (clob/gamma), **clock skew vs exchange** |
| 📈 Portfolio & health | realized/unrealized/guaranteed PnL, cash, open positions, locked capital, uptime |

### Reading the key bottleneck signals
- **`loop_lag` p99 high** → event loop starved (blocking call / GC / preemption). Cross-check `involuntary ctx-switches/s` and `gc collections/s`.
- **`tick backpressure` (drain size) rising + `queue_wait` p99 rising** → can't keep up with the WS firehose; score fewer pairs or speed up the scan.
- **`opportunity_lifetime` ≲ tick-to-trade sum** → arbs die faster than you react; you're structurally locked out (co-locate / cut latency).
- **`wire_latency` high but `clock_skew` small** → genuine network/feed delay, not a clock issue (co-locate). High skew → fix NTP.
- **`two-sided fraction` low / `crossed` > 0** → book data-quality problems (adverse-selection risk).

## Tuning resolution vs. overhead

- Latency-snapshot freshness = `--profile-dump-seconds` (default 10; set 5 to match
  the 5s scrape). Lower = fresher, still off the hot path.
- Live vitals (cash/positions/opportunities/scan-time) are fresh on **every** scrape.
- For finer peak capture, lower `scrape_interval` in `prometheus.yml` (and the dump
  interval to match).

## Metric reference (prefix `arb_`, ~80 families)

- **Latency**: `arb_latency_ms{series,quantile}`, `arb_latency_mean_ms{series}`,
  `arb_latency_samples{series}`, `arb_latency_bucket{series,le}` (heatmaps). Series:
  wire_latency, queue_wait, wakeup_latency, loop_lag, scan_interval,
  book_age_at_decision, signing_latency, rest_call_latency, order_rtt, gc_pause,
  opportunity_lifetime, s1_dirtyset…s6_try_enter, total_scan.
- **Throughput/WS**: `arb_scans_total`, `arb_scan_tokens_{total,fetched}`,
  `arb_ws_{book,price_change,bba}_msgs_total`, `arb_ws_frames_total`,
  `arb_ws_bytes_total`, `arb_ws_parse_errors_total`, `arb_ws_connected_slots`,
  `arb_ws_feed_staleness_ms`, `arb_ws_last_drain_size`, `arb_ws_token_updates_total`,
  `arb_ws_event_wakeups_total`, `arb_empty_wakeups_total`, `arb_ws_reconnects_total`.
- **Opportunities/alpha**: `arb_executable_candidates`, `arb_near_candidates`,
  `arb_rejected_candidates`, `arb_best_net_cost`, `arb_distance_to_entry`,
  `arb_near_arb_appearances_total`, `arb_entries_{filled,missed}_total`,
  `arb_missed_total{reason}`.
- **Microstructure**: `arb_book_spread_{median,p90,p99,max}`,
  `arb_book_depth_{median,total}`, `arb_book_imbalance_median`,
  `arb_book_{crossed,locked,one_sided,two_sided,observed}`, `arb_book_two_sided_frac`.
- **OS/system**: `arb_proc_cpu_percent`, `arb_sys_cpu_percent`, `arb_proc_rss_bytes`,
  `arb_proc_{threads,open_fds}`, `arb_ctx_switch_{involuntary,voluntary}_per_s`,
  `arb_ctx_switch_involuntary_total`, `arb_page_faults_{per_s,major_total}`,
  `arb_load_avg_1m`, `arb_gc_collections_total{generation}`.
- **Network**: `arb_net_tcp_connect_ms{host}`, `arb_net_https_get_ms{host}`,
  `arb_net_reachable{host}`, `arb_net_clock_skew_ms`.
- **Portfolio/health**: `arb_cash_usd`, `arb_locked_capital_usd`, `arb_open_positions`,
  `arb_realized_pnl_usd`, `arb_unrealized_pnl_usd`, `arb_guaranteed_profit_usd`,
  `arb_ws_connected`, `arb_max_book_age_ms`, `arb_scan_time_ms`, `arb_uptime_seconds`.

All distributions update at `--profile-dump-seconds` cadence; live vitals every scrape.
OS sampling is 1 Hz; network probes every `--profile-net-probe-seconds` (default 15s,
set 0 to disable outbound probes).
