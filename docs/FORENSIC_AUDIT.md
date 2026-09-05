# Competitive Forensic Audit

Forensic audit mode records why the bot did or did not capture each opportunity. It is designed for competitive diagnosis: latency, stale books, rate limits, capital locks, duplicate/cooldown guards, and contested market bursts.

## Install

```bash
python -m pip install -e '.[audit]'
```

The normal scanner does not require `pyarrow`. The extra is only needed for `--audit-mode forensic` and `audit-report`.

## Run With Audit

```bash
poly-arb paper-arb-sim \
  --scan-all-candidates \
  --candidate-pairs config/generated_crypto_may7.yaml \
  --book-source websocket \
  --fallback-to-polling true \
  --entry-rest-recheck true \
  --enable-n-leg-trading \
  --audit-mode forensic \
  --audit-dir reports/audit \
  --audit-raw-ws true
```

Each run writes to `reports/audit/<run_id>/`. Tables are partitioned Parquet datasets named like `decisions.parquet/`, `ws_events.parquet/`, and `missed_fills.parquet/` so partial runs remain readable after a crash.

## Report

```bash
poly-arb audit-report reports/audit/<run_id> \
  --top-missed 25 \
  --format html,markdown,json
```

Compare two runs:

```bash
poly-arb audit-report reports/audit/new-run \
  --compare reports/audit/old-run \
  --format html
```

Outputs:

- `report/index.html`: visual forensic report with funnel charts, latency metrics, ambient activity, and missed-opportunity timelines.
- `report/summary.md`: static summary.
- `report/summary.json`: machine-readable metrics and comparison deltas.

## Core Tables

- `decisions.parquet`: one row per moment the bot could act.
- `ws_events.parquet`: raw WebSocket event ledger with exchange, receipt, and processing timestamps.
- `network.parquet`: REST/WebSocket reconnect latency, status codes, 429/425/5xx flags, and queue/semaphore waits.
- `book_snapshots.parquet` and `book_levels.parquet`: full observed depth per scan.
- `market_activity.parquet`: ambient update/trade cadence and contested-market score.
- `pair_observations.parquet` and `n_leg_candidates.parquet`: all candidate checks and computed costs.
- `opportunity_windows.parquet`: reconstructed profitable windows across scans.
- `portfolio_snapshots.parquet`: cash, locked capital, positions, and exposure at decision time.
- `orders.parquet`: live order attempts and acknowledgements.
- `missed_fills.parquet`: classified misses such as REST recheck loss, depth depletion, capital block, or duplicate/cooldown guard.
- `timeline_events.parquet`: chronological events for plotting missed opportunities.

