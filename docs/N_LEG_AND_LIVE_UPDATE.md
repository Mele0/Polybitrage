# N-leg and Live Trading Update

This update turns the previous 3-leg range/threshold extension into a generic N-leg scanner and adds a guarded live trading path.

## What Changed

- Added generic N-leg range/threshold package construction.
- Preserved the original 3-leg behavior with `--n-leg-max-ranges 1`.
- Added wider contiguous range packages with `--n-leg-max-ranges 2`, or all windows with `--n-leg-max-ranges 0`.
- Cached static scan plans so pair/token/spec construction is not repeated every scan.
- Optimized N-leg scoring so expensive depth sizing only runs for candidates that can still beat the current best candidate.
- Batched REST order book fetches through `/books` chunks.
- Reduced loop sleep floors so WebSocket scans can react faster.
- Added a Rich live dashboard mode so the terminal updates one panel in place instead of printing endless panels.
- Added guarded live mode using the Polymarket CLOB SDK.
- Added live environment validation, explicit confirmation phrases, session spend caps, leg caps, REST rechecks, FOK order posting, and CSV logging for live order attempts.
- Fixed the WebSocket N-leg sizing issue: WebSocket top-of-book updates can be faster than polling but may not carry reliable depth. N-leg entries now REST recheck and depth-size before entering, and they skip instead of spending max capital when depth is untrusted.

## N-leg Logic

For `k` contiguous range markets, the scanner builds:

```text
low threshold YES + high threshold NO + each covered range NO
```

The number of legs is `k + 2`, and the guaranteed payout is `k + 1`.

Examples:

- One range: `low YES + high NO + range NO`, 3 legs, guaranteed payout `2`.
- Two ranges: `low YES + high NO + range_1 NO + range_2 NO`, 4 legs, guaranteed payout `3`.
- More ranges follow the same pattern.

The recommended live-monitoring setting is `--n-leg-max-ranges 1` or `2`. Use `0` for broader research scans because it scans every contiguous range window.

## Safety Model For Live Use

Paper mode remains the default. Live mode only activates when all live flags and environment variables are present.

The recommended live architecture is:

```text
WebSocket for fast detection -> REST recheck for all entry legs -> depth-aware sizing -> FOK live order bundle
```

This is important because WebSocket can see good candidates faster than polling, but a `best_bid_ask` update may not include reliable executable depth. The live path uses fresh REST books before size selection and before order placement.

Polymarket does not make multi-leg fills atomic across every leg. The implementation uses FOK limit buys and batch posting when supported by the SDK, but real market movement can still reject or partially affect execution behavior depending on exchange/API semantics.

Each running process keeps its own paper cash/open-position state in memory. If you start a second paper bot, it starts with a fresh budget and can enter a package the first process already holds. In live mode the bot now takes a process lock by default at `data/live_trading.lock` so two live bot processes do not accidentally trade the same opportunities independently. Use `--disable-run-lock` only when you intentionally want isolated live processes.

## One-time Setup

```bash
cd Polybitrage
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For live trading support:

```bash
python -m pip install -e '.[live]'
```

## Generate Today's Crypto Candidates

For May 4, 2026:

```bash
poly-arb discover-tomorrow-crypto-pairs \
  --assets BTC,ETH,SOL,XRP \
  --date-slug may-4 \
  --adjacent-only false \
  --include-boundary-ambiguous true \
  --out config/generated_crypto_may4.yaml
```

## Run Paper N-leg Simulation

```bash
poly-arb paper-arb-sim \
  --scan-all-candidates \
  --candidate-pairs config/generated_crypto_may4.yaml \
  --relation-safety all \
  --book-source websocket \
  --fallback-to-polling true \
  --allow-stale-websocket-cache \
  --websocket-stale-book-ms 5000 \
  --order-book-cache-ms 500 \
  --max-concurrent-requests 30 \
  --no-dynamic-fee-rates \
  --duration-minutes 240 \
  --poll-seconds 0.05 \
  --sizing-mode max_profit \
  --optimizer-net-cutoff 1.05 \
  --entry-threshold 1.0 \
  --min-edge-threshold 0.0025 \
  --budget 10000 \
  --max-trade-size 10000 \
  --max-total-locked-capital 10000 \
  --enable-n-leg-trading \
  --n-leg-sizing-mode optimized \
  --n-leg-max-ranges 2 \
  --dashboard-interval-seconds 1 \
  --report-interval-seconds 10 \
  --scan-log-interval-seconds 1 \
  --show-top 8 \
  --out reports/crypto_may4_fast_sim.csv \
  --trades-out reports/crypto_may4_fast_trades.csv \
  --save-markdown reports/crypto_may4_fast_sim.md
```

## Run Continuously Across Dates

Use this when you want the process to keep running forever. It generates the active date's candidates, runs until the ET rollover time plus delay, then generates the next date and restarts the scan automatically.

The default rollover is `12:00` ET plus a 2 minute delay, matching the noon ET close implied by the 5pm London example. If the market you are trading really rolls at midnight ET, use `--rollover-time-et 00:00`.

```bash
poly-arb paper-arb-sim \
  --continuous-rollover \
  --rollover-time-et 12:00 \
  --rollover-delay-minutes 2 \
  --rollover-assets BTC,ETH,SOL,XRP \
  --rollover-pairs-template 'config/generated_crypto_{date_slug}.yaml' \
  --scan-all-candidates \
  --relation-safety all \
  --book-source websocket \
  --fallback-to-polling true \
  --allow-stale-websocket-cache \
  --websocket-stale-book-ms 5000 \
  --order-book-cache-ms 500 \
  --max-concurrent-requests 30 \
  --no-dynamic-fee-rates \
  --poll-seconds 0.05 \
  --sizing-mode max_profit \
  --optimizer-net-cutoff 1.05 \
  --entry-threshold 1.0 \
  --min-edge-threshold 0.0025 \
  --budget 10000 \
  --max-trade-size 10000 \
  --max-total-locked-capital 10000 \
  --enable-n-leg-trading \
  --n-leg-sizing-mode optimized \
  --n-leg-max-ranges 2 \
  --dashboard-interval-seconds 1 \
  --report-interval-seconds 10 \
  --scan-log-interval-seconds 1 \
  --show-top 8 \
  --out 'reports/crypto_{date_slug}_fast_sim.csv' \
  --trades-out 'reports/crypto_{date_slug}_fast_trades.csv' \
  --save-markdown 'reports/crypto_{date_slug}_fast_sim.md'
```

## Live Trading Environment

Only run live mode if you are legally allowed to trade Polymarket and you understand orders use real money.

```bash
export POLYMARKET_LIVE_TRADING_ENABLED=true
export POLYMARKET_PRIVATE_KEY='...'
export POLYMARKET_FUNDER_ADDRESS='...'
export POLYMARKET_SIGNATURE_TYPE=2
export POLYMARKET_API_KEY='...'              # optional; SDK can derive from private key
export POLYMARKET_API_SECRET='...'           # optional
export POLYMARKET_API_PASSPHRASE='...'       # optional
```

## Run Live N-leg Trading

Start with a tiny session cap:

```bash
poly-arb paper-arb-sim \
  --execution-mode live \
  --live-confirmation I_UNDERSTAND_THIS_USES_REAL_MONEY \
  --live-compliance-ack I_AM_ALLOWED_TO_TRADE_POLYMARKET \
  --live-max-session-spend 25 \
  --live-price-buffer-ticks 0 \
  --live-max-legs-per-bundle 4 \
  --scan-all-candidates \
  --candidate-pairs config/generated_crypto_may4.yaml \
  --relation-safety clean \
  --book-source websocket \
  --fallback-to-polling true \
  --allow-stale-websocket-cache \
  --websocket-stale-book-ms 5000 \
  --order-book-cache-ms 500 \
  --max-concurrent-requests 30 \
  --duration-minutes 30 \
  --poll-seconds 0.05 \
  --sizing-mode max_profit \
  --optimizer-net-cutoff 1.05 \
  --entry-threshold 1.0 \
  --min-edge-threshold 0.0025 \
  --budget 25 \
  --max-trade-size 5 \
  --max-total-locked-capital 25 \
  --enable-n-leg-trading \
  --n-leg-sizing-mode optimized \
  --n-leg-max-ranges 1 \
  --dashboard-interval-seconds 1 \
  --report-interval-seconds 10 \
  --scan-log-interval-seconds 1 \
  --show-top 8 \
  --out reports/live_crypto_may4_scan.csv \
  --trades-out reports/live_crypto_may4_positions.csv \
  --live-orders-out reports/live_crypto_may4_orders.csv \
  --save-markdown reports/live_crypto_may4.md
```

## Run Live Continuously Across Dates

This is the live equivalent. Keep the spend caps tiny while testing.

```bash
poly-arb paper-arb-sim \
  --continuous-rollover \
  --rollover-time-et 12:00 \
  --rollover-delay-minutes 2 \
  --rollover-assets BTC,ETH,SOL,XRP \
  --rollover-pairs-template 'config/generated_crypto_{date_slug}.yaml' \
  --execution-mode live \
  --live-confirmation I_UNDERSTAND_THIS_USES_REAL_MONEY \
  --live-compliance-ack I_AM_ALLOWED_TO_TRADE_POLYMARKET \
  --live-max-session-spend 25 \
  --live-price-buffer-ticks 0 \
  --live-max-legs-per-bundle 4 \
  --scan-all-candidates \
  --relation-safety clean \
  --book-source websocket \
  --fallback-to-polling true \
  --allow-stale-websocket-cache \
  --websocket-stale-book-ms 5000 \
  --order-book-cache-ms 500 \
  --max-concurrent-requests 30 \
  --poll-seconds 0.05 \
  --sizing-mode max_profit \
  --optimizer-net-cutoff 1.05 \
  --entry-threshold 1.0 \
  --min-edge-threshold 0.0025 \
  --budget 25 \
  --max-trade-size 5 \
  --max-total-locked-capital 25 \
  --enable-n-leg-trading \
  --n-leg-sizing-mode optimized \
  --n-leg-max-ranges 1 \
  --dashboard-interval-seconds 1 \
  --report-interval-seconds 10 \
  --scan-log-interval-seconds 1 \
  --show-top 8 \
  --out 'reports/live_crypto_{date_slug}_scan.csv' \
  --trades-out 'reports/live_crypto_{date_slug}_positions.csv' \
  --live-orders-out 'reports/live_crypto_{date_slug}_orders.csv' \
  --save-markdown 'reports/live_crypto_{date_slug}.md'
```

## Validation

```bash
python -m pytest -q
```

Expected result for this branch:

```text
24 passed
```
