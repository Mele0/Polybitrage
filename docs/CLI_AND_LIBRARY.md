# CLI and Library Usage

The project is now installable as a small Python package with console commands. The current package name is still intentionally conservative, so the import path remains `src.*` for now. A deeper namespace rename can happen later without changing the trading logic.

## Install

From the project folder:

```bash
cd Polybitrage
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For live trading dependencies:

```bash
python -m pip install -e '.[live]'
```

## Command Line

After install, use either command:

```bash
poly-arb --help
polymarket-arb --help
```

The old form still works:

```bash
python -m src.main --help
```

## Commands

- `poly-arb paper-arb-sim --help`
  Run the scanner. Use this for normal paper monitoring, live trading, and continuous date rollover.

- `poly-arb discover-tomorrow-crypto-pairs --help`
  Generate dated crypto range/threshold candidate YAML from Polymarket Gamma event pages.

- `poly-arb build-watchlist --help`
  Build a smaller active watchlist from a broad generated candidate universe.

- `poly-arb benchmark-scan --help`
  Measure scan speed before deciding whether to scan all candidates in the hot loop.

## Examples

Generate candidates:

```bash
poly-arb discover-tomorrow-crypto-pairs \
  --assets BTC,ETH,SOL,XRP \
  --date-slug may-4 \
  --adjacent-only false \
  --include-boundary-ambiguous true \
  --out config/generated_crypto_may4.yaml
```

Run continuously across every day Polymarket currently lists, with no config files (recommended for crypto N-leg scanning):

```bash
poly-arb paper-arb-sim \
  --live-universe \
  --live-universe-assets BTC,ETH,SOL,XRP \
  --live-universe-horizon-days 14 \
  --live-universe-refresh-seconds 900 \
  --relation-safety all \
  --book-source websocket \
  --fallback-to-polling true \
  --allow-stale-websocket-cache \
  --poll-seconds 0.05 \
  --sizing-mode max_profit \
  --entry-threshold 1.0 \
  --min-edge-threshold 0.0025 \
  --budget 10000 \
  --max-trade-size 10000 \
  --max-total-locked-capital 10000 \
  --enable-n-leg-trading \
  --n-leg-sizing-mode optimized \
  --n-leg-max-ranges 2 \
  --out 'reports/crypto_live_universe_sim.csv' \
  --trades-out 'reports/crypto_live_universe_trades.csv' \
  --save-markdown 'reports/crypto_live_universe_sim.md'
```

`--live-universe` discovers candidate pairs directly from Polymarket's Gamma API for every day currently listed (today plus `--live-universe-horizon-days - 1` days ahead) and periodically refreshes that universe while the bot runs — no `discover-tomorrow-crypto-pairs` / `--candidate-pairs` / `--continuous-rollover` step is needed. Each discovered pair is tagged with the market's `event_date`, which is threaded through to trade and audit records (`reports/*_trades.csv` and the `n_leg_candidates`/`paper_arb_trades` audit datasets) so you can see which calendar day an opportunity belonged to.

The older file-based pipeline below still works and remains useful for offline debugging or generating a fixed snapshot of a single day's candidates:

```bash
poly-arb paper-arb-sim \
  --continuous-rollover \
  --rollover-time-et 12:00 \
  --rollover-delay-minutes 2 \
  --rollover-pairs-template 'config/generated_crypto_{date_slug}.yaml' \
  --scan-all-candidates \
  --relation-safety all \
  --book-source websocket \
  --fallback-to-polling true \
  --allow-stale-websocket-cache \
  --poll-seconds 0.05 \
  --sizing-mode max_profit \
  --entry-threshold 1.0 \
  --min-edge-threshold 0.0025 \
  --budget 10000 \
  --max-trade-size 10000 \
  --max-total-locked-capital 10000 \
  --enable-n-leg-trading \
  --n-leg-sizing-mode optimized \
  --n-leg-max-ranges 2 \
  --out 'reports/crypto_{date_slug}_fast_sim.csv' \
  --trades-out 'reports/crypto_{date_slug}_fast_trades.csv' \
  --save-markdown 'reports/crypto_{date_slug}_fast_sim.md'
```

## Importable API

The main pieces are importable:

```python
from pathlib import Path

from src.discovery import discover_crypto_pairs
from src.simulator import PaperArbSimulator, SimulatorSettings

settings = SimulatorSettings(
    pairs_path=Path("config/generated_crypto_may4.yaml"),
    scan_all_candidates=True,
    candidate_pairs_path=Path("config/generated_crypto_may4.yaml"),
    enable_n_leg_trading=True,
)
```

The CLI is the recommended interface for now. The importable API is useful for tests, notebooks, or future orchestration code, but it has not been frozen as a stable public API yet.
