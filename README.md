<p align="center">
  <img src="assets/logo.jpeg" alt="Polybitrage logo" width="160">
</p>

<h1 align="center">Polybitrage</h1>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-1f6feb">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-stable-000000?logo=rust&logoColor=white">
  <img alt="C / Cython" src="https://img.shields.io/badge/C-Cython-A8B9CC?logo=c&logoColor=white">
  <img alt="Grafana" src="https://img.shields.io/badge/Monitoring-Grafana%20%2B%20Prometheus-E6522C?logo=grafana&logoColor=white">
</p>

**Polybitrage is a high-frequency trading engine for prediction markets.** It
continuously scans logically related binary markets on Polymarket (and Kalshi),
detects risk-defined arbitrage, sizes each opportunity against live order-book
depth, and either records it in a paper simulation or executes it through a
guarded live path.

The core idea is simple: if one market's outcome is logically contained in
another's, their prices must obey a relationship. When they briefly do not, a
combination of legs (`Buy YES(parent) + Buy NO(child)`, and its N-leg
generalizations) has a guaranteed payout for less than that payout's cost. The
hard part is not the math — it is seeing the mispricing, pricing it against real
depth, and acting before it closes. Polybitrage is built end to end around that
latency budget.

- **Paper mode by default.** No credentials, no orders — it only reads public
  market data and simulates fills against the live book.
- **Guarded live mode.** Real orders are gated behind explicit flags,
  environment credentials, per-session spend caps, and REST rechecks.
- **Instrumented for high-frequency work.** Every stage of the tick-to-trade
  path is measured and exported to Prometheus/Grafana.

> Polybitrage is a personal research and engineering project. It is not
> investment advice and is not a solicitation to trade. See
> [Disclaimer](#disclaimer).

---

## Architecture

| Layer | Role |
|-------|------|
| **Python 3.11+** | Async scan loop, CLI, market discovery, risk sizing, simulation, reporting |
| **Rust** | Zero-copy Polymarket CLOB WebSocket client (`pyo3` extension), Tokio runtime, lock-free `DashMap` order-book state |
| **C (via Cython)** | Compiled hot-path scan classifier (`-O3 -march=native`), with a pure-numpy fallback |

A single-threaded asyncio loop does the scanning and decision-making, fed by the
multi-threaded Rust client over a GIL-released hand-off so the event loop never
blocks on I/O. Full data-flow and a runtime diagram are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Performance

The design goal is to take the bot's *own* latency out of the tick-to-trade path
so that capture rate is bounded by the network, not by the code. It gets there:
with the WebSocket cache warm, the entire hot path — diffing the changed tokens,
classifying the full pair universe in the compiled kernel, sizing, and deciding —
runs in **microseconds**. The only millisecond-scale costs left are round trips
to the exchange, which are a physical floor that no software can beat.

All timings come from nanosecond-resolution `perf_counter_ns` instrumentation;
the latency histograms bucket from 50 µs upward, and a scan is flagged "slow"
only above 5 ms. Figures are indicative, from a development machine against the
live Polymarket CLOB — not a benchmark guarantee.

### What the bot controls — compute (microseconds)

With the WebSocket cache warm (75 pairs, 32 unique tokens):

| Hot-path stage | Typical |
|----------------|---------|
| Dirty-set diff (changed tokens only) | 3 µs |
| Order-book cache read (Rust DashMap) | 45 µs |
| Classify full pair universe (C/Cython kernel) | 1 µs |
| N-leg scan + filter + depth-aware sizing | 10 µs |
| **Full steady-state scan** | **~59 µs** |


### The physical floor — network (milliseconds)

These are round trips to the exchange. They dominate only when a fresh REST read
is required, and they cannot be optimized in software — only by co-locating
closer to the venue.

| Network-bound stage | Typical |
|---------------------|---------|
| Order-book fetch, REST batch | ~220–295 ms |
| REST recheck immediately before entry | one round trip |
| Order round-trip | network-bound |

### Book source, end to end

| Book source | First / fresh scan | Steady-state scan |
|-------------|--------------------|-------------------|
| WebSocket (Rust cache) | seeds from REST once | **sub-100 µs** while connected |
| Polling + short cache | ~60–90 ms | ~0 ms on cache hit |
| Polling (REST, no cache) | ~220–295 ms | — |

Warming the WebSocket cache turns a ~240 ms REST round trip into a
microsecond-scale in-memory read — roughly three to four orders of magnitude off
the dominant cost of every scan.

---

## Repository layout

Everything lives in this one folder — clone it and you are looking at the whole
project.

```
Polybitrage/
├── src/                    Application code (Python)
│   ├── main.py             CLI entry point (poly-arb / polybitrage)
│   ├── simulator.py        Paper/live scan loop, sizing, risk controls
│   ├── discovery.py        Polymarket candidate generation (Gamma API)
│   ├── polymarket/         CLOB + Gamma clients, WS clients, live trader, models
│   ├── kalshi/             Kalshi REST + FIX clients, discovery, live trader
│   ├── worldcup/           World Cup market relation scanner
│   ├── fast/               Cython/C hot-path classifier + numpy fallback
│   ├── network/            Signal bridge
│   ├── audit*.py           Forensic audit capture + profit-leakage reports
│   └── *_profiler.py       Latency, market, and system profilers
├── rust_ws_client/         Rust pyo3 WebSocket market-data extension (polymarket_rs)
├── config/                 Example pair / watchlist configs
├── docs/                   Architecture, CLI, N-leg/live, audit, Kalshi, World Cup
├── monitoring/             Prometheus + Grafana stack (docker-compose)
├── scripts/                Network latency probe
├── tests/                  pytest suite
├── reports/                Generated CSV / Markdown / audit output (gitignored)
├── assets/                 Logo and images
├── .github/                Issue / PR templates, contributing, code of conduct, security
├── pyproject.toml          Package + dependency definitions
├── setup_cython.py         Builds the C hot-path extension
├── supervisord.conf        Process management for headless deployment
├── LICENSE                 MIT license
└── .env.example            Environment variable template
```

---

## Quick start

### 1. Install (paper mode — no credentials needed)

```bash
cd Polybitrage
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

### 2. Run a single smoke-test scan

```bash
poly-arb paper-arb-sim --pairs config/watchlist_pairs.yaml --once
```

### 3. Run the paper simulator

```bash
poly-arb paper-arb-sim --pairs config/watchlist_pairs.yaml --budget 100 --book-source websocket --sizing-mode max_profit --entry-threshold 1.0 --min-edge-threshold 0.0025 --max-trade-size 20 --out reports/paper_sim.csv --trades-out reports/paper_trades.csv --save-markdown reports/paper_sim.md
```

The console prints only the live essentials each poll (cash, locked capital,
open positions, best pair, distance to entry, action taken). Detailed rows go to
`reports/`. If `--duration-minutes` is omitted, it runs until `Ctrl-C`.

---

## What to touch, and when

Everything below is configuration — CLI flags plus the YAML files in `config/`.
No source code needs editing.

### The one file you will actually edit

[`config/watchlist_pairs.yaml`](config/watchlist_pairs.yaml) is the list of
market pairs the bot watches. Each entry carries the two markets' token IDs and
an `enabled:` flag — set `enabled: true` on the pairs you want live and `false`
to park them. Paper mode needs nothing else.

### Finding new opportunities (discovery → watchlist)

Two commands, run in sequence. Discovery scans Polymarket for candidate pairs;
`build-watchlist` then ranks them by liquidity and edge into a smaller,
executable list you point the simulator at.

```bash
poly-arb discover-tomorrow-crypto-pairs --assets BTC,ETH,SOL,XRP --days-ahead 1 --out config/generated_crypto_tomorrow.yaml
poly-arb build-watchlist --pairs config/generated_crypto_tomorrow.yaml --top-n 10 --out config/clean_watchlist.yaml
```

### Risk controls

These flags bound how much the bot can commit and how good an edge must be
before it enters:

| Flag | Controls |
|------|----------|
| `--min-edge-threshold` | Minimum net edge (after fees) required to enter |
| `--max-trade-size` | Cap on capital spent per single entry |
| `--max-total-locked-capital` | Cap on total capital locked across all open positions |
| `--max-open-positions` | Cap on concurrent open positions |

### Everyday toggles

| Task | How |
|------|-----|
| Change the book source | `--book-source websocket` (fast, default) or `--book-source polling` |
| Run continuously across daily market rollover | Add `--continuous-rollover --rollover-time-et 12:00` |
| Check scan speed before going live | `poly-arb benchmark-scan --iterations 5` |
| See the full command reference | `poly-arb --help` and [`docs/CLI_AND_LIBRARY.md`](docs/CLI_AND_LIBRARY.md) |

---

## Configuration

| Variable | Purpose |
|----------|---------|
| `POLYMARKET_GAMMA_BASE_URL` | Gamma API base (market discovery) |
| `POLYMARKET_CLOB_BASE_URL` | CLOB REST base (order books) |
| `POLYMARKET_WS_MARKET_URL` | Public market WebSocket URL |
| `POLYMARKET_LIVE_TRADING_ENABLED` | Master switch for live mode (`false` by default) |
| `POLYMARKET_PRIVATE_KEY` | Wallet key — **live only**, never commit |
| `POLYMARKET_API_KEY` / `_SECRET` / `_PASSPHRASE` | CLOB API credentials (SDK can derive from the private key) |
| `POLYMARKET_FUNDER_ADDRESS` | Funding wallet address (live only) |
| `POLYMARKET_SIGNATURE_TYPE` / `_CHAIN_ID` | Signing configuration |

Public market data needs no credentials, so paper mode requires nothing here.
`.env` is gitignored, and no credentials are stored in this repository.

---

## Monitoring

The bot exposes Prometheus metrics at `/metrics` on its health port. A ready-to
-run Prometheus + Grafana stack auto-provisions a 60-panel dashboard covering
tick-to-trade latency, scanner stage breakdowns, WebSocket health, order-book
microstructure, OS/scheduler jitter, network RTT, and portfolio PnL.

```bash
poly-arb paper-arb-sim --headless --profile --health-host 0.0.0.0 --health-port 8765 --pairs config/watchlist_pairs.yaml
```

```bash
cd monitoring && docker compose up -d
```

Grafana is then at `http://localhost:3000` and Prometheus at
`http://localhost:9090`. Full metric reference and dashboard walkthrough:
[`monitoring/README.md`](monitoring/README.md).

---

## Deployment and specifications

| Component | Specification |
|-----------|---------------|
| **Host** | Linux, compute-optimized cloud instance (for example AWS EC2), region chosen for lowest RTT to the venue |
| **Region choice** | Measure candidates with [`scripts/latency_probe.py`](scripts/latency_probe.py) and pick the lowest TCP/HTTPS RTT before committing |
| **Runtime** | Python 3.11+ (Rust toolchain and C/Cython optional, for the native fast paths) |
| **Process supervision** | [`supervisord.conf`](supervisord.conf) — auto-restart with backoff, log rotation |
| **Observability** | Prometheus + Grafana via [`monitoring/docker-compose.yml`](monitoring/docker-compose.yml) |
| **Health / metrics** | `GET /health` and `GET /metrics` on the configured health port |

Edit the `directory` and `command` lines in `supervisord.conf` to match your
deployment path, then:

```bash
pip install supervisor
supervisord -c supervisord.conf
supervisorctl -c supervisord.conf status
```

---

## Live trading

Live mode places real orders with real money and is **disabled by default.** Do
not enable it unless you are legally permitted to trade on the venue and accept
full responsibility for the outcome.

Live mode adds several independent guards: the `POLYMARKET_LIVE_TRADING_ENABLED`
switch, explicit confirmation and compliance flags, environment-supplied
credentials, fill-or-kill order posting, REST rechecks immediately before entry,
per-session spend caps, per-bundle leg caps, and a process lock preventing two
live bots from trading the same wallet.

```bash
python -m pip install -e '.[live]'
```

The full guarded workflow is documented in
[`docs/N_LEG_AND_LIVE_UPDATE.md`](docs/N_LEG_AND_LIVE_UPDATE.md).

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Runtime design, threading model, data flow diagram |
| [`docs/CLI_AND_LIBRARY.md`](docs/CLI_AND_LIBRARY.md) | Full command reference and library usage |
| [`docs/N_LEG_AND_LIVE_UPDATE.md`](docs/N_LEG_AND_LIVE_UPDATE.md) | N-leg packages and the guarded live executor |
| [`docs/FORENSIC_AUDIT.md`](docs/FORENSIC_AUDIT.md) | Forensic audit capture and profit-leakage reporting |
| [`docs/KALSHI_SUPPORT.md`](docs/KALSHI_SUPPORT.md) | Kalshi discovery and read-only scanning |
| [`docs/WORLDCUP.md`](docs/WORLDCUP.md) | World Cup multi-market relation scanning |
| [`monitoring/README.md`](monitoring/README.md) | Prometheus + Grafana setup and metric reference |

---

## Contributing

Contributions are welcome. Areas that are especially useful for this project:

- **New venue adapters** or order-book providers beyond Polymarket CLOB and Kalshi
- **Arbitrage relations and strategies** — additional N-leg packages, sizing, and exit logic
- **Latency work** on the hot path — the Rust WebSocket client, the Cython scan kernel, or the async scan loop
- **Live-execution safety** — risk controls, order handling, and reconciliation
- **Observability** — new Grafana panels or Prometheus metrics
- **Market discovery** — new candidate sources and pair-generation heuristics
- **Documentation, tests, and reproducibility**

The short version: check the open issues, fork, create a branch, add tests where
it makes sense, and open a pull request. For anything substantial — and for any
change to the live-trading path — open an issue first so the approach can be
discussed. Full details, including local setup and the PR checklist, are in
[`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md); all participation is
governed by the [Code of Conduct](.github/CODE_OF_CONDUCT.md). To report a
security issue, see [`.github/SECURITY.md`](.github/SECURITY.md).

## License

Released under the [MIT License](LICENSE).

---

## Disclaimer

This project is provided for research and educational purposes only. It is not
financial, investment, or trading advice, and nothing here is a recommendation
to trade any market. Prediction-market trading may be restricted or illegal in
your jurisdiction; you are solely responsible for complying with all applicable
laws and with each venue's terms of service. Live trading risks real financial
loss. Use at your own risk.
