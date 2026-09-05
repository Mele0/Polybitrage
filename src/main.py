from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import replace
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.audit_profit_report import AuditProfitReportConfig, generate_audit_profit_report, parse_float_csv
from src.audit_report import generate_audit_report
from src.discovery import discover_crypto_pairs
from src.kalshi.discovery import discover_kalshi_pairs
from src.simulator import PaperArbSimulator, SimulatorSettings, benchmark_scan, build_watchlist_once

ET_ZONE = ZoneInfo("America/New_York")
TOP_LEVEL_DESCRIPTION = """Polymarket/Kalshi pair and N-leg arbitrage toolkit.

Use this command to discover dated crypto candidates, run paper scans, benchmark scan speed,
build smaller watchlists, or run the guarded live executor. Paper mode is the default.
"""
TOP_LEVEL_EPILOG = """When to use each command:
  paper-arb-sim                  Run the scanner. Use this for paper mode, live mode, and continuous rollover.
  discover-tomorrow-crypto-pairs Generate dated BTC/ETH/SOL/XRP pair candidates from Gamma event pages.
  discover-kalshi-pairs          Generate Kalshi same-market YES+NO candidate YAML from public market data.
  build-watchlist                Shrink a broad candidate YAML into a liquid executable watchlist.
  benchmark-scan                 Measure scan speed before choosing polling/websocket settings.
  audit-report                   Build forensic latency/fill/miss reports from an audit run.
  audit-profit-report            Build profit leakage and improvement forensics from an audit run.

Common help commands:
  poly-arb --help
  poly-arb paper-arb-sim --help
  poly-arb discover-tomorrow-crypto-pairs --help
  poly-arb discover-kalshi-pairs --help

Equivalent legacy entrypoint:
  python -m src.main <command> [options]
"""
SIM_EPILOG = """Examples:
  Paper scan:
    poly-arb paper-arb-sim --scan-all-candidates --candidate-pairs config/generated_crypto_may4.yaml

  Continuous dated crypto scan:
    poly-arb paper-arb-sim --continuous-rollover --rollover-time-et 12:00 --enable-n-leg-trading

  Live trading, guarded by flags and environment variables:
    poly-arb paper-arb-sim --execution-mode live --live-max-session-spend 25 ...

Use WebSocket for Polymarket speed, with entry REST rechecks enabled. Use polling when comparing against REST-only behavior.
Kalshi currently supports read-only paper/audit scans through REST polling.
"""
DISCOVER_EPILOG = """Examples:
  Tomorrow's crypto candidates:
    poly-arb discover-tomorrow-crypto-pairs --assets BTC,ETH,SOL,XRP --days-ahead 1

  Specific date:
    poly-arb discover-tomorrow-crypto-pairs --date-slug may-4 --out config/generated_crypto_may4.yaml
"""
KALSHI_DISCOVER_EPILOG = """Examples:
  Open Kalshi markets as same-market YES+NO complement candidates:
    poly-arb discover-kalshi-pairs --status open --max-markets 100 --out config/generated_kalshi_pairs.yaml

  Filter by series/event/tickers:
    poly-arb discover-kalshi-pairs --series-ticker KXBTC --status open --out config/kalshi_btc.yaml
    poly-arb discover-kalshi-pairs --tickers KXEXAMPLE-26JAN01-T50,KXEXAMPLE-26JAN01-B50
"""
WATCHLIST_EPILOG = """Example:
  poly-arb build-watchlist --pairs config/generated_crypto_may4.yaml --relation-safety clean --top-n 20
"""
BENCHMARK_EPILOG = """Example:
  poly-arb benchmark-scan --pairs config/generated_crypto_may4.yaml --iterations 10
"""
AUDIT_EPILOG = """Examples:
  Single run report:
    poly-arb audit-report reports/audit/20260507T120000Z-12345 --top-missed 25 --format html,markdown,json

  Compare two runs:
    poly-arb audit-report reports/audit/new-run --compare reports/audit/old-run --format html
"""
AUDIT_PROFIT_EPILOG = """Example:
  poly-arb audit-profit-report \\
    --audit-dir reports/audit/20260508T082009Z-76506 \\
    --out-dir reports/audit/20260508T082009Z-76506/profit_forensics \\
    --top-windows 50 \\
    --freshness-ms 250,500,1000,2000,5000 \\
    --edge-threshold-grid 0,0.001,0.0025,0.005,0.01
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poly-arb",
        description=TOP_LEVEL_DESCRIPTION,
        epilog=TOP_LEVEL_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-roi-threshold",
        type=float,
        dest="global_min_roi_threshold",
        default=None,
        help=argparse.SUPPRESS,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    sim = sub.add_parser(
        "paper-arb-sim",
        help="Run paper scans, live mode, or continuous rollover.",
        description=(
            "Run the scanner against a configured pair file or generated candidate universe. "
            "This is the command to use for normal paper monitoring, guarded live trading, "
            "and never-ending date rollover."
        ),
        epilog=SIM_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sim.add_argument("--pairs", type=Path, default=Path("config/watchlist_pairs.yaml"))
    sim.add_argument(
        "--exchange",
        choices=["polymarket", "kalshi"],
        default="polymarket",
        help="Market venue/API backend. Kalshi support is paper/audit REST polling only for now.",
    )
    sim.add_argument("--budget", type=float, default=100.0)
    sim.add_argument(
        "--duration-minutes",
        type=float,
        default=None,
        help="Optional runtime limit. If omitted, the simulator runs until stopped with Ctrl-C.",
    )
    sim.add_argument("--poll-seconds", type=float, default=0.25)
    sim.add_argument("--near-arb-threshold", type=float, default=1.02)
    sim.add_argument("--entry-threshold", type=float, default=1.0)
    sim.add_argument(
        "--min-edge-threshold",
        type=float,
        default=0.0025,
        help="Minimum net edge per share after fees/slippage. Set 0 for every positive paper arb.",
    )
    sim.add_argument(
        "--min-roi-threshold",
        type=float,
        default=0.0,
        help="Skip entries whose ROI is below this threshold (for example 0.01 = 1%%).",
    )
    sim.add_argument("--max-trade-size", type=float, default=20.0, help="Maximum dollars to lock per paper entry.")
    sim.add_argument("--min-trade-size", type=float, default=1.0, help="Minimum dollars to lock per paper entry.")
    sim.add_argument("--capital-fraction-per-trade", type=float, default=1.0)
    sim.add_argument("--sizing-mode", choices=["fixed", "max_profit"], default="max_profit")
    sim.add_argument(
        "--optimizer-net-cutoff",
        type=float,
        default=1.05,
        help=(
            "Only run the expensive depth optimizer for pairs whose top-of-book net "
            "cost is at or below this value. Deeper ask levels cannot improve a buy "
            "price, so high values mostly waste scan time."
        ),
    )
    sim.add_argument("--allow-multiple-open-per-pair", action="store_true")
    sim.add_argument("--cooldown-seconds-per-pair", type=float, default=30.0)
    sim.add_argument("--max-open-positions", type=int, default=None)
    sim.add_argument("--max-open-positions-per-pair", type=int, default=None)
    sim.add_argument("--max-total-locked-capital", type=float, default=None)
    sim.add_argument("--include-disabled", action="store_true")
    sim.add_argument("--slippage-buffer", type=float, default=None)
    sim.add_argument("--fee-rate", type=float, default=None, help="Optional manual fee rate override. Default fetches public CLOB fee rates, then falls back to YAML/category estimates.")
    sim.add_argument(
        "--dynamic-fee-rates",
        dest="dynamic_fee_rates",
        action="store_true",
        default=True,
        help="Fetch token fee rates from the public CLOB /fee-rate endpoint when --fee-rate is not set.",
    )
    sim.add_argument(
        "--no-dynamic-fee-rates",
        dest="dynamic_fee_rates",
        action="store_false",
        help="Disable dynamic fee lookup and use YAML/category fallback rates instead.",
    )
    sim.add_argument("--max-concurrent-requests", type=int, default=10)
    sim.add_argument(
        "--kalshi-base-url",
        default="https://external-api.kalshi.com/trade-api/v2",
        help="Kalshi Trade API base URL used when --exchange kalshi.",
    )
    sim.add_argument(
        "--kalshi-fix-api-key",
        default="",
        help="Kalshi FIX API key (SenderCompID). Also read from KALSHI_FIX_API_KEY env var.",
    )
    sim.add_argument(
        "--kalshi-fix-api-secret",
        default="",
        help="Kalshi FIX API secret (Password). Also read from KALSHI_FIX_API_SECRET env var.",
    )
    sim.add_argument(
        "--kalshi-fix-host",
        default="mm.fix.elections.kalshi.com",
        help="Kalshi FIX endpoint hostname.",
    )
    sim.add_argument(
        "--kalshi-fix-port",
        type=int,
        default=8233,
        help="Kalshi FIX port. 8233=no-retransmission (fastest), 8234=with-retransmission.",
    )
    sim.add_argument("--book-source", choices=["polling", "websocket", "fix", "rust-websocket"], default="websocket")
    sim.add_argument("--order-book-cache-ms", type=int, default=0)
    sim.add_argument("--websocket-url", default="wss://ws-subscriptions-clob.polymarket.com/ws/market")
    sim.add_argument("--websocket-stale-book-ms", type=int, default=2000)
    sim.add_argument(
        "--websocket-fallback-cache-ms",
        type=int,
        default=10000,
        help=(
            "Minimum milliseconds before retrying REST fallback for the same missing/stale "
            "WebSocket token. Prevents full-universe REST refetches from blocking every scan."
        ),
    )
    sim.add_argument("--websocket-reconnect-seconds", type=float, default=5.0)
    sim.add_argument("--fallback-to-polling", type=_parse_bool, default=True)
    sim.add_argument(
        "--allow-stale-websocket-cache",
        action="store_true",
        help=(
            "For WebSocket mode, keep displaying cached books after the stale threshold "
            "instead of refreshing the whole watchlist through REST. Entries are still "
            "protected by --entry-rest-recheck."
        ),
    )
    sim.add_argument(
        "--entry-rest-recheck",
        type=_parse_bool,
        default=True,
        help="Before opening a paper position, re-fetch the two required books through REST and recompute entry cost.",
    )
    sim.add_argument(
        "--background-entry",
        action="store_true",
        default=False,
        help=(
            "Spawn N-leg and pair entry REST rechecks as background asyncio tasks so the "
            "scan loop is never blocked by a 50–200 ms REST round-trip.  Capital is "
            "pre-reserved before the first await to prevent double-spending.  Requires "
            "--entry-rest-recheck (default on).  Action strings are reported in the "
            "following scan row."
        ),
    )
    sim.add_argument(
        "--near-arb-priority-seed",
        type=_parse_bool,
        default=True,
        help=(
            "Aggressively REST-seed tokens belonging to near-arb pairs "
            "(net_total_cost ≤ --near-arb-threshold) so stale WebSocket prices are "
            "cleared within near_arb_seed_interval_seconds rather than waiting for "
            "the full fallback_cache_ms window.  Only active when --book-source=websocket."
        ),
    )
    sim.add_argument("--relation-safety", choices=["all", "clean", "boundary_ambiguous"], default="all")
    sim.add_argument("--allow-boundary-ambiguous-guaranteed", action="store_true")
    sim.add_argument("--min-display-price", type=float, default=None)
    sim.add_argument("--max-threshold-distance-pct", type=float, default=None)
    sim.add_argument("--spot-prices", default="", help="Comma-separated spot prices like BTC=76339,ETH=2300.")
    sim.add_argument("--exit-mode", choices=["hold_until_resolution", "close_when_edge_converges"], default="hold_until_resolution")
    sim.add_argument("--take-profit-pct", type=float, default=0.0025)
    sim.add_argument("--stop-loss-pct", type=float, default=0.005)
    sim.add_argument("--take-profit-absolute", type=float, default=None)
    sim.add_argument("--stop-loss-absolute", type=float, default=None)
    sim.add_argument("--out", type=Path, default=Path("reports/paper_arb_sim_clean.csv"))
    sim.add_argument("--trades-out", type=Path, default=Path("reports/paper_arb_trades_clean.csv"))
    sim.add_argument("--live-orders-out", type=Path, default=Path("reports/live_orders.csv"))
    sim.add_argument("--save-markdown", type=Path, default=Path("reports/paper_arb_sim_clean.md"))
    sim.add_argument("--disable-markdown", action="store_true")
    sim.add_argument("--scan-log-interval-seconds", type=float, default=0.0, help="Throttle scan CSV writes; trades are still always logged.")
    sim.add_argument("--clear-screen", action="store_true")
    sim.add_argument(
        "--live-dashboard",
        dest="live_dashboard",
        action="store_true",
        default=True,
        help="Use one Rich live-updating dashboard instead of printing a new panel each refresh.",
    )
    sim.add_argument(
        "--no-live-dashboard",
        dest="live_dashboard",
        action="store_false",
        help="Disable the Rich live dashboard and print each refresh normally.",
    )
    sim.add_argument(
        "--headless",
        action="store_true",
        help="Daemon mode: zero console output (no banner, dashboard, per-scan CSV, "
             "markdown, or profiler summary). The bot only trades and serves telemetry "
             "at http://<health-host>:<health-port>/metrics for Prometheus/Grafana. "
             "Pair with --profile so latency series populate. Lowest per-scan overhead.",
    )
    sim.add_argument(
        "--health-host", default="127.0.0.1",
        help="Bind host for the health + Prometheus /metrics server (default 127.0.0.1). "
             "Use 0.0.0.0 to allow Docker/remote Prometheus scraping (trusted networks only).",
    )
    sim.add_argument(
        "--health-port", type=int, default=8765,
        help="Port for the health + /metrics server (default 8765). Set to 0 to disable it.",
    )
    sim.add_argument("--once", action="store_true", help="Run one scan and exit.")
    sim.add_argument("--show-top", type=int, default=3, help="Show the N cheapest quoted pairs in the compact dashboard.")
    sim.add_argument("--dashboard-interval-seconds", type=float, default=0.25, help="Throttle terminal dashboard updates while still logging scans.")
    sim.add_argument("--report-interval-seconds", type=float, default=5.0, help="Throttle Markdown report writes to reduce IO during fast scans.")
    sim.add_argument("--dynamic-watchlist", action="store_true", help="Refresh a small active watchlist from a broad candidate file while running.")
    sim.add_argument(
        "--live-universe",
        action="store_true",
        help="Continuously discover crypto range/threshold pairs across every day Polymarket currently lists "
        "(no config files needed). Replaces --pairs/--candidate-pairs/--continuous-rollover for crypto N-leg scanning.",
    )
    sim.add_argument(
        "--live-universe-assets",
        default="BTC,ETH,SOL,XRP",
        help="Comma-separated asset symbols to discover with --live-universe (default BTC,ETH,SOL,XRP).",
    )
    sim.add_argument(
        "--live-universe-horizon-days",
        type=int,
        default=14,
        help="How many days ahead (including today) to scan for dated markets with --live-universe (default 14).",
    )
    sim.add_argument(
        "--live-universe-refresh-seconds",
        type=float,
        default=900.0,
        help="How often to re-discover the live multi-day pair universe while running (default 900s).",
    )
    sim.add_argument("--scan-all-candidates", action="store_true", help="Scan the full candidate universe every poll. Fastest way to avoid missing pairs when the universe is small enough.")
    sim.add_argument("--execution-mode", choices=["paper", "live"], default="paper")
    sim.add_argument("--live-confirmation", default="", help="Must equal I_UNDERSTAND_THIS_USES_REAL_MONEY for live trading.")
    sim.add_argument("--live-compliance-ack", default="", help="Must equal I_AM_ALLOWED_TO_TRADE_POLYMARKET for live trading.")
    sim.add_argument("--live-max-session-spend", type=float, default=None, help="Hard cap on total requested live order notional for this process.")
    sim.add_argument("--live-price-buffer-ticks", type=int, default=0, help="Add this many ticks to each buy leg limit price after REST recheck.")
    sim.add_argument("--live-max-legs-per-bundle", type=int, default=5, help="Refuse live N-leg bundles larger than this many legs.")
    sim.add_argument("--live-signature-type", type=int, default=None, help="Override POLYMARKET_SIGNATURE_TYPE. 0 EOA, 1 POLY_PROXY, 2 GNOSIS_SAFE.")
    sim.add_argument(
        "--run-lock-path",
        type=Path,
        default=None,
        help="Optional process lock path. Live mode defaults to data/live_trading.lock to prevent accidental duplicate live bots.",
    )
    sim.add_argument(
        "--disable-run-lock",
        action="store_true",
        help="Disable the process lock. Only use this if you intentionally run isolated bot processes.",
    )
    sim.add_argument(
        "--enable-n-leg-trading",
        "--enable-three-leg-trading",
        dest="enable_n_leg_trading",
        action="store_true",
        help="Allow paper entries for best detected N-leg range/threshold opportunities when gross edge is positive.",
    )
    sim.add_argument(
        "--n-leg-sizing-mode",
        "--three-leg-sizing-mode",
        dest="n_leg_sizing_mode",
        choices=["optimized", "max_trade"],
        default="optimized",
        help="For N-leg entries: use depth optimizer or spend max allowed capital directly.",
    )
    sim.add_argument(
        "--n-leg-max-ranges",
        type=int,
        default=1,
        help=(
            "Maximum contiguous range markets to combine into one N-leg package. "
            "The default 1 preserves 3-leg behavior; use 0 for every contiguous window."
        ),
    )
    sim.add_argument(
        "--allow-incremental-n-leg",
        dest="allow_incremental_n_leg",
        action="store_true",
        default=False,
        help=(
            "Allow a second (incremental) N-leg entry on the same opportunity when "
            "residual profitable depth is detected beyond the current open position. "
            "Requires --enable-n-leg-trading. Controlled by --incremental-n-leg-min-profit."
        ),
    )
    sim.add_argument(
        "--incremental-n-leg-min-profit",
        dest="incremental_n_leg_min_profit",
        type=float,
        default=0.50,
        help="Minimum expected dollar profit for an incremental N-leg entry (default 0.50).",
    )
    sim.add_argument("--candidate-pairs", type=Path, default=None, help="Broad candidate universe used when --dynamic-watchlist is set.")
    sim.add_argument("--watchlist-refresh-seconds", type=float, default=60.0)
    sim.add_argument("--watchlist-top-n", type=int, default=10)
    sim.add_argument("--watchlist-max-net-cost", type=float, default=1.05)
    sim.add_argument("--watchlist-min-depth", type=float, default=1.0)
    sim.add_argument("--watchlist-out", type=Path, default=Path("config/clean_dynamic_watchlist_pairs.yaml"))
    sim.add_argument("--watchlist-ranking-csv", type=Path, default=Path("reports/clean_dynamic_watchlist_ranking.csv"))
    sim.add_argument(
        "--continuous-rollover",
        action="store_true",
        help=(
            "Run forever across dated crypto events. At the ET rollover time plus delay, "
            "generate the next date's pairs and restart the simulator with those pairs."
        ),
    )
    sim.add_argument(
        "--rollover-time-et",
        default="12:00",
        help="ET wall-clock time when the active dated crypto market is considered closed. Use 12:00 for noon or 00:00 for midnight.",
    )
    sim.add_argument(
        "--rollover-delay-minutes",
        type=float,
        default=2.0,
        help="Wait this many minutes after the ET rollover time before switching to the next date.",
    )
    sim.add_argument(
        "--rollover-retry-seconds",
        type=float,
        default=60.0,
        help="If next-date discovery has no usable pairs yet, wait this long and retry instead of exiting.",
    )
    sim.add_argument("--rollover-assets", default="BTC,ETH,SOL,XRP")
    sim.add_argument(
        "--rollover-pairs-template",
        default="config/generated_crypto_{date_slug}.yaml",
        help="Candidate YAML path template. Available placeholders: {date_slug}, {date_compact}, {date_iso}.",
    )
    sim.add_argument(
        "--rollover-adjacent-only",
        type=_parse_bool,
        default=False,
        help="Discovery setting for continuous rollover pair generation.",
    )
    sim.add_argument(
        "--rollover-include-boundary-ambiguous",
        type=_parse_bool,
        default=True,
        help="Discovery setting for continuous rollover pair generation.",
    )
    sim.add_argument(
        "--rollover-min-display-price",
        type=float,
        default=0.001,
        help="Discovery setting for continuous rollover pair generation.",
    )
    # ── Phase 3: two-node signal bridge ───────────────────────────────────────
    sim.add_argument(
        "--signal-bridge-mode",
        choices=["off", "sender", "receiver"],
        default="off",
        help=(
            "Two-node UDP signal coordination. 'sender' (London/Polymarket node) emits "
            "HMAC-signed UDP datagrams when an opportunity is detected. 'receiver' (Chicago/"
            "Kalshi node) listens and executes via FIX Order Entry."
        ),
    )
    sim.add_argument(
        "--signal-bridge-host",
        default="127.0.0.1",
        help="For sender: target Chicago node IP. For receiver: bind address.",
    )
    sim.add_argument(
        "--signal-bridge-port",
        type=int,
        default=9999,
        help="UDP port for the signal bridge.",
    )
    sim.add_argument(
        "--signal-bridge-secret",
        default="",
        help="HMAC-SHA256 shared secret. Also read from SIGNAL_BRIDGE_SECRET env var.",
    )
    sim.add_argument("--audit-mode", choices=["off", "forensic"], default="off")
    sim.add_argument("--audit-dir", type=Path, default=Path("reports/audit"))
    sim.add_argument("--audit-run-id", default=None)
    sim.add_argument("--audit-raw-ws", type=_parse_bool, default=True)
    sim.add_argument("--audit-flush-seconds", type=float, default=1.0)
    sim.add_argument("--audit-clock-sync-seconds", type=float, default=30.0)
    sim.add_argument("--audit-network-probe-seconds", type=float, default=5.0)
    sim.add_argument(
        "--profile",
        action="store_true",
        help="Enable the low-overhead latency observatory: per-stage scan timing PLUS "
             "tick-to-trade pipeline latency (wire/queue/wakeup/loop-lag/order-rtt), "
             "GC pauses, and a missed-opportunity reason histogram. Writes --profile-dir; "
             "analyze with `python -m src.profile_report --profile-dir <dir>`.",
    )
    sim.add_argument("--profile-dir", type=Path, default=Path("reports/profile"), help="Directory for profiling output (stages.csv, slow_scans.jsonl, counters.json, summary.txt).")
    sim.add_argument("--profile-slow-scan-ms", type=float, default=5.0, help="Dump a full per-stage breakdown for any scan slower than this (ms).")
    sim.add_argument("--profile-dump-seconds", type=float, default=10.0, help="How often to snapshot per-stage percentiles to stages.csv.")
    sim.add_argument("--profile-loop-lag-ms", type=float, default=5.0, help="Event-loop lag monitor sleep interval (ms); smaller = finer starvation detection.")
    sim.add_argument("--profile-net-probe-seconds", type=float, default=15.0, help="Active network/clock-skew probe cadence (s); 0 disables outbound probes.")

    watch = sub.add_parser(
        "build-watchlist",
        help="Build a small executable/liquid watchlist from broad candidates.",
        description=(
            "Read a broad candidate YAML, fetch current books, rank candidates by executable cost, "
            "and write a smaller watchlist for faster monitoring."
        ),
        epilog=WATCHLIST_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    watch.add_argument("--pairs", type=Path, default=Path("config/generated_range_threshold_pairs_may1.yaml"))
    watch.add_argument("--out", type=Path, default=Path("config/clean_watchlist_pairs.yaml"))
    watch.add_argument("--ranking-csv", type=Path, default=Path("reports/clean_watchlist_ranking.csv"))
    watch.add_argument("--top-n", type=int, default=10)
    watch.add_argument("--max-net-cost", type=float, default=1.05)
    watch.add_argument("--min-depth", type=float, default=1.0)
    watch.add_argument("--enabled-only", action="store_true", help="Only consider enabled pairs from the input YAML.")
    watch.add_argument("--entry-threshold", type=float, default=1.0)
    watch.add_argument("--min-edge-threshold", type=float, default=0.0025)
    watch.add_argument("--near-arb-threshold", type=float, default=1.02)
    watch.add_argument("--slippage-buffer", type=float, default=None)
    watch.add_argument("--fee-rate", type=float, default=None)
    watch.add_argument("--max-concurrent-requests", type=int, default=10)
    watch.add_argument("--relation-safety", choices=["all", "clean", "boundary_ambiguous"], default="all")
    watch.add_argument("--min-display-price", type=float, default=None)
    watch.add_argument("--max-threshold-distance-pct", type=float, default=None)
    watch.add_argument("--spot-prices", default="")

    bench = sub.add_parser(
        "benchmark-scan",
        help="Measure how fast a pair file can be scanned with public CLOB books.",
        description=(
            "Fetch public CLOB books for a pair file and report scan timing. "
            "Use this before deciding whether a candidate universe is small enough for hot-loop scanning."
        ),
        epilog=BENCHMARK_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bench.add_argument("--pairs", type=Path, default=Path("config/generated_range_threshold_pairs_may1.yaml"))
    bench.add_argument("--iterations", type=int, default=5)
    bench.add_argument("--enabled-only", action="store_true")
    bench.add_argument("--entry-threshold", type=float, default=1.0)
    bench.add_argument("--min-edge-threshold", type=float, default=0.0025)
    bench.add_argument("--near-arb-threshold", type=float, default=1.02)
    bench.add_argument("--slippage-buffer", type=float, default=None)
    bench.add_argument("--fee-rate", type=float, default=None)
    bench.add_argument("--sizing-mode", choices=["fixed", "max_profit"], default="max_profit")
    bench.add_argument("--optimizer-net-cutoff", type=float, default=1.05)
    bench.add_argument("--max-concurrent-requests", type=int, default=10)

    discover = sub.add_parser(
        "discover-tomorrow-crypto-pairs",
        help="Discover dated crypto range/above candidates from Gamma events.",
        description=(
            "Generate BTC/ETH/SOL/XRP range/threshold candidate YAML from Polymarket Gamma event pages. "
            "Use this before paper-arb-sim when a new daily market opens."
        ),
        epilog=DISCOVER_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    discover.add_argument("--assets", default="BTC,ETH,SOL,XRP")
    discover.add_argument("--days-ahead", type=int, default=1)
    discover.add_argument("--date-slug", default=None, help="Override date slug, e.g. may-1.")
    discover.add_argument("--out", type=Path, default=Path("config/generated_tomorrow_crypto_pairs.yaml"))
    discover.add_argument("--include-boundary-ambiguous", type=_parse_bool, default=True)
    discover.add_argument("--adjacent-only", type=_parse_bool, default=False)
    discover.add_argument("--min-display-price", type=float, default=0.001)
    discover.add_argument("--max-threshold-distance-pct", type=float, default=None)
    discover.add_argument("--spot-prices", default="")

    kalshi_discover = sub.add_parser(
        "discover-kalshi-pairs",
        help="Generate Kalshi same-market YES+NO candidate YAML.",
        description=(
            "Fetch Kalshi public markets and emit candidate YAML that the scanner can consume with "
            "--exchange kalshi. The generated relation is same-market YES+NO complement only; "
            "cross-market relations should be added manually until there is explicit rule parsing."
        ),
        epilog=KALSHI_DISCOVER_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    kalshi_discover.add_argument("--kalshi-base-url", default="https://external-api.kalshi.com/trade-api/v2")
    kalshi_discover.add_argument("--status", default="open")
    kalshi_discover.add_argument("--event-ticker", default=None)
    kalshi_discover.add_argument("--series-ticker", default=None)
    kalshi_discover.add_argument("--tickers", default="", help="Comma-separated Kalshi market tickers.")
    kalshi_discover.add_argument("--search", default="", help="Comma-separated terms that must appear in market metadata.")
    kalshi_discover.add_argument("--limit", type=int, default=100, help="Kalshi page size, max 1000.")
    kalshi_discover.add_argument("--max-pages", type=int, default=5)
    kalshi_discover.add_argument("--max-markets", type=int, default=200)
    kalshi_discover.add_argument("--min-volume", type=float, default=None)
    kalshi_discover.add_argument("--min-open-interest", type=float, default=None)
    kalshi_discover.add_argument(
        "--allow-unquoted",
        action="store_true",
        help="Include markets without both YES and NO quoted asks. Default filters these out.",
    )
    kalshi_discover.add_argument(
        "--skip-orderbook-depth-check",
        action="store_true",
        help="Skip detailed orderbook validation. Faster, but generated candidates may reject as missing ask/depth.",
    )
    kalshi_discover.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=10,
        help="Maximum concurrent Kalshi orderbook checks during discovery.",
    )
    kalshi_discover.add_argument("--enabled", action="store_true", help="Write generated pairs as enabled instead of disabled.")
    kalshi_discover.add_argument("--out", type=Path, default=Path("config/generated_kalshi_pairs.yaml"))

    audit = sub.add_parser(
        "audit-report",
        help="Generate forensic reports from one audit run, optionally comparing another run.",
        description=(
            "Read a forensic audit directory and produce an HTML, Markdown, and/or JSON report. "
            "The report includes opportunity funnels, latency waterfalls, missed-fill timelines, "
            "ambient market activity, and side-by-side run comparison."
        ),
        epilog=AUDIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    audit.add_argument("run_dir", type=Path)
    audit.add_argument("--compare", type=Path, default=None)
    audit.add_argument("--top-missed", type=int, default=25)
    audit.add_argument("--format", default="html,markdown,json", help="Comma-separated formats: html, markdown, json.")

    audit_profit = sub.add_parser(
        "audit-profit-report",
        help="Generate read-only profit leakage and bottleneck forensics from one audit run.",
        description=(
            "Read forensic audit Parquet tables and produce profit-leakage, stale-cache, "
            "duplicate-policy, candidate-noise, scan-efficiency, and improvement-priority reports. "
            "This command is read-only and never places orders."
        ),
        epilog=AUDIT_PROFIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    audit_profit.add_argument("--audit-dir", type=Path, required=True)
    audit_profit.add_argument("--out-dir", type=Path, required=True)
    audit_profit.add_argument("--top-windows", type=int, default=50)
    audit_profit.add_argument("--freshness-ms", default="250,500,1000,2000,5000")
    audit_profit.add_argument("--edge-threshold-grid", default="0,0.001,0.0025,0.005,0.01")
    audit_profit.add_argument("--memory-limit", default="8GB")
    audit_profit.add_argument("--threads", type=int, default=None)

    wc_discover = sub.add_parser(
        "discover-worldcup",
        help="Discover World Cup events/markets and infer their relation graph.",
        description=(
            "Enumerate active Polymarket World Cup events via the Gamma API, normalize every "
            "binary market, group by gameId, infer the relation graph (complement baskets, "
            "subset ladders, quantity-vs-threshold, exact-score subsets), and write a paper-only "
            "subset-arb watchlist YAML plus a relations dump. No orders are placed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    wc_discover.add_argument("--out-dir", default="reports/worldcup")
    wc_discover.add_argument("--overrides", default="config/worldcup_overrides.yaml")
    wc_discover.add_argument("--include-closed", action="store_true")
    wc_discover.add_argument("--no-futures", action="store_true", help="Skip tournament/player futures ladders.")

    wc_scan = sub.add_parser(
        "worldcup-scan",
        help="Scan World Cup relations for executable arbitrage (paper/dry-run/live).",
        description=(
            "Run the World Cup arbitrage scanner: discover -> infer relations -> fetch CLOB books -> "
            "compute depth-aware executable edge -> state-validate worst case -> rank -> execute. "
            "Paper mode is the default; dry-run prints intended orders; live reuses the guarded executor."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    wc_scan.add_argument("--config", default="config/worldcup.yaml")
    wc_scan.add_argument("--execution-mode", choices=["paper", "dry_run", "live"], default=None)
    wc_scan.add_argument("--once", action="store_true", help="Run a single scan and exit.")
    wc_scan.add_argument("--poll-seconds", type=float, default=2.0)
    wc_scan.add_argument("--duration-minutes", type=float, default=None)
    wc_scan.add_argument("--no-execute", action="store_true", help="Evaluate and log only; never enter (pure scan).")
    wc_scan.add_argument("--enable-statistical", action="store_true")
    wc_scan.add_argument("--no-deterministic", action="store_true")
    wc_scan.add_argument("--min-edge-bps", type=float, default=None)
    wc_scan.add_argument("--max-leg-slippage", type=float, default=None)
    wc_scan.add_argument("--fee-rate", type=float, default=None)
    wc_scan.add_argument("--min-liquidity", type=float, default=None)
    wc_scan.add_argument("--max-stale-ms", type=float, default=None)
    wc_scan.add_argument("--max-trade-size", type=float, default=None)
    wc_scan.add_argument("--max-event-exposure", type=float, default=None)
    wc_scan.add_argument("--show-top", type=int, default=None)
    wc_scan.add_argument("--out-dir", default=None)
    wc_scan.add_argument("--live-confirmation", default="")
    wc_scan.add_argument("--live-compliance-ack", default="")
    wc_scan.add_argument("--live-max-session-spend", type=float, default=None)
    wc_scan.add_argument("--live-max-legs-per-bundle", type=int, default=5)
    return parser


async def _main_async() -> None:
    args = build_parser().parse_args()
    if args.command == "paper-arb-sim":
        book_source = args.book_source
        if args.exchange == "kalshi" and book_source in ("websocket", "rust-websocket"):
            print("Kalshi WebSocket mode is not available; using REST polling. Use --book-source fix for real-time FIX feed.")
            book_source = "polling"
        settings = SimulatorSettings(
            pairs_path=args.pairs,
            exchange=args.exchange,
            budget=args.budget,
            duration_minutes=args.duration_minutes,
            poll_seconds=args.poll_seconds,
            near_arb_threshold=args.near_arb_threshold,
            entry_threshold=args.entry_threshold,
            min_edge_threshold=args.min_edge_threshold,
            min_roi_threshold=(
                args.global_min_roi_threshold if args.global_min_roi_threshold is not None else args.min_roi_threshold
            ),
            max_trade_size=args.max_trade_size,
            min_trade_size=args.min_trade_size,
            capital_fraction_per_trade=args.capital_fraction_per_trade,
            sizing_mode=args.sizing_mode,
            optimizer_net_cutoff=args.optimizer_net_cutoff,
            allow_multiple_open_per_pair=args.allow_multiple_open_per_pair,
            cooldown_seconds_per_pair=args.cooldown_seconds_per_pair,
            max_open_positions=args.max_open_positions,
            max_open_positions_per_pair=args.max_open_positions_per_pair,
            max_total_locked_capital=args.max_total_locked_capital,
            include_disabled=args.include_disabled,
            slippage_buffer=args.slippage_buffer,
            fee_rate=args.fee_rate,
            dynamic_fee_rates=args.dynamic_fee_rates,
            max_concurrent_requests=args.max_concurrent_requests,
            kalshi_base_url=args.kalshi_base_url,
            kalshi_fix_api_key=args.kalshi_fix_api_key or os.environ.get("KALSHI_FIX_API_KEY", ""),
            kalshi_fix_api_secret=args.kalshi_fix_api_secret or os.environ.get("KALSHI_FIX_API_SECRET", ""),
            kalshi_fix_host=args.kalshi_fix_host,
            kalshi_fix_port=args.kalshi_fix_port,
            book_source=book_source,
            order_book_cache_ms=args.order_book_cache_ms,
            websocket_url=args.websocket_url,
            websocket_stale_book_ms=args.websocket_stale_book_ms,
            websocket_fallback_cache_ms=args.websocket_fallback_cache_ms,
            websocket_reconnect_seconds=args.websocket_reconnect_seconds,
            fallback_to_polling=args.fallback_to_polling,
            allow_stale_websocket_cache=args.allow_stale_websocket_cache,
            entry_rest_recheck=args.entry_rest_recheck,
            background_entry=args.background_entry,
            near_arb_priority_seed=args.near_arb_priority_seed,
            relation_safety=args.relation_safety,
            allow_boundary_ambiguous_guaranteed=args.allow_boundary_ambiguous_guaranteed,
            min_display_price=args.min_display_price,
            max_threshold_distance_pct=args.max_threshold_distance_pct,
            spot_prices=_parse_spot_prices(args.spot_prices),
            exit_mode=args.exit_mode,
            take_profit_pct=args.take_profit_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_absolute=args.take_profit_absolute,
            stop_loss_absolute=args.stop_loss_absolute,
            out=args.out,
            trades_out=args.trades_out,
            live_orders_out=args.live_orders_out,
            save_markdown=args.save_markdown,
            disable_markdown=args.disable_markdown,
            scan_log_interval_seconds=args.scan_log_interval_seconds,
            clear_screen=args.clear_screen,
            live_dashboard=args.live_dashboard,
            headless=args.headless,
            health_host=args.health_host,
            health_port=(args.health_port or None),  # 0 → None → disabled
            once=args.once,
            show_top=args.show_top,
            dashboard_interval_seconds=args.dashboard_interval_seconds,
            report_interval_seconds=args.report_interval_seconds,
            dynamic_watchlist=args.dynamic_watchlist,
            live_universe=args.live_universe,
            live_universe_assets=args.live_universe_assets,
            live_universe_horizon_days=args.live_universe_horizon_days,
            live_universe_refresh_seconds=args.live_universe_refresh_seconds,
            scan_all_candidates=args.scan_all_candidates,
            execution_mode=args.execution_mode,
            live_confirmation=args.live_confirmation,
            live_compliance_ack=args.live_compliance_ack,
            live_max_session_spend=args.live_max_session_spend,
            live_price_buffer_ticks=args.live_price_buffer_ticks,
            live_max_legs_per_bundle=args.live_max_legs_per_bundle,
            live_signature_type=args.live_signature_type,
            run_lock_path=args.run_lock_path,
            disable_run_lock=args.disable_run_lock,
            enable_n_leg_trading=args.enable_n_leg_trading,
            n_leg_sizing_mode=args.n_leg_sizing_mode,
            n_leg_max_ranges=args.n_leg_max_ranges,
            enable_three_leg_trading=args.enable_n_leg_trading,
            three_leg_sizing_mode=args.n_leg_sizing_mode,
            allow_incremental_n_leg=args.allow_incremental_n_leg,
            incremental_n_leg_min_profit=args.incremental_n_leg_min_profit,
            candidate_pairs_path=args.candidate_pairs,
            watchlist_refresh_seconds=args.watchlist_refresh_seconds,
            watchlist_top_n=args.watchlist_top_n,
            watchlist_max_net_cost=args.watchlist_max_net_cost,
            watchlist_min_depth=args.watchlist_min_depth,
            watchlist_out=args.watchlist_out,
            watchlist_ranking_csv=args.watchlist_ranking_csv,
            audit_mode=args.audit_mode,
            audit_dir=args.audit_dir,
            audit_run_id=args.audit_run_id,
            audit_raw_ws=args.audit_raw_ws,
            audit_flush_seconds=args.audit_flush_seconds,
            audit_clock_sync_seconds=args.audit_clock_sync_seconds,
            audit_network_probe_seconds=args.audit_network_probe_seconds,
            profile=args.profile,
            profile_dir=args.profile_dir,
            profile_slow_scan_ms=args.profile_slow_scan_ms,
            profile_dump_seconds=args.profile_dump_seconds,
            profile_loop_lag_ms=args.profile_loop_lag_ms,
            profile_net_probe_seconds=args.profile_net_probe_seconds,
        )
        if args.continuous_rollover:
            await _run_continuous_rollover(args, settings)
        else:
            await PaperArbSimulator(settings).run()
    elif args.command == "build-watchlist":
        summary = await build_watchlist_once(
            pairs_path=args.pairs,
            out_path=args.out,
            ranking_csv_path=args.ranking_csv,
            top_n=args.top_n,
            max_net_cost=args.max_net_cost,
            min_depth=args.min_depth,
            include_disabled=not args.enabled_only,
            entry_threshold=args.entry_threshold,
            min_edge_threshold=args.min_edge_threshold,
            near_arb_threshold=args.near_arb_threshold,
            slippage_buffer=args.slippage_buffer,
            fee_rate=args.fee_rate,
            max_concurrent_requests=args.max_concurrent_requests,
            relation_safety=args.relation_safety,
            min_display_price=args.min_display_price,
            max_threshold_distance_pct=args.max_threshold_distance_pct,
            spot_prices=_parse_spot_prices(args.spot_prices),
        )
        print("Paper-only watchlist build complete. No live orders were placed.")
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "benchmark-scan":
        summary = await benchmark_scan(
            pairs_path=args.pairs,
            iterations=args.iterations,
            include_disabled=not args.enabled_only,
            entry_threshold=args.entry_threshold,
            min_edge_threshold=args.min_edge_threshold,
            near_arb_threshold=args.near_arb_threshold,
            slippage_buffer=args.slippage_buffer,
            fee_rate=args.fee_rate,
            sizing_mode=args.sizing_mode,
            optimizer_net_cutoff=args.optimizer_net_cutoff,
            max_concurrent_requests=args.max_concurrent_requests,
        )
        print("Paper-only scan benchmark complete. No live orders were placed.")
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "discover-tomorrow-crypto-pairs":
        summary = await discover_crypto_pairs(
            out_path=args.out,
            assets=[asset.strip() for asset in args.assets.split(",") if asset.strip()],
            days_ahead=args.days_ahead,
            date_slug=args.date_slug,
            date_year=None,
            include_boundary_ambiguous=args.include_boundary_ambiguous,
            adjacent_only=args.adjacent_only,
            min_display_price=args.min_display_price,
            spot_prices=_parse_spot_prices(args.spot_prices),
            max_threshold_distance_pct=args.max_threshold_distance_pct,
        )
        print("Paper-only discovery complete. Generated pairs are disabled by default.")
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "discover-kalshi-pairs":
        summary = await discover_kalshi_pairs(
            out_path=args.out,
            base_url=args.kalshi_base_url,
            status=args.status,
            event_ticker=args.event_ticker,
            series_ticker=args.series_ticker,
            tickers=_parse_csv(args.tickers),
            search=args.search,
            limit=args.limit,
            max_pages=args.max_pages,
            max_markets=args.max_markets,
            min_volume=args.min_volume,
            min_open_interest=args.min_open_interest,
            require_quotes=not args.allow_unquoted,
            require_orderbook_depth=not args.skip_orderbook_depth_check,
            max_concurrent_requests=args.max_concurrent_requests,
            enabled=args.enabled,
        )
        print("Kalshi discovery complete. Generated pairs use same-market YES+NO complement logic.")
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "audit-report":
        formats = {item.strip().lower() for item in args.format.split(",") if item.strip()}
        invalid = formats - {"html", "markdown", "json"}
        if invalid:
            raise SystemExit(f"Unknown audit report format(s): {', '.join(sorted(invalid))}")
        summary = generate_audit_report(
            args.run_dir,
            compare_dir=args.compare,
            top_missed=args.top_missed,
            formats=formats,
        )
        report_dir = args.run_dir / "report"
        print(f"Audit report complete for run {summary['run_id']}.")
        if "html" in formats:
            print(f"HTML: {report_dir / 'index.html'}")
        if "markdown" in formats:
            print(f"Markdown: {report_dir / 'summary.md'}")
        if "json" in formats:
            print(f"JSON: {report_dir / 'summary.json'}")
    elif args.command == "audit-profit-report":
        try:
            freshness_ms = parse_float_csv(args.freshness_ms)
            edge_threshold_grid = parse_float_csv(args.edge_threshold_grid)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        summary = generate_audit_profit_report(
            AuditProfitReportConfig(
                audit_dir=args.audit_dir,
                out_dir=args.out_dir,
                top_windows=args.top_windows,
                freshness_ms=freshness_ms,
                edge_threshold_grid=edge_threshold_grid,
                memory_limit=args.memory_limit,
                threads=args.threads,
            )
        )
        print(f"Profit forensics complete for run {summary['run_id']}.")
        print(f"Markdown: {args.out_dir / 'profit_forensics.md'}")
        print(f"JSON: {args.out_dir / 'summary.json'}")
    elif args.command == "discover-worldcup":
        from src.worldcup.cli import run_worldcup_discover
        await run_worldcup_discover(args)
    elif args.command == "worldcup-scan":
        from src.worldcup.cli import run_worldcup_scan
        await run_worldcup_scan(args)


def main() -> None:
    try:
        import uvloop
        uvloop.run(_main_async())
    except ImportError:
        asyncio.run(_main_async())


async def _run_continuous_rollover(args: argparse.Namespace, base_settings: SimulatorSettings) -> None:
    if args.once:
        raise SystemExit("--continuous-rollover cannot be combined with --once.")
    rollover_time = _parse_et_wall_time(args.rollover_time_et)
    delay = timedelta(minutes=max(0.0, args.rollover_delay_minutes))
    assets = [asset.strip() for asset in args.rollover_assets.split(",") if asset.strip()]

    while True:
        now_et = datetime.now(ET_ZONE)
        session_date, switch_at = _active_rollover_session(now_et, rollover_time, delay)
        date_slug = _date_slug_for_date(session_date)
        pairs_path = _format_rollover_path(Path(args.rollover_pairs_template), session_date)
        print(
            f"Continuous rollover: active date={date_slug} until "
            f"{switch_at.isoformat()} ET; generating {pairs_path}"
        )
        summary = await discover_crypto_pairs(
            out_path=pairs_path,
            assets=assets,
            days_ahead=0,
            date_slug=date_slug,
            date_year=session_date.year,
            include_boundary_ambiguous=args.rollover_include_boundary_ambiguous,
            adjacent_only=args.rollover_adjacent_only,
            min_display_price=args.rollover_min_display_price,
            spot_prices=base_settings.spot_prices,
            max_threshold_distance_pct=base_settings.max_threshold_distance_pct,
        )
        if int(summary.get("pairs_written") or 0) <= 0:
            retry_seconds = max(5.0, float(args.rollover_retry_seconds))
            print(
                f"Continuous rollover: no pairs discovered for {date_slug} "
                f"(missing={summary.get('missing_events')}); retrying in {retry_seconds:.0f}s."
            )
            await asyncio.sleep(retry_seconds)
            continue

        now_et = datetime.now(ET_ZONE)
        runtime_minutes = max(0.05, (switch_at - now_et).total_seconds() / 60.0)
        settings = replace(
            base_settings,
            scan_all_candidates=True,
            candidate_pairs_path=pairs_path,
            duration_minutes=runtime_minutes,
            out=_format_rollover_path(base_settings.out, session_date),
            trades_out=_format_rollover_path(base_settings.trades_out, session_date),
            live_orders_out=_format_rollover_path(base_settings.live_orders_out, session_date),
            save_markdown=_format_rollover_path(base_settings.save_markdown, session_date),
            watchlist_out=_format_rollover_path(base_settings.watchlist_out, session_date),
            watchlist_ranking_csv=_format_rollover_path(base_settings.watchlist_ranking_csv, session_date),
        )
        print(
            f"Continuous rollover: running {date_slug} for {runtime_minutes:.2f} minutes "
            f"with {summary.get('pairs_written')} generated pairs."
        )
        await PaperArbSimulator(settings).run()

        now_et = datetime.now(ET_ZONE)
        if now_et < switch_at:
            await asyncio.sleep(max(0.0, (switch_at - now_et).total_seconds()))


def _parse_et_wall_time(value: str) -> datetime_time:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected --rollover-time-et in HH:MM 24-hour format.")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected --rollover-time-et in HH:MM 24-hour format.") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise argparse.ArgumentTypeError("Expected --rollover-time-et in HH:MM 24-hour format.")
    return datetime_time(hour=hour, minute=minute)


def _active_rollover_session(
    now_et: datetime,
    rollover_time: datetime_time,
    delay: timedelta,
) -> tuple[date, datetime]:
    if now_et.tzinfo is None:
        now_et = now_et.replace(tzinfo=ET_ZONE)
    now_et = now_et.astimezone(ET_ZONE)
    today_switch = datetime.combine(now_et.date(), rollover_time, tzinfo=ET_ZONE) + delay
    if now_et < today_switch:
        return now_et.date(), today_switch
    next_date = now_et.date() + timedelta(days=1)
    next_switch = datetime.combine(next_date, rollover_time, tzinfo=ET_ZONE) + delay
    return next_date, next_switch


def _date_slug_for_date(value: date) -> str:
    return f"{value.strftime('%B').lower()}-{value.day}"


def _format_rollover_path(path: Path, session_date: date) -> Path:
    raw = str(path)
    values = {
        "date_slug": _date_slug_for_date(session_date),
        "date_compact": session_date.strftime("%Y%m%d"),
        "date_iso": session_date.isoformat(),
    }
    if "{" in raw and "}" in raw:
        return Path(raw.format(**values))
    return path


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def _parse_spot_prices(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (value or "").split(","):
        if not part.strip() or "=" not in part:
            continue
        key, raw = part.split("=", 1)
        try:
            out[key.strip().upper()] = float(raw.strip())
        except ValueError:
            continue
    return out


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


if __name__ == "__main__":
    main()
