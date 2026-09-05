from __future__ import annotations

import asyncio
import collections
import csv
import io
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any

import yaml

from src.audit import AuditConfig, AuditDependencyError, ForensicAudit
from src import market_profiler, net_profiler, profiling, system_profiler
from src.kalshi.client import KalshiClient, kalshi_token_id, parse_kalshi_token_id
from src.kalshi.fix_client import KalshiFixMarketDataClient
from src.kalshi.live_trader import KalshiLiveTradingConfig, KalshiLiveTradingError, KalshiLiveTrader
from src.fast import (
    scan_classify_batch as _scan_classify_batch,
    n_leg_vwap_score as _n_leg_vwap_score,
    find_best_size_fast as _fast_optimizer,
    SCAN_BACKEND as _SCAN_BACKEND,
    OPTIMIZER_BACKEND as _OPTIMIZER_BACKEND,
    REJECTED as _SCAN_REJECTED,
    NEAR_ARB as _SCAN_NEAR_ARB,
    EXECUTABLE as _SCAN_EXECUTABLE,
)
from src.polymarket.clob_client import ClobClient
from src.polymarket.clob_ws_client import ClobWebSocketClient
from src.polymarket.gamma_client import GammaClient, resolve_binary_token_ids_from_market
from src.polymarket.live_trader import LiveOrderLeg, LiveOrderResult, LiveTrader, LiveTradingConfig, LiveTradingError
from src.polymarket.models import OrderBook
from src.polymarket.order_book_provider import (
    OrderBookProvider,
    PollingOrderBookProvider,
    WebSocketOrderBookProvider,
)
from src.book_snapshot import load_snapshot as _load_book_snapshot
from src.fill_rate_tracker import FillRateTracker
from src.health_server import HealthServer as _HealthServer
from src.resolution_model import pair_days_to_resolution, resolution_size_multiplier

try:  # Rich is nice, but the simulator should still work without it.
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Live = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]

import numpy as _np


def _log_task_exception(task: "asyncio.Task[Any]") -> None:
    """Done-callback: surface exceptions from fire-and-forget background tasks.

    asyncio silently discards task exceptions if the Task object is never
    awaited.  Attach this callback to every ``asyncio.create_task()`` call so
    that programming errors and unexpected failures are always visible in the
    process output.
    """
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Normal shutdown path — not an error.
    except Exception:  # pragma: no cover
        import sys
        import traceback
        print(
            f"[BG TASK ERROR] {task.get_name()}:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)


class _ScanArrays:
    """Pre-allocated numpy arrays for the vectorised scan loop.

    Created once per unique ScanPlan.  Static config arrays (fee rates,
    slippages, etc.) are filled at construction time and never change.
    Per-scan: only parent_asks and child_asks are overwritten, then
    scan_classify_batch writes into gross_out, net_out, class_out.
    """

    __slots__ = (
        "n",
        "parent_asks", "child_asks",
        "fee_rates_1", "fee_rates_2",
        "slippages", "fee_buffers", "min_edges",
        "gross_out", "net_out", "class_out",
        "book_valid",
        "obs_cache",   # list[PairObservation | None] for dirty-set scan
    )

    def __init__(self, n: int) -> None:
        self.n = n
        _f64 = _np.float64
        _i32 = _np.int32
        self.parent_asks = _np.full(n, 2.0, dtype=_f64)
        self.child_asks  = _np.full(n, 2.0, dtype=_f64)
        self.fee_rates_1 = _np.zeros(n, dtype=_f64)
        self.fee_rates_2 = _np.zeros(n, dtype=_f64)
        self.slippages   = _np.zeros(n, dtype=_f64)
        self.fee_buffers = _np.zeros(n, dtype=_f64)
        self.min_edges   = _np.zeros(n, dtype=_f64)
        self.gross_out   = _np.zeros(n, dtype=_f64)
        self.net_out     = _np.zeros(n, dtype=_f64)
        self.class_out   = _np.zeros(n, dtype=_i32)
        self.book_valid  = _np.ones(n, dtype=bool)
        # One cached PairObservation per pair, populated on first full scan.
        # Dirty-set scans reuse this for pairs whose tokens did not update.
        self.obs_cache: list[Any] = [None] * n


class _NLegScanArrays:
    """Pre-allocated numpy arrays for vectorised N-leg opportunity scanning.

    Built once per ScanPlan and reused across scan cycles.  The static parts
    (payout, min_fill) are filled at construction time.  Only the ask and
    depth matrices are overwritten per scan cycle — and only for rows whose
    tokens appear in the dirty-set.

    Layout: rows = N-leg specs, columns = legs (zero-padded to max_legs).
    Missing books → NaN in ask_matrix.
    """

    __slots__ = (
        "n_specs", "max_legs",
        "ask_matrix",     # (n_specs, max_legs) float64, NaN = book missing
        "depth_matrix",   # (n_specs, max_legs) float64
        "payout_vec",     # (n_specs,) float64 — guaranteed_payout, static
        "min_fill_vec",   # (n_specs,) float64 — precomputed, static
        "token_id_rows",  # list[list[str]] — token_id per spec per leg (static)
        "spec_list",      # list[NLegOpportunitySpec] — preserved reference (static)
    )

    def __init__(
        self,
        specs: list["NLegOpportunitySpec"],
        min_trade_size: float,
    ) -> None:
        n = len(specs)
        max_legs = max((len(s.legs) for s in specs), default=3) if specs else 3
        self.n_specs = n
        self.max_legs = max_legs
        self.ask_matrix = _np.full((n, max_legs), _np.nan, dtype=_np.float64)
        self.depth_matrix = _np.zeros((n, max_legs), dtype=_np.float64)
        self.payout_vec = _np.array(
            [s.guaranteed_payout for s in specs], dtype=_np.float64
        )
        self.min_fill_vec = _np.array(
            [min_trade_size / max(1.0, s.guaranteed_payout) for s in specs],
            dtype=_np.float64,
        )
        self.token_id_rows: list[list[str]] = [
            [leg.token_id for leg in s.legs] for s in specs
        ]
        self.spec_list: list[Any] = list(specs)


DEFAULT_TAKER_FEE_RATES = {
    "crypto": 0.072,
    "sports": 0.03,
    "finance": 0.04,
    "politics": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "tech": 0.04,
    "mentions": 0.04,
    "geopolitics": 0.0,
    "kalshi": 0.07,
    "other": 0.05,
}

YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


@dataclass
class PairConfig:
    name: str
    enabled: bool = True
    parent_market_slug: str = ""
    child_market_slug: str = ""
    parent_market_ticker: str = ""
    child_market_ticker: str = ""
    parent_outcome_label: str = ""
    child_outcome_label: str = ""
    parent_yes_token_id: str | None = None
    parent_no_token_id: str | None = None
    child_yes_token_id: str | None = None
    child_no_token_id: str | None = None
    parent_frontend_url: str | None = None
    child_frontend_url: str | None = None
    parent_display_price: float | None = None
    child_display_price: float | None = None
    relation: str = "child_implies_parent"
    relation_subtype: str | None = None
    relation_safety: str = "unknown"
    boundary_ambiguity: bool = False
    boundary_warning: str | None = None
    warnings: list[str] = field(default_factory=list)
    trade_template: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PairConfig":
        return cls(
            name=str(payload.get("name") or "unnamed_pair"),
            enabled=bool(payload.get("enabled", True)),
            parent_market_slug=str(payload.get("parent_market_slug") or ""),
            child_market_slug=str(payload.get("child_market_slug") or ""),
            parent_market_ticker=str(payload.get("parent_market_ticker") or payload.get("parent_kalshi_ticker") or ""),
            child_market_ticker=str(payload.get("child_market_ticker") or payload.get("child_kalshi_ticker") or ""),
            parent_outcome_label=str(payload.get("parent_outcome_label") or ""),
            child_outcome_label=str(payload.get("child_outcome_label") or ""),
            parent_yes_token_id=_optional_str(payload.get("parent_yes_token_id")),
            child_no_token_id=_optional_str(payload.get("child_no_token_id")),
            parent_frontend_url=_optional_str(payload.get("parent_frontend_url")),
            child_frontend_url=_optional_str(payload.get("child_frontend_url")),
            parent_display_price=_optional_float(payload.get("parent_display_price")),
            child_display_price=_optional_float(payload.get("child_display_price")),
            parent_no_token_id=_optional_str(payload.get("parent_no_token_id")),
            child_yes_token_id=_optional_str(payload.get("child_yes_token_id")),
            relation=str(payload.get("relation") or "child_implies_parent"),
            relation_subtype=_optional_str(payload.get("relation_subtype")),
            relation_safety=str(payload.get("relation_safety") or "unknown"),
            boundary_ambiguity=bool(payload.get("boundary_ambiguity", False)),
            boundary_warning=_optional_str(payload.get("boundary_warning")),
            warnings=list(payload.get("warnings") or []),
            trade_template=dict(payload.get("trade_template") or {}),
            overrides=dict(payload.get("overrides") or {}),
            raw=payload,
        )


@dataclass
class PairObservation:
    timestamp: str
    pair_name: str
    parent_outcome_label: str
    child_outcome_label: str
    parent_yes_ask: float | None
    child_no_ask: float | None
    parent_yes_bid: float | None
    child_no_bid: float | None
    gross_total_cost: float | None
    estimated_fee_total_per_unit: float
    slippage_buffer: float
    net_total_cost: float | None
    entry_threshold: float
    distance_to_entry: float | None
    worst_case_profit_per_unit: float | None
    best_case_profit_per_unit: float | None
    max_executable_size: float
    classification: str
    optimal_size: float | None = None
    optimal_required_capital: float | None = None
    optimal_guaranteed_profit: float | None = None
    optimal_net_cost_per_unit: float | None = None
    rejection_reason: str | None = None


@dataclass
class PaperPosition:
    trade_id: int
    entry_time: str
    status: str
    pair_name: str
    relation_subtype: str | None
    entry_trade_type: str
    size: float
    parent_outcome_label: str
    child_outcome_label: str
    parent_yes_entry_price: float
    child_no_entry_price: float
    gross_total_cost_per_unit: float
    net_total_cost_per_unit: float
    locked_capital: float
    worst_case_profit: float
    best_case_profit: float
    exit_time: str | None = None
    exit_parent_yes_bid: float | None = None
    exit_child_no_bid: float | None = None
    exit_total_value: float | None = None
    liquidation_value_gross: float | None = None
    exit_fee_total: float = 0.0
    liquidation_value_net: float | None = None
    liquidation_pnl: float = 0.0
    mtm_value: float | None = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    exit_reason: str | None = None
    hold_minutes: float = 0.0
    event_date: str | None = None


@dataclass(frozen=True)
class NLeg:
    label: str
    token_id: str


@dataclass(frozen=True)
class NLegOpportunitySpec:
    name: str
    legs: tuple[NLeg, ...]
    guaranteed_payout: float
    relation_subtype: str = "n_leg_range_threshold"
    event_date: str | None = None


@dataclass(frozen=True)
class PairScanTarget:
    pair: PairConfig
    leg_token_ids: tuple[str | None, str | None]


@dataclass(frozen=True)
class ScanPlan:
    targets: tuple[PairScanTarget, ...]
    n_leg_specs: tuple[NLegOpportunitySpec, ...]
    token_ids: tuple[str, ...]
    # Reverse index: token_id → indices into `targets` that use that token.
    # Built once at plan construction; used by dirty-set scan to skip re-scoring
    # pairs whose underlying token prices have not changed since the last scan.
    token_to_pair_idxs: dict[str, tuple[int, ...]]
    # Set of token IDs used in any N-leg spec; used to decide whether to
    # re-run _best_n_leg_opportunity when only certain tokens updated.
    n_leg_token_ids: frozenset[str]


@dataclass
class ScanRow:
    timestamp: str
    cash_available: float
    locked_capital: float
    open_positions_count: int
    realized_pnl: float
    unrealized_pnl: float
    liquidation_pnl: float
    guaranteed_profit_if_held: float
    best_case_profit_if_held: float
    best_pair_name: str | None
    best_total_cost: float | None
    net_total_cost: float | None
    entry_threshold: float
    distance_to_entry: float | None
    best_worst_case_profit: float | None
    best_optimal_size: float | None
    best_optimal_guaranteed_profit: float | None
    executable_candidates_count: int
    near_arb_candidates_count: int
    rejected_count: int
    books_missing_count: int
    asks_missing_count: int
    scan_time_ms: float
    book_source: str
    unique_tokens: int
    unique_tokens_fetched: int
    cache_hits: int
    failed_book_count: int
    websocket_connected: bool
    websocket_reconnect_count: int
    fallback_to_polling_used: bool
    token_update_count: int
    event_triggered_recomputes: int
    max_book_age_ms: float | None
    update_latency_ms: float | None
    action_taken: str
    best_n_leg_name: str | None = None
    best_n_leg_leg_count: int | None = None
    best_n_leg_gross_cost: float | None = None
    best_n_leg_guaranteed_payout: float | None = None
    best_n_leg_gross_edge: float | None = None
    best_three_leg_name: str | None = None
    best_three_leg_gross_cost: float | None = None
    best_three_leg_gross_edge: float | None = None


@dataclass
class LiveOrderLogRow:
    timestamp: str
    strategy_name: str
    leg_count: int
    requested_notional: float
    success: bool
    error: str | None
    responses_json: str


@dataclass
class SimulatorSettings:
    pairs_path: Path
    exchange: str = "polymarket"
    budget: float = 100.0
    duration_minutes: float | None = None
    poll_seconds: float = 0.25
    entry_threshold: float = 1.0
    min_edge_threshold: float | None = 0.0025
    min_roi_threshold: float = 0.0
    near_arb_threshold: float = 1.02
    max_trade_size: float = 20.0
    min_trade_size: float = 1.0
    capital_fraction_per_trade: float = 1.0
    sizing_mode: str = "max_profit"
    optimizer_net_cutoff: float = 1.05
    allow_multiple_open_per_pair: bool = False
    cooldown_seconds_per_pair: float = 30.0
    max_open_positions: int | None = None
    max_open_positions_per_pair: int | None = None
    max_total_locked_capital: float | None = None
    include_disabled: bool = False
    slippage_buffer: float | None = None
    fee_rate: float | None = None
    dynamic_fee_rates: bool = True
    max_concurrent_requests: int = 10
    kalshi_base_url: str = "https://external-api.kalshi.com/trade-api/v2"
    kalshi_fix_api_key: str = ""
    kalshi_fix_api_secret: str = ""
    kalshi_fix_host: str = "mm.fix.elections.kalshi.com"
    kalshi_fix_port: int = 8233
    book_source: str = "websocket"
    order_book_cache_ms: int = 0
    websocket_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    websocket_stale_book_ms: int = 2000
    websocket_fallback_cache_ms: int = 10000
    websocket_reconnect_seconds: float = 5.0
    fallback_to_polling: bool = True
    allow_stale_websocket_cache: bool = False
    entry_rest_recheck: bool = True
    # When True, N-leg entry REST rechecks are spawned as background asyncio tasks
    # so the scan loop is never blocked by a 50–200 ms REST round-trip.  Capital is
    # pre-reserved (up to max_trade_size) before the first await so two concurrent
    # tasks cannot double-spend.  Action strings are reported on the *next* scan row.
    # Default False keeps the original sequential behaviour so unit tests that call
    # _scan_once() directly still pass without an extra event-loop flush.
    background_entry: bool = False
    # When True, tokens belonging to near-arb pairs (net_total_cost ≤
    # near_arb_threshold) are immediately REST-seeded via priority_seed() so that
    # stale WS prices are cleared faster than the normal fallback_cache_ms window.
    # Seeding is throttled to at most once per near_arb_seed_interval_seconds and
    # limited to the top-K closest candidates to avoid excessive REST load.
    near_arb_priority_seed: bool = True
    near_arb_seed_interval_seconds: float = 2.0
    near_arb_seed_top_k: int = 5
    relation_safety: str = "all"
    allow_boundary_ambiguous_guaranteed: bool = False
    min_display_price: float | None = None
    max_threshold_distance_pct: float | None = None
    spot_prices: dict[str, float] = field(default_factory=dict)
    exit_mode: str = "hold_until_resolution"
    take_profit_pct: float = 0.0025
    stop_loss_pct: float = 0.005
    take_profit_absolute: float | None = None
    stop_loss_absolute: float | None = None
    out: Path = Path("reports/paper_arb_sim_clean.csv")
    trades_out: Path = Path("reports/paper_arb_trades_clean.csv")
    live_orders_out: Path = Path("reports/live_orders.csv")
    save_markdown: Path = Path("reports/paper_arb_sim_clean.md")
    disable_markdown: bool = False
    scan_log_interval_seconds: float = 0.0
    clear_screen: bool = False
    live_dashboard: bool = True
    # Headless daemon mode: no console output at all (no startup banner, live
    # dashboard, per-scan CSV, markdown, or profiler summary).  The bot just
    # trades and emits telemetry to Prometheus /metrics on --health-port.  Use
    # with --profile so latency series populate.  Minimal per-scan overhead.
    headless: bool = False
    once: bool = False
    show_top: int = 3
    dashboard_interval_seconds: float = 0.0
    report_interval_seconds: float = 5.0
    dynamic_watchlist: bool = False
    scan_all_candidates: bool = False
    candidate_pairs_path: Path | None = None
    live_universe: bool = False
    live_universe_assets: str = "BTC,ETH,SOL,XRP"
    live_universe_horizon_days: int = 14
    live_universe_refresh_seconds: float = 900.0
    execution_mode: str = "paper"
    live_confirmation: str = ""
    live_compliance_ack: str = ""
    live_max_session_spend: float | None = None
    live_price_buffer_ticks: int = 0
    live_max_legs_per_bundle: int = 5
    live_signature_type: int | None = None
    run_lock_path: Path | None = None
    disable_run_lock: bool = False
    enable_n_leg_trading: bool = False
    n_leg_sizing_mode: str = "optimized"
    n_leg_max_ranges: int | None = 1
    enable_three_leg_trading: bool = False
    three_leg_sizing_mode: str = "optimized"
    # Incremental N-leg: allow a second entry on the same opportunity when new
    # depth becomes available after an existing open position.
    allow_incremental_n_leg: bool = False
    # Minimum expected dollar profit for an incremental N-leg entry.
    incremental_n_leg_min_profit: float = 0.50
    watchlist_refresh_seconds: float = 60.0
    watchlist_top_n: int = 10
    watchlist_max_net_cost: float = 1.05
    watchlist_min_depth: float = 1.0
    watchlist_out: Path = Path("config/clean_dynamic_watchlist_pairs.yaml")
    watchlist_ranking_csv: Path = Path("reports/clean_dynamic_watchlist_ranking.csv")
    audit_mode: str = "off"
    audit_dir: Path = Path("reports/audit")
    audit_run_id: str | None = None
    audit_raw_ws: bool = True
    audit_flush_seconds: float = 1.0
    audit_clock_sync_seconds: float = 30.0
    audit_network_probe_seconds: float = 5.0
    # ── Fill-rate tracking ────────────────────────────────────────────────────
    # Track per-pair fill rates using an EMA and scale the minimum edge
    # requirement by inverse confidence.  Pairs that historically fail to fill
    # (REST recheck no longer valid, depth gone) require a proportionally
    # higher raw edge before an entry is attempted.
    fill_rate_tracking: bool = True
    fill_rate_ema_alpha: float = 0.15      # EMA decay weight per observation
    fill_rate_min_observations: int = 3    # Obs before confidence deviates from 1.0
    fill_rate_confidence_floor: float = 0.2  # Min confidence (caps edge penalty at 5×)
    # ── Time-to-resolution sizing ─────────────────────────────────────────────
    # Scale position size by sqrt(target_days / days_to_resolution) so that
    # markets resolving soon receive more capital per unit of edge (faster
    # capital recycling, better annualised return).  Requires end_date / end_time
    # in the pair YAML or API payload; silently skips if not found.
    resolution_time_sizing: bool = True
    resolution_target_days: float = 30.0      # Baseline: 30-day market = multiplier 1.0
    resolution_max_multiplier: float = 2.0    # Cap for very short-dated markets
    resolution_min_multiplier: float = 0.3    # Floor for very long-dated markets
    # ── Book state persistence ────────────────────────────────────────────────
    # Periodically snapshot WS order-book state so that on restart the scanner
    # has warm books immediately rather than a ~10 s cold-start window.  The
    # snapshot is accepted only if it is younger than book_snapshot_max_age_s.
    book_snapshot_path: Path | None = Path(".book_snapshot.json")
    book_snapshot_interval_s: float = 60.0   # How often to save (seconds)
    book_snapshot_max_age_s: float = 120.0   # Reject snapshot older than this
    # ── HTTP health endpoint ──────────────────────────────────────────────────
    # Lightweight JSON health check at GET /health.  Set health_port to None
    # to disable.  By default binds to localhost only (127.0.0.1) so it is not
    # accessible from outside the machine without explicit network exposure.
    health_host: str = "127.0.0.1"
    health_port: int | None = 8765
    # ── Profiling (MVP) ───────────────────────────────────────────────────────
    # Low-overhead per-stage scan latency profiler.  Off by default; when on,
    # writes reports/profile/{stages.csv,slow_scans.jsonl,summary.txt}.  Gated so
    # a non-profiled run pays a single bool check.  See src/profiling.py.
    profile: bool = False
    profile_dir: Path = Path("reports/profile")
    profile_slow_scan_ms: float = 5.0
    profile_dump_seconds: float = 10.0
    # Event-loop lag monitor sleep interval (ms).  Smaller = finer-grained
    # starvation detection at slightly higher sample rate; 5 ms ≈ 200 Hz.
    profile_loop_lag_ms: float = 5.0
    # Active network/clock-skew probe cadence (s).  0 disables the outbound probes
    # (e.g. on a metered link); passive wire_latency still works.
    profile_net_probe_seconds: float = 15.0


class PaperArbSimulator:
    """Fast paper-only simulator.

    The hot loop can use either REST polling or the public CLOB WebSocket cache.
    There is intentionally no credential handling and no order-placement code.
    """

    def __init__(self, settings: SimulatorSettings) -> None:
        self.settings = settings
        self.console = Console() if Console else None
        self.cash = settings.budget
        self.positions: list[PaperPosition] = []
        # Bounded deque — prevents unbounded memory growth during multi-hour runs.
        # At ~20 scans/s a 72 000-entry window covers ≈ 1 hour of history.
        # _total_scans tracks the true lifetime count for the markdown summary.
        self.scan_rows: collections.deque[ScanRow] = collections.deque(maxlen=72_000)
        self._total_scans: int = 0
        self.missed: list[dict[str, Any]] = []
        self.last_trade_time_by_pair: dict[str, float] = {}
        self.open_pair_names: set[str] = set()
        self.open_count_by_pair: dict[str, int] = {}
        self._next_trade_id = 1
        self._latest_top_lines: list[str] = []
        self._open_n_leg_names: set[str] = set()
        self._open_three_leg_names = self._open_n_leg_names
        self._live: Any = None
        self._dashboard_line_count = 0
        self._clob_client: Any | None = None
        self._pair_fee_rate_cache: dict[str, tuple[float, float]] = {}
        self._pair_slippage_cache: dict[str, float] = {}
        self._token_fee_rate_cache: dict[str, float] = {}
        self._scan_plan_cache: ScanPlan | None = None
        self._live_trader: LiveTrader | None = None
        self._live_presign_queue: list[tuple[str, float, float, str | None, bool | None]] = []
        # Per-pair config cache: fee_buffer, max_spread, min_edge_threshold
        self._pair_cfg_cache: dict[str, tuple[float, float | None, float]] = {}
        self._scan_plan_cache_pairs: list[Any] | None = None  # identity-based invalidation
        self._scan_arrays: _ScanArrays | None = None  # pre-allocated C scan buffers
        self._audit = ForensicAudit(
            AuditConfig(
                mode=settings.audit_mode,
                audit_dir=settings.audit_dir,
                run_id=settings.audit_run_id,
                raw_ws=settings.audit_raw_ws,
                flush_seconds=settings.audit_flush_seconds,
                clock_sync_seconds=settings.audit_clock_sync_seconds,
                network_probe_seconds=settings.audit_network_probe_seconds,
            )
        )
        self._audit_windows: dict[str, dict[str, Any]] = {}
        # ── Concurrency: background entry ────────────────────────────────────
        # Maps candidate name → estimated reserved capital for in-flight
        # background entry tasks, so _effective_cash() can exclude it.
        self._inflight_entries: dict[str, float] = {}
        # Completed background entry action strings, drained at the next scan.
        self._pending_bg_actions: list[str] = []
        # Alpha-decay tracking: pair_name → perf_counter_ns when it first became a
        # near-arb.  When a pair leaves the near set we record how long it lived
        # (opportunity_lifetime) — if that's shorter than our tick-to-trade, we can
        # never capture it.  Set/cleared each scan; cheap dict ops over a small set.
        self._near_since_ns: dict[str, int] = {}
        # Active provider, set in _run_unlocked so background tasks can call
        # priority_seed() without needing a provider reference passed around.
        self._provider: OrderBookProvider | None = None
        # Monotonic timestamp of the last near-arb priority seed fire.
        self._last_priority_seed_at: float = 0.0
        # Cached result of _best_n_leg_opportunity; reused when no N-leg tokens updated.
        self._cached_best_n_leg: dict[str, Any] = {}
        # Speculative N-leg REST pre-fetch pipeline.
        # Immediately after each scan the main loop fires an asyncio.Task that
        # fetches fresh REST books for the current best N-leg candidate.  Because
        # httpx is non-blocking, the HTTP round-trip (~40 ms) runs concurrently
        # with wait_for_updates (~50 ms WS idle window) and finishes before the
        # next scan starts.  The next scan's bg_enter_n_leg task then finds
        # pre-fetched books ready and needs no REST call of its own, cutting
        # background-entry latency from ~40 ms to ~1 ms.
        self._n_leg_speculative_prefetch: asyncio.Task | None = None
        self._n_leg_speculative_name: str = ""
        # Speculative pair REST pre-fetch pipeline.
        # Near-arb pair tokens from each scan are pre-fetched during the WS
        # wait so that when pairs become executable in the next scan, REST
        # books are already in-memory and bg_enter_pair needs no extra REST.
        self._pair_speculative_prefetch: asyncio.Task | None = None
        # Tokens fetched by the current speculative task (for coverage check).
        self._pair_speculative_token_set: frozenset[str] = frozenset()
        # Near-arb pair tokens from the last scan; consumed by the main loop
        # to decide which tokens to pre-fetch during the next WS wait.
        self._hot_pair_tokens: list[str] = []
        # ── Priority-2 components ─────────────────────────────────────────────
        # Fill-rate EMA tracker — records market-driven fill outcomes per pair.
        self._fill_tracker = FillRateTracker(
            ema_alpha=settings.fill_rate_ema_alpha,
            min_observations=settings.fill_rate_min_observations,
            confidence_floor=settings.fill_rate_confidence_floor,
        )
        # Pre-allocated numpy arrays for vectorised N-leg scanning.
        # Built in _scan_plan_for alongside _scan_arrays; None until first plan.
        self._n_leg_arrays: _NLegScanArrays | None = None

    async def run(self) -> None:
        with _RunLock(self._effective_run_lock_path()):
            await self._run_unlocked()

    async def _run_unlocked(self) -> None:
        pairs = await self._load_initial_pairs()
        if not pairs:
            raise SystemExit("No usable pairs found. Need parent_yes_token_id and child_no_token_id, or resolvable market slugs.")

        _ensure_parent(self.settings.out)
        _ensure_parent(self.settings.trades_out)
        if self._live_mode:
            _ensure_parent(self.settings.live_orders_out)
        if not self.settings.disable_markdown:
            _ensure_parent(self.settings.save_markdown)
        _write_csv_header(self.settings.out, ScanRow)
        _write_csv_header(self.settings.trades_out, PaperPosition)
        if self._live_mode:
            _write_csv_header(self.settings.live_orders_out, LiveOrderLogRow)

        self._print_startup(pairs)
        run_status = "completed"
        run_error: BaseException | None = None
        audit_started = False
        try:
            try:
                self._audit.start(settings=self.settings)
                audit_started = self._audit.enabled
            except AuditDependencyError as exc:
                raise SystemExit(str(exc)) from exc
            async with self._make_market_data_client() as clob:
                self._clob_client = clob
                # Pre-warm TLS session so the first real request has no handshake overhead
                if hasattr(clob, "warmup"):
                    await clob.warmup()
                self._audit.start_background_tasks(clob)
                self._initialize_live_trader()
                await self._prepare_pair_caches(pairs, clob)
                provider = self._make_provider(clob)
                self._provider = provider
                # Load book snapshot before WS connects so books are warm immediately.
                _snap_path = self.settings.book_snapshot_path
                if _snap_path is not None:
                    _snap_books = _load_book_snapshot(
                        _snap_path,
                        max_age_seconds=self.settings.book_snapshot_max_age_s,
                    )
                    if _snap_books:
                        ws_client_ = getattr(provider, "ws_client", None)
                        if ws_client_ is not None and hasattr(ws_client_, "seed_books"):
                            ws_client_.seed_books(_snap_books)
                await provider.start()
                # Start periodic snapshot saver (fire-and-forget; exceptions logged).
                if _snap_path is not None:
                    _snap_task = asyncio.create_task(self._bg_save_book_snapshot())
                    _snap_task.add_done_callback(_log_task_exception)
                # Start HTTP health endpoint (best-effort; failure does not abort run).
                _health_server: _HealthServer | None = None
                if self.settings.health_port is not None:
                    try:
                        _health_server = _HealthServer(
                            self,
                            host=self.settings.health_host,
                            port=self.settings.health_port,
                        )
                        await _health_server.start()
                    except Exception as _hs_exc:
                        import sys as _sys
                        print(f"[WARN] Health server failed to start: {_hs_exc}", file=_sys.stderr)
                        _health_server = None
                started = time.monotonic()
                next_watchlist_refresh = started + self.settings.watchlist_refresh_seconds
                next_live_universe_refresh = started + self.settings.live_universe_refresh_seconds
                next_dashboard_at = started
                next_report_at = started
                next_scan_log_at = started
                self._start_live_dashboard()
                _is_ws = self.settings.book_source in ("websocket", "fix", "rust-websocket")
                _updated_tokens: set[str] | None = None  # None → full scan on first iteration
                # ── Profiling harness — background dumper + loop-lag + OS + network samplers ──
                _profile_task: asyncio.Task | None = None
                _loop_lag_task: asyncio.Task | None = None
                _bg_profiler_tasks: list[asyncio.Task] = []
                if self.settings.profile:
                    profiling.setup(
                        self.settings.profile_dir,
                        slow_scan_ms=self.settings.profile_slow_scan_ms,
                    )
                    _profile_task = asyncio.create_task(self._bg_profile_dump())
                    _profile_task.add_done_callback(_log_task_exception)
                    # Independent task that measures how late the event loop wakes a
                    # timer — i.e. how starved the loop is.  Runs on the SAME loop it
                    # measures (that is the point), at ~1/profile_loop_lag_ms kHz.
                    _loop_lag_task = asyncio.create_task(
                        profiling.loop_lag_monitor(
                            max(0.001, self.settings.profile_loop_lag_ms / 1000.0)
                        )
                    )
                    _loop_lag_task.add_done_callback(_log_task_exception)
                    # OS/process resource sampler (CPU, RSS, ctx-switches, GC) — 1 Hz.
                    _bg_profiler_tasks.append(asyncio.create_task(system_profiler.sampler(1.0)))
                    # Active network + clock-skew probes — slow cadence, fully off-path.
                    if self.settings.profile_net_probe_seconds > 0:
                        _bg_profiler_tasks.append(
                            asyncio.create_task(net_profiler.prober(self.settings.profile_net_probe_seconds))
                        )
                    for _bt in _bg_profiler_tasks:
                        _bt.add_done_callback(_log_task_exception)
                _prev_scan_ns = 0  # for scan_interval (cadence) measurement
                try:
                    while True:
                        if self.settings.dynamic_watchlist and time.monotonic() >= next_watchlist_refresh:
                            pairs = await self._refresh_dynamic_watchlist()
                            await self._prepare_pair_caches(pairs, clob)
                            next_watchlist_refresh = time.monotonic() + self.settings.watchlist_refresh_seconds
                            _updated_tokens = None  # watchlist changed → force full scan
                        # Live universe (N-day) refresh — periodically rebuild the pair
                        # set from all in-window markets.  Like the watchlist refresh,
                        # a changed universe forces a full scan so the dirty-set logic
                        # re-seeds against the new token set.
                        if self.settings.live_universe and time.monotonic() >= next_live_universe_refresh:
                            pairs = await self._refresh_live_universe()
                            await self._prepare_pair_caches(pairs, clob)
                            next_live_universe_refresh = time.monotonic() + self.settings.live_universe_refresh_seconds
                            _updated_tokens = None  # universe changed → force full scan
                        # ── Scan cadence + wakeup classification ──────────────────────
                        # scan_interval = wall gap between consecutive scan starts.  In
                        # WS mode this is bimodal: short when riding a burst of updates,
                        # ~poll_seconds when idle.  Counting event vs. empty wakeups lets
                        # the report say what fraction of scans were actually driven by
                        # market data (the rest are housekeeping ticks).
                        if profiling.ENABLED and (_rec := profiling.RECORDER) is not None:
                            _scan_start_ns = time.perf_counter_ns()
                            if _prev_scan_ns:
                                _rec.record_series("scan_interval", _scan_start_ns - _prev_scan_ns)
                            _prev_scan_ns = _scan_start_ns
                            _rec.incr("scans")
                            if _is_ws:
                                if _updated_tokens:
                                    _rec.incr("ws_event_wakeups")
                                elif _updated_tokens is not None:
                                    _rec.incr("empty_wakeups")  # WS timeout, no updates
                        row = await self._scan_once(
                            pairs, provider,
                            updated_tokens=_updated_tokens if _is_ws else None,
                        )
                        self.scan_rows.append(row)
                        self._total_scans += 1
                        now_monotonic = time.monotonic()
                        # Headless: skip ALL console/disk reporting (scan CSV, live
                        # dashboard, markdown).  Telemetry goes to Prometheus /metrics
                        # instead; this removes the per-scan dashboard render and IO
                        # that otherwise run every cycle (the bulk of non-scan cost).
                        if not self.settings.headless:
                            if now_monotonic >= next_scan_log_at:
                                _append_csv_row(self.settings.out, row)
                                next_scan_log_at = now_monotonic + max(0.0, self.settings.scan_log_interval_seconds)
                            if now_monotonic >= next_dashboard_at:
                                self._render_dashboard(row)
                                next_dashboard_at = now_monotonic + max(0.0, self.settings.dashboard_interval_seconds)
                        if not self.settings.disable_markdown and not self.settings.headless and now_monotonic >= next_report_at:
                            _md_task = asyncio.create_task(asyncio.to_thread(self._write_markdown))
                            _md_task.add_done_callback(_log_task_exception)
                            next_report_at = now_monotonic + max(0.1, self.settings.report_interval_seconds)

                        if self.settings.once:
                            break
                        elapsed_minutes = (time.monotonic() - started) / 60
                        if self.settings.duration_minutes is not None and elapsed_minutes >= self.settings.duration_minutes:
                            break
                        # Fire background pre-signing for live mode before entering idle wait.
                        if self._live_trader is not None and self._live_presign_queue:
                            _presign_task = asyncio.create_task(
                                self._live_trader.prefetch_for_candidates(self._live_presign_queue)
                            )
                            _presign_task.add_done_callback(_log_task_exception)
                        # ── Speculative N-leg REST pre-fetch ──────────────────────────────
                        # Fire a REST fetch for the current best N-leg's tokens RIGHT NOW
                        # so it runs concurrently with wait_for_updates (~50 ms WS idle).
                        # asyncio httpx is non-blocking: the HTTP request is dispatched
                        # immediately and the response arrives while the event loop is
                        # blocked in wait_for_updates — no extra wall-clock time spent.
                        # By the time the next scan spawns bg_enter_n_leg, the task is
                        # done and _consume_n_leg_speculative_books() returns fresh REST
                        # books; the background task needs no additional REST call.
                        #
                        # Rules:
                        #  - Only when WS is active (REST mode has no idle window to hide in)
                        #  - Only when entry_rest_recheck is on (otherwise recheck is skipped)
                        #  - Only when N-leg trading is enabled
                        #  - Skip if the best N-leg is already inflight (no entry will happen)
                        #  - Re-use a running task for the SAME N-leg; cancel+restart only
                        #    when the best candidate changed (different token set needed)
                        if (
                            _is_ws
                            and self.settings.entry_rest_recheck
                            and self._clob_client is not None
                            and self._n_leg_trading_enabled()
                        ):
                            _best_nl = self._cached_best_n_leg
                            _nl_name = str(_best_nl.get("name") or "") if _best_nl else ""
                            if _best_nl and _nl_name and _nl_name not in self._inflight_entries:
                                _nl_tokens = [str(t) for t in (_best_nl.get("leg_token_ids") or []) if t]
                                if _nl_tokens:
                                    _sp_cur = self._n_leg_speculative_prefetch
                                    _sp_stale = self._n_leg_speculative_name != _nl_name
                                    if _sp_cur is None or _sp_cur.done() or _sp_stale:
                                        if _sp_cur is not None and not _sp_cur.done() and _sp_stale:
                                            _sp_cur.cancel()
                                        _sp_new = asyncio.create_task(
                                            self._clob_client.get_order_books(
                                                _nl_tokens,
                                                max_concurrent_requests=self.settings.max_concurrent_requests,
                                            )
                                        )
                                        # Silence "exception was never retrieved" for
                                        # this best-effort task: _consume_n_leg_speculative_books
                                        # handles errors silently; bg_enter_n_leg falls back
                                        # to its own REST call if the prefetch failed.
                                        _sp_new.add_done_callback(
                                            lambda t: t.exception() if not t.cancelled() else None
                                        )
                                        self._n_leg_speculative_prefetch = _sp_new
                                        self._n_leg_speculative_name = _nl_name
                        # ── Speculative pair REST pre-fetch ───────────────────────────────
                        # Mirrors the N-leg pipeline above.  Near-arb pair tokens from this
                        # scan are fetched in the background so that when those pairs become
                        # executable in the next scan, bg_enter_pair finds REST books ready
                        # and needs no additional REST round-trip.
                        # Token-set change detection: if _hot_pair_tokens differs from the
                        # currently-fetching task's token set, cancel and restart (new
                        # candidates have appeared).  If unchanged, re-use the running task.
                        if (
                            _is_ws
                            and self.settings.entry_rest_recheck
                            and self._clob_client is not None
                            and self._hot_pair_tokens
                        ):
                            _hot_set = frozenset(self._hot_pair_tokens)
                            _p_cur = self._pair_speculative_prefetch
                            _p_stale = _hot_set != self._pair_speculative_token_set
                            if _p_cur is None or _p_cur.done() or _p_stale:
                                if _p_cur is not None and not _p_cur.done() and _p_stale:
                                    _p_cur.cancel()
                                _p_new = asyncio.create_task(
                                    self._clob_client.get_order_books(
                                        self._hot_pair_tokens,
                                        max_concurrent_requests=self.settings.max_concurrent_requests,
                                    )
                                )
                                _p_new.add_done_callback(
                                    lambda t: t.exception() if not t.cancelled() else None
                                )
                                self._pair_speculative_prefetch = _p_new
                                self._pair_speculative_token_set = _hot_set
                        if _is_ws and hasattr(provider, "wait_for_updates"):
                            # Capture updated tokens for the NEXT scan's dirty-set logic.
                            _updated_tokens = await provider.wait_for_updates(  # type: ignore[attr-defined]
                                timeout=max(0.01, self.settings.poll_seconds)
                            )
                        else:
                            await asyncio.sleep(max(0.01, self.settings.poll_seconds))
                            _updated_tokens = None  # REST mode: always full scan
                finally:
                    self._stop_live_dashboard()
                    if _profile_task is not None:
                        # Disarm first so loop_lag_monitor's `while ENABLED` exits and
                        # no further samples are recorded during shutdown.
                        profiling.teardown()
                        for _t in (_profile_task, _loop_lag_task, *_bg_profiler_tasks):
                            if _t is None:
                                continue
                            _t.cancel()
                            try:
                                await _t
                            except asyncio.CancelledError:
                                pass
                        if profiling.RECORDER is not None:
                            profiling.RECORDER.write_summary(quiet=self.settings.headless)
                    if _health_server is not None:
                        await _health_server.stop()
                    self._provider = None
                    await provider.stop()
                    await self._audit.stop_background_tasks()
                    self._clob_client = None
        except BaseException as exc:
            run_status = "crashed"
            run_error = exc
            raise
        finally:
            if audit_started:
                self._audit_finalize_open_windows()
                self._audit.close(status=run_status, error=run_error, settings=self.settings)

        if not self.settings.disable_markdown and not self.settings.headless:
            self._write_markdown()

    def _make_market_data_client(self) -> Any:
        if self.settings.exchange == "kalshi":
            return KalshiClient(base_url=self.settings.kalshi_base_url, audit=self._audit)
        return ClobClient(audit=self._audit)

    def _make_provider(self, clob: Any) -> OrderBookProvider:
        polling = PollingOrderBookProvider(
            clob,
            max_concurrent_requests=self.settings.max_concurrent_requests,
            order_book_cache_ms=self.settings.order_book_cache_ms,
            stale_book_ms=self.settings.websocket_stale_book_ms,
        )
        if self.settings.exchange == "kalshi":
            if self.settings.book_source == "fix":
                if not self.settings.kalshi_fix_api_key or not self.settings.kalshi_fix_api_secret:
                    raise SystemExit(
                        "Kalshi FIX mode requires --kalshi-fix-api-key and --kalshi-fix-api-secret "
                        "(or KALSHI_FIX_API_KEY / KALSHI_FIX_API_SECRET env vars)."
                    )
                fix_client = KalshiFixMarketDataClient(
                    api_key=self.settings.kalshi_fix_api_key,
                    api_secret=self.settings.kalshi_fix_api_secret,
                    host=self.settings.kalshi_fix_host,
                    port=self.settings.kalshi_fix_port,
                    stale_book_ms=self.settings.websocket_stale_book_ms,
                    reconnect_delay_seconds=self.settings.websocket_reconnect_seconds,
                )
                return WebSocketOrderBookProvider(
                    fix_client,
                    polling_provider=polling,
                    stale_book_ms=self.settings.websocket_stale_book_ms,
                    fallback_to_polling=True,
                    allow_stale_cache=False,
                    fallback_cache_ms=self.settings.websocket_fallback_cache_ms,
                )
            if self.settings.book_source == "websocket":
                raise SystemExit("Kalshi WebSocket mode is not available; use --book-source fix or --book-source polling.")
            return polling
        if self.settings.book_source in ("websocket", "rust-websocket"):
            ws_client: Any
            if self.settings.book_source == "rust-websocket":
                try:
                    from src.polymarket.rust_clob_ws_client import RustClobWsClient  # noqa: PLC0415
                    ws_client = RustClobWsClient(
                        self.settings.websocket_url,
                        reconnect_delay_seconds=self.settings.websocket_reconnect_seconds,
                        stale_book_ms=self.settings.websocket_stale_book_ms,
                    )
                    import sys as _sys
                    print(
                        "[INFO] Using Rust WebSocket client (polymarket_rs) — "
                        "~10 μs WS→book latency",
                        file=_sys.stderr,
                        flush=True,
                    )
                except ImportError:
                    import sys as _sys
                    print(
                        "[WARN] polymarket_rs not found — build it with "
                        "`cd rust_ws_client && maturin build --release && pip install ...`. "
                        "Falling back to Python WS client.",
                        file=_sys.stderr,
                        flush=True,
                    )
                    ws_client = ClobWebSocketClient(
                        self.settings.websocket_url,
                        reconnect_delay_seconds=self.settings.websocket_reconnect_seconds,
                        stale_book_ms=self.settings.websocket_stale_book_ms,
                        audit=self._audit,
                    )
            else:
                ws_client = ClobWebSocketClient(
                    self.settings.websocket_url,
                    reconnect_delay_seconds=self.settings.websocket_reconnect_seconds,
                    stale_book_ms=self.settings.websocket_stale_book_ms,
                    audit=self._audit,
                )
            return WebSocketOrderBookProvider(
                ws_client,
                polling_provider=polling,
                stale_book_ms=self.settings.websocket_stale_book_ms,
                fallback_to_polling=self.settings.fallback_to_polling,
                allow_stale_cache=self.settings.allow_stale_websocket_cache,
                fallback_cache_ms=self.settings.websocket_fallback_cache_ms,
            )
        return polling

    @property
    def _live_mode(self) -> bool:
        return self.settings.execution_mode == "live"

    def _effective_run_lock_path(self) -> Path | None:
        if self.settings.disable_run_lock:
            return None
        if self.settings.run_lock_path is not None:
            return self.settings.run_lock_path
        if self._live_mode:
            return Path("data/live_trading.lock")
        return None

    def _initialize_live_trader(self) -> None:
        if not self._live_mode or self._live_trader is not None:
            return
        if self.settings.exchange == "kalshi":
            self._initialize_kalshi_live_trader()
            return
        if self.settings.exchange != "polymarket":
            raise SystemExit(f"Live trading is not yet implemented for --exchange {self.settings.exchange}.")
        max_session_spend = self.settings.live_max_session_spend
        if max_session_spend is None:
            raise SystemExit("Live mode requires --live-max-session-spend.")
        try:
            config = LiveTradingConfig.from_env(
                host="https://clob.polymarket.com",
                chain_id=137,
                signature_type=self.settings.live_signature_type,
                max_session_spend=max_session_spend,
                max_legs_per_bundle=self.settings.live_max_legs_per_bundle,
                confirmation=self.settings.live_confirmation,
                compliance_ack=self.settings.live_compliance_ack,
            )
            self._live_trader = LiveTrader(config)
        except LiveTradingError as exc:
            raise SystemExit(str(exc)) from exc

    def _initialize_kalshi_live_trader(self) -> None:
        max_session_spend = self.settings.live_max_session_spend
        if max_session_spend is None:
            raise SystemExit("Kalshi live mode requires --live-max-session-spend.")
        if self.settings.book_source != "fix":
            raise SystemExit("Kalshi live mode requires --book-source fix (FIX Order Entry is on port 8228).")
        try:
            config = KalshiLiveTradingConfig.from_env(
                confirmation=self.settings.live_confirmation,
                compliance_ack=self.settings.live_compliance_ack,
                max_session_spend=max_session_spend,
                max_legs_per_bundle=self.settings.live_max_legs_per_bundle,
                host=self.settings.kalshi_fix_host,
                port=8228,
            )
            self._live_trader = KalshiLiveTrader(config)  # type: ignore[assignment]
        except KalshiLiveTradingError as exc:
            raise SystemExit(str(exc)) from exc

    async def _load_initial_pairs(self) -> list[PairConfig]:
        if self.settings.dynamic_watchlist and self.settings.exchange != "polymarket":
            raise SystemExit("Dynamic watchlist building is currently implemented only for --exchange polymarket.")
        if self.settings.live_universe:
            if self.settings.exchange != "polymarket":
                raise SystemExit("--live-universe is currently implemented only for --exchange polymarket.")
            return await self._refresh_live_universe(first=True)
        if self.settings.scan_all_candidates:
            path = self.settings.candidate_pairs_path or self.settings.pairs_path
            pairs = await load_pairs(path, include_disabled=True)
            await resolve_missing_tokens(pairs, exchange=self.settings.exchange)
            return self._filter_pairs([pair for pair in pairs if all(_pair_leg_token_ids(pair))])
        if self.settings.dynamic_watchlist:
            return await self._refresh_dynamic_watchlist(first=True)
        pairs = await load_pairs(self.settings.pairs_path, include_disabled=self.settings.include_disabled)
        await resolve_missing_tokens(pairs, exchange=self.settings.exchange)
        return self._filter_pairs([pair for pair in pairs if all(_pair_leg_token_ids(pair))])

    async def _refresh_dynamic_watchlist(self, *, first: bool = False) -> list[PairConfig]:
        candidate_path = self.settings.candidate_pairs_path or self.settings.pairs_path
        summary = await build_watchlist_once(
            pairs_path=candidate_path,
            out_path=self.settings.watchlist_out,
            ranking_csv_path=self.settings.watchlist_ranking_csv,
            top_n=self.settings.watchlist_top_n,
            max_net_cost=self.settings.watchlist_max_net_cost,
            min_depth=self.settings.watchlist_min_depth,
            include_disabled=True,
            entry_threshold=self.settings.entry_threshold,
            min_edge_threshold=self.settings.min_edge_threshold,
            near_arb_threshold=self.settings.near_arb_threshold,
            slippage_buffer=self.settings.slippage_buffer,
            fee_rate=self.settings.fee_rate,
            max_concurrent_requests=self.settings.max_concurrent_requests,
            relation_safety=self.settings.relation_safety,
            min_display_price=self.settings.min_display_price,
            max_threshold_distance_pct=self.settings.max_threshold_distance_pct,
            spot_prices=self.settings.spot_prices,
        )
        pairs = await load_pairs(self.settings.watchlist_out, include_disabled=False)
        await resolve_missing_tokens(pairs, exchange=self.settings.exchange)
        pairs = self._filter_pairs([pair for pair in pairs if all(_pair_leg_token_ids(pair))])
        prefix = "Initial dynamic watchlist" if first else "Dynamic watchlist refreshed"
        if not self.settings.headless:
            print(
                f"{prefix}: selected={len(pairs)} from usable={summary['usable_pairs']} "
                f"best={summary['best_pair']} net={_fmt(summary['best_net_cost'])}"
            )
        return pairs

    async def _refresh_live_universe(self, *, first: bool = False) -> list[PairConfig]:
        from src.discovery import discover_live_multiday_pairs

        assets = [asset.strip().upper() for asset in self.settings.live_universe_assets.split(",") if asset.strip()]
        summary = await discover_live_multiday_pairs(
            assets=assets or None,
            horizon_days=self.settings.live_universe_horizon_days,
            include_boundary_ambiguous=True,
            adjacent_only=False,
            min_display_price=self.settings.min_display_price,
            spot_prices=self.settings.spot_prices,
            max_threshold_distance_pct=self.settings.max_threshold_distance_pct,
        )
        pairs = [PairConfig.from_dict(item) for item in summary["pairs"]]
        await resolve_missing_tokens(pairs, exchange=self.settings.exchange)
        pairs = self._filter_pairs([pair for pair in pairs if all(_pair_leg_token_ids(pair))])
        prefix = "Initial live universe" if first else "Live universe refreshed"
        print(
            f"{prefix}: pairs={len(pairs)} dates={summary['event_dates_covered']} "
            f"events_scanned={summary['events_scanned']} assets={summary['assets']}"
        )
        return pairs

    async def _prepare_pair_caches(self, pairs: list[PairConfig], clob: Any | None = None) -> None:
        self._pair_slippage_cache = {pair.name: _pair_slippage(pair, self.settings.slippage_buffer) for pair in pairs}
        fallback = {pair.name: _pair_fee_rates(pair, self.settings.fee_rate) for pair in pairs}
        self._pair_fee_rate_cache = dict(fallback)
        self._pair_cfg_cache = {
            pair.name: (
                _pair_fee_buffer(pair),
                _pair_max_spread(pair),
                _pair_min_edge_threshold(pair, self.settings.min_edge_threshold),
            )
            for pair in pairs
        }
        if self.settings.fee_rate is not None or not self.settings.dynamic_fee_rates or clob is None:
            return
        token_ids: list[str] = []
        for pair in pairs:
            token_ids.extend([token for token in _pair_leg_token_ids(pair) if token])
        rates = await clob.get_fee_rates(token_ids, max_concurrent_requests=self.settings.max_concurrent_requests)
        for token_id, rate in rates.items():
            if rate is not None:
                self._token_fee_rate_cache[token_id] = rate
        for pair in pairs:
            leg_1, leg_2 = _pair_leg_token_ids(pair)
            fallback_1, fallback_2 = fallback[pair.name]
            self._pair_fee_rate_cache[pair.name] = (
                rates.get(leg_1) if leg_1 and rates.get(leg_1) is not None else fallback_1,
                rates.get(leg_2) if leg_2 and rates.get(leg_2) is not None else fallback_2,
            )
        # Refresh scan arrays if they exist (fee rates may have changed)
        if self._scan_arrays is not None and self._scan_plan_cache is not None:
            self._scan_arrays = self._build_scan_arrays(self._scan_plan_cache)

    def _filter_pairs(self, pairs: list[PairConfig]) -> list[PairConfig]:
        return [
            pair
            for pair in pairs
            if _passes_relation_safety(pair, self.settings.relation_safety)
            and _passes_display_price(pair, self.settings.min_display_price)
            and _passes_threshold_distance(pair, self.settings.spot_prices, self.settings.max_threshold_distance_pct)
        ]

    def _scan_plan_for(self, pairs: list[PairConfig]) -> ScanPlan:
        # O(1) identity check: same list object (watchlist unchanged) → reuse cached plan.
        # Storing the reference also prevents GC from reusing the same address for a new list.
        if self._scan_plan_cache is not None and self._scan_plan_cache_pairs is pairs:
            return self._scan_plan_cache

        targets = tuple(PairScanTarget(pair=pair, leg_token_ids=_pair_leg_token_ids(pair)) for pair in pairs)
        n_leg_specs = tuple(_derive_range_threshold_n_leg_specs(pairs, max_ranges=self._n_leg_max_ranges()))

        # Build token → pair-index reverse map for dirty-set scan.
        _tok_to_idxs: dict[str, list[int]] = {}
        token_ids: list[str] = []
        for idx, target in enumerate(targets):
            for token in target.leg_token_ids:
                if token:
                    _tok_to_idxs.setdefault(token, []).append(idx)
                    token_ids.append(token)
        for spec in n_leg_specs:
            token_ids.extend(leg.token_id for leg in spec.legs)

        plan = ScanPlan(
            targets=targets,
            n_leg_specs=n_leg_specs,
            token_ids=tuple(dict.fromkeys(token_id for token_id in token_ids if token_id)),
            token_to_pair_idxs={t: tuple(idxs) for t, idxs in _tok_to_idxs.items()},
            n_leg_token_ids=frozenset(
                leg.token_id for spec in n_leg_specs for leg in spec.legs
            ),
        )
        self._scan_plan_cache_pairs = pairs
        self._scan_plan_cache = plan
        # Rebuild pre-allocated scan buffers with static config values
        self._scan_arrays = self._build_scan_arrays(plan)
        # Rebuild N-leg vectorised scan arrays
        self._n_leg_arrays = (
            _NLegScanArrays(list(n_leg_specs), self.settings.min_trade_size)
            if n_leg_specs
            else None
        )
        return plan

    def _build_scan_arrays(self, plan: ScanPlan) -> _ScanArrays:
        """Allocate numpy scan buffers and fill static per-pair config values.

        Falls back to computing values directly from pair config when the caches
        are not yet populated (e.g., tests that call _scan_once directly).
        """
        n = len(plan.targets)
        arrays = _ScanArrays(n)
        for i, target in enumerate(plan.targets):
            name = target.pair.name
            pair = target.pair

            cached_fr = self._pair_fee_rate_cache.get(name)
            fr1, fr2 = cached_fr if cached_fr is not None else _pair_fee_rates(pair, self.settings.fee_rate)

            cached_slip = self._pair_slippage_cache.get(name)
            slip = cached_slip if cached_slip is not None else _pair_slippage(pair, self.settings.slippage_buffer)

            cached_cfg = self._pair_cfg_cache.get(name)
            if cached_cfg is not None:
                fb, _, me = cached_cfg
            else:
                fb = _pair_fee_buffer(pair)
                me = _pair_min_edge_threshold(pair, self.settings.min_edge_threshold)

            arrays.fee_rates_1[i] = fr1
            arrays.fee_rates_2[i] = fr2
            arrays.slippages[i]   = slip
            arrays.fee_buffers[i] = fb
            arrays.min_edges[i]   = me
        return arrays

    def _scan_plan_signature(self, pairs: list[PairConfig]) -> tuple[Any, ...]:
        return (
            self._n_leg_max_ranges(),
            tuple(
                (
                    pair.name,
                    pair.parent_market_slug,
                    pair.child_market_slug,
                    pair.parent_outcome_label,
                    pair.child_outcome_label,
                    pair.parent_market_ticker,
                    pair.child_market_ticker,
                    pair.parent_yes_token_id,
                    pair.parent_no_token_id,
                    pair.child_yes_token_id,
                    pair.child_no_token_id,
                    _trade_template_signature(pair),
                    _pair_raw_str(pair, "parent_event_slug"),
                    _pair_raw_str(pair, "child_event_slug"),
                )
                for pair in pairs
            ),
        )

    # ── Concurrency helpers ───────────────────────────────────────────────────

    def _effective_cash(self) -> float:
        """Available cash minus capital reserved for in-flight background entries.

        Using self.cash directly would allow two concurrent background tasks to
        both see the full budget and double-spend.  Pre-reservation in
        _inflight_entries prevents that without requiring a lock (asyncio is
        single-threaded; the reservation happens before the first ``await``).
        """
        return self.cash - sum(self._inflight_entries.values())

    def _consume_n_leg_speculative_books(self, n_leg_name: str) -> dict[str, OrderBook]:
        """Return pre-fetched REST books from the speculative prefetch task, if ready.

        Called just before spawning a bg_enter_n_leg task.  When the speculative
        prefetch task (started during the preceding WS wait period) has completed
        for the same N-leg name, its books are returned so the background entry
        task needs no additional REST round-trip.

        Returns an empty dict when the task has not completed, was cancelled,
        raised an exception, or was for a different N-leg name.  The background
        entry task falls back to its own REST call in that case.
        """
        sp = self._n_leg_speculative_prefetch
        if (
            sp is not None
            and sp.done()
            and not sp.cancelled()
            and self._n_leg_speculative_name == n_leg_name
        ):
            try:
                return sp.result().books
            except Exception:
                pass
        return {}

    async def _bg_profile_dump(self) -> None:
        """Periodically snapshot per-stage percentiles to stages.csv (off the hot path).

        Runs as a background task when ``--profile`` is set.  Uses ``to_thread`` so
        the numpy percentile pass and file write never block the event loop.
        """
        interval = max(1.0, self.settings.profile_dump_seconds)
        while True:
            await asyncio.sleep(interval)
            rec = profiling.RECORDER
            if rec is not None:
                await asyncio.to_thread(rec.dump)

    async def _bg_save_book_snapshot(self) -> None:
        """Periodically persist WS book state to disk (best-effort).

        Runs as a fire-and-forget background task from ``_run_unlocked``.
        Only active when the provider is a WebSocketOrderBookProvider and
        ``book_snapshot_path`` is set.  Failures are silently swallowed by
        the ``save_snapshot`` helper (which logs at WARNING).
        """
        snap_path = self.settings.book_snapshot_path
        interval = max(10.0, self.settings.book_snapshot_interval_s)
        if snap_path is None:
            return
        while True:
            await asyncio.sleep(interval)
            provider = self._provider
            if provider is None:
                continue
            # Only WebSocketOrderBookProvider wraps a ClobWebSocketClient.
            ws_client = getattr(provider, "ws_client", None)
            if ws_client is None:
                continue
            save_fn = getattr(ws_client, "save_snapshot", None)
            if save_fn is not None:
                try:
                    await asyncio.to_thread(save_fn, snap_path)
                except Exception:
                    pass  # save_snapshot already logs; never crash the loop.

    async def _bg_enter_pair(
        self,
        timestamp: str,
        obs: PairObservation,
        pairs: list[PairConfig],
        books: dict[str, OrderBook],
        pair_by_name: dict[str, PairConfig],
        _prefetched_rest_books: dict[str, OrderBook],
    ) -> None:
        """Background wrapper for _try_enter; drains result into _pending_bg_actions."""
        pair_name = obs.pair_name
        try:
            action = await self._try_enter(
                timestamp, obs, pairs, books,
                pair_by_name=pair_by_name,
                _prefetched_rest_books=_prefetched_rest_books or None,
            )
        except Exception as exc:
            action = f"skipped: background entry error ({pair_name}: {exc})"
        finally:
            self._inflight_entries.pop(pair_name, None)
        self._profile_entry_outcome(action)
        self._pending_bg_actions.append(action)

    async def _bg_enter_n_leg(
        self,
        best_n_leg: dict[str, Any],
        books: dict[str, OrderBook],
        _prefetched_rest_books: dict[str, OrderBook],
    ) -> None:
        """Background wrapper for _try_enter_n_leg; drains result into _pending_bg_actions."""
        name = str(best_n_leg.get("name") or "")
        try:
            action = await self._try_enter_n_leg(
                best_n_leg, books,
                _prefetched_rest_books=_prefetched_rest_books or None,
            )
        except Exception as exc:
            action = f"skipped: background n-leg entry error ({name}: {exc})"
        finally:
            self._inflight_entries.pop(name, None)
        self._profile_entry_outcome(action)
        self._pending_bg_actions.append(action)

    def _profile_entry_outcome(self, action: str) -> None:
        """Record a backgrounded executable candidate's outcome for the profiler.

        Every candidate that reaches a bg-entry wrapper was *executable* at scan
        time (net cost below the entry threshold / positive N-leg edge), so any
        non-``entered`` outcome is a genuine missed fill.  We bucket it by reason
        (rest_recheck_lost, depth_depleted, capital_blocked, …) so the report can
        show *why* near-arbs fail to convert — the core "where am I losing
        opportunities?" question.  Reuses the audit's miss classifier.
        """
        if not (profiling.ENABLED and (rec := profiling.RECORDER) is not None):
            return
        if action.startswith("entered"):
            rec.incr("entries_filled")
        else:
            rec.incr("entries_missed")
            rec.incr_missed(_audit_miss_classification(action))

    # ─────────────────────────────────────────────────────────────────────────

    async def _scan_once(
        self,
        pairs: list[PairConfig],
        provider: OrderBookProvider,
        updated_tokens: set[str] | None = None,
    ) -> ScanRow:
        """Run one scan cycle.

        Parameters
        ----------
        updated_tokens:
            * ``None``        → full scan (default; used when book source is REST/FIX
                                or on the very first WS scan before any event arrives).
            * non-empty set   → dirty-set scan: only pairs whose tokens appear in
                                this set are re-scored; all others use cached
                                observations.  N-leg scan is skipped when none of
                                the updated tokens belong to any N-leg spec.
            * empty set       → timeout fired with no WS update; housekeeping only
                                (position marking, dashboard), skips the numpy classify
                                step entirely and re-uses cached observations.
        """
        started = time.perf_counter()
        scan_perf_ns = time.perf_counter_ns()
        scan_wall_ns = time.time_ns()
        if profiling.ENABLED and profiling.RECORDER is not None:
            profiling.RECORDER.begin_scan()
        scan_id = self._audit.next_id("scan") if self._audit.enabled else None
        timestamp = now_iso()
        plan = self._scan_plan_for(pairs)
        unique_token_ids = list(plan.token_ids)
        self._audit_record_timeline(
            event_kind="scan_start",
            scan_id=scan_id,
            timestamp_ns=scan_wall_ns,
            perf_ns=scan_perf_ns,
            message=f"tokens={len(unique_token_ids)} pairs={len(pairs)}",
        )
        # ── Early dirty-set + minimal book fetch ──────────────────────────────
        # CRITICAL ORDERING: compute which tokens need fresh books BEFORE calling
        # provider.get_books().  For the Rust WS path, get_book() per-token runs
        # from_float_data() (Python OrderBook reconstruction ≈ 170 µs/token).
        # Fetching all 131 tokens on every scan — including WS timeouts where
        # nothing changed — costs ~22 ms and defeats the purpose of the Rust client.
        #
        # Strategy:
        #   updated_tokens=None  → full scan: fetch all tokens (REST/first WS scan)
        #   updated_tokens={}    → WS timeout: fetch ONLY open-position tokens
        #   updated_tokens={...} → WS update: fetch dirty-pair + open + n-leg tokens
        #
        # Open-position indices are always merged into dirty_idxs so that:
        #   a) their tokens are included in the minimal fetch set, and
        #   b) _fast_observe_all re-fills their price arrays (needed for bid lookup).
        if updated_tokens is None:
            # Full scan — fetch everything, classify everything.
            dirty_idxs: frozenset[int] | None = None
            n_leg_dirty = True
            needed_token_ids = unique_token_ids
        else:
            # Identify open-position pair indices (always re-scored for exit detection).
            _open_idxs: frozenset[int] = frozenset(
                idx for idx, target in enumerate(plan.targets)
                if target.pair.name in self.open_pair_names
            )
            if not updated_tokens:
                # WS timeout — no book changed.  Only open positions need fresh books.
                dirty_idxs = _open_idxs          # empty when no open positions
                n_leg_dirty = False
            else:
                # WS update — re-score pairs whose tokens changed + open positions.
                _ws_dirty: frozenset[int] = frozenset(
                    idx
                    for token in updated_tokens
                    for idx in plan.token_to_pair_idxs.get(token, ())
                )
                dirty_idxs = _ws_dirty | _open_idxs
                n_leg_dirty = bool(updated_tokens & plan.n_leg_token_ids)

            # Build the minimal token fetch set.
            _needed: set[str] = set()
            for idx in dirty_idxs:
                for t in plan.targets[idx].leg_token_ids:
                    if t:
                        _needed.add(t)
            if n_leg_dirty:
                # Need all n-leg tokens when any n-leg token changed (can't partially
                # score n-leg specs — every leg's book is required for the argmax).
                for spec in plan.n_leg_specs:
                    for leg in spec.legs:
                        _needed.add(leg.token_id)
            needed_token_ids = list(_needed)

        _prof_t1 = time.perf_counter_ns()
        with self._audit.context(scan_id=scan_id):
            books = await provider.get_books(needed_token_ids)
        _prof_t2 = time.perf_counter_ns()
        # Microstructure sampling from books already in hand (O(1)/book, no fetch).
        if profiling.ENABLED:
            market_profiler.observe_books(books)
            market_profiler.maybe_flush(time.monotonic())
        self._audit_record_books(scan_id, books, provider, unique_token_ids)

        observations = self._fast_observe_all(timestamp, plan, books, dirty_idxs=dirty_idxs)
        _prof_t3 = time.perf_counter_ns()
        observation_ids = self._audit_record_pair_observations(scan_id, timestamp, plan.targets, observations)
        self._mark_positions(timestamp, observations)

        valid = [obs for obs in observations if obs.net_total_cost is not None]
        best = min(valid, key=lambda obs: obs.net_total_cost) if valid else None
        # Skip expensive N-leg scan when no N-leg tokens changed.
        if n_leg_dirty:
            best_n_leg = self._best_n_leg_opportunity(list(plan.n_leg_specs), books)
        else:
            best_n_leg = self._cached_best_n_leg if hasattr(self, "_cached_best_n_leg") else {}
        self._cached_best_n_leg = best_n_leg
        _prof_t4 = time.perf_counter_ns()
        n_leg_candidate_ids = self._audit_record_n_leg_candidates(scan_id, plan.n_leg_specs, books, best_n_leg)
        self._audit_update_opportunity_windows(
            scan_id=scan_id,
            timestamp_ns=time.time_ns(),
            observations=observations,
            best_n_leg=best_n_leg,
        )
        self._latest_top_lines = [
            f"{index}. {obs.pair_name} net={obs.net_total_cost:.4f} gross={obs.gross_total_cost:.4f} "
            f"dist={obs.distance_to_entry:.4f}"
            for index, obs in enumerate(sorted(valid, key=lambda item: item.net_total_cost or math.inf)[: self.settings.show_top], 1)
        ]
        executable = [obs for obs in valid if obs.net_total_cost is not None and obs.net_total_cost < self.settings.entry_threshold]
        near = [obs for obs in valid if obs.net_total_cost is not None and obs.net_total_cost <= self.settings.near_arb_threshold]
        # ── Alpha-decay tracking ──────────────────────────────────────────────
        # Record how long each near-arb opportunity persists.  When a pair leaves
        # the near set, opportunity_lifetime captures its duration — the single
        # most important number for a latency-bound arb bot: if opportunities die
        # faster than our tick-to-trade, we structurally cannot capture them.
        if profiling.ENABLED and (_rec := profiling.RECORDER) is not None:
            _now_ns = time.perf_counter_ns()
            _near_now = {obs.pair_name for obs in near}
            for _pn in _near_now:
                if _pn not in self._near_since_ns:
                    self._near_since_ns[_pn] = _now_ns
                    _rec.incr("near_arb_appearances")
            for _pn in [p for p in self._near_since_ns if p not in _near_now]:
                _rec.record_series("opportunity_lifetime", _now_ns - self._near_since_ns.pop(_pn))
        rejected = [obs for obs in observations if obs.classification == "REJECTED"]
        books_missing = sum(1 for obs in rejected if obs.rejection_reason == "missing_order_book")
        asks_missing = sum(1 for obs in rejected if obs.rejection_reason == "missing_ask")

        actions: list[str] = []
        attempted_actions: list[str] = []
        pair_by_name = {pair.name: pair for pair in pairs}

        # Persist near-arb tokens for the speculative pair pre-fetch launched in
        # the main loop before the next wait_for_updates.  Only near candidates
        # (not just executable) are included because executable pairs are a subset
        # of near and the pre-fetch fires one scan before they might be entered.
        self._hot_pair_tokens = list(dict.fromkeys(
            t
            for obs in near
            for t in _pair_leg_token_ids(pair_by_name[obs.pair_name])
            if t
        ))
        _prof_t5 = time.perf_counter_ns()

        # ── Drain completed background entries from previous scans ─────────────
        # Background entry tasks deposit their results into _pending_bg_actions.
        # We flush them here (at the top of each scan) so they appear in the
        # action column of the row that follows the scan that spawned them.
        if self._pending_bg_actions:
            _bg_done = self._pending_bg_actions[:]
            self._pending_bg_actions.clear()
            for _bg_a in _bg_done:
                attempted_actions.append(_bg_a)
                if _bg_a.startswith("entered"):
                    actions.append(_bg_a)

        # ── Consume speculative pair REST books (zero blocking await) ─────────
        # The main loop fires a background REST fetch for near-arb pair tokens
        # immediately after each scan (before wait_for_updates).  asyncio httpx
        # is non-blocking: the HTTP response arrives while the event loop idles
        # in wait_for_updates, so the task is typically done before the next
        # scan starts.  When ready, books are forwarded to every bg_enter_pair
        # task with zero additional REST latency.
        #
        # Coverage check: only pass speculative books to a pair's background task
        # if both of its tokens were in the pre-fetched set.  Pairs that were not
        # near-arb in the previous scan (and therefore not prefetched) receive
        # _prefetched_rest_books=None — their bg_enter_pair task then does its
        # own REST call, which runs concurrently with the next wait_for_updates
        # and is invisible to the scan loop.
        rest_books_cache: dict[str, OrderBook] = {}
        _p_sp = self._pair_speculative_prefetch
        if _p_sp is not None and _p_sp.done() and not _p_sp.cancelled():
            try:
                rest_books_cache = _p_sp.result().books
            except Exception:
                pass

        for candidate in sorted(executable, key=lambda obs: obs.net_total_cost or math.inf):
            # Skip candidates already being handled by an in-flight background task
            # so we never double-spend capital.
            if candidate.pair_name in self._inflight_entries:
                attempted_actions.append(
                    f"skipped: background entry in-flight ({candidate.pair_name})"
                )
                continue
            candidate_pair = pair_by_name[candidate.pair_name]
            opportunity_id = _audit_pair_opportunity_id(candidate.pair_name)
            decision_id = self._audit.next_id("decision") if self._audit.enabled else None
            portfolio_snapshot_id = self._audit_record_portfolio_snapshot(scan_id, timestamp, decision_id)
            decision_started_perf_ns = time.perf_counter_ns()
            decision_started_wall_ns = time.time_ns()
            # Pair entry is always background: REST recheck must never block
            # the scan loop.  Capital is reserved synchronously (before any
            # await) to prevent double-spend across concurrent tasks.
            # Pass speculative books only when they cover both of the pair's
            # tokens; otherwise the background task falls back to its own
            # REST call — which runs during the next wait_for_updates and is
            # invisible to the scan loop.
            _bg_reserve = min(
                _pair_max_trade_size(candidate_pair, self.settings.max_trade_size),
                self._effective_cash() * self.settings.capital_fraction_per_trade,
                self._effective_cash(),
            )
            if _bg_reserve < self.settings.min_trade_size:
                action = f"skipped: insufficient cash for background entry ({candidate.pair_name})"
                attempted_actions.append(action)
                actions.append(action)
                # Executable candidate we couldn't even stake — count as a miss so
                # capital starvation shows up alongside the recheck-stage reasons.
                if profiling.ENABLED and (_rec := profiling.RECORDER) is not None:
                    _rec.incr("entries_missed")
                    _rec.incr_missed("capital_blocked")
                break
            _t1, _t2 = _pair_leg_token_ids(candidate_pair)
            _pair_covered = (
                (_t1 is None or _t1 in rest_books_cache)
                and (_t2 is None or _t2 in rest_books_cache)
            )
            self._inflight_entries[candidate.pair_name] = _bg_reserve
            _enter_task = asyncio.create_task(
                self._bg_enter_pair(
                    timestamp, candidate, pairs, books, pair_by_name,
                    _prefetched_rest_books=rest_books_cache if _pair_covered else None,
                )
            )
            _enter_task.add_done_callback(_log_task_exception)
            action = f"bg entry spawned ({candidate.pair_name})"
            attempted_actions.append(action)
            continue
            decision_finished_perf_ns = time.perf_counter_ns()
            attempted_actions.append(action)
            self._audit_record_decision(
                scan_id=scan_id,
                decision_id=decision_id,
                opportunity_id=opportunity_id,
                candidate_type="pair",
                candidate_name=candidate.pair_name,
                observation_id=observation_ids.get(candidate.pair_name),
                n_leg_candidate_id=None,
                portfolio_snapshot_id=portfolio_snapshot_id,
                action=action,
                decision_wall_ns=decision_started_wall_ns,
                decision_perf_ns=decision_started_perf_ns,
                decision_to_ack_ms=(decision_finished_perf_ns - decision_started_perf_ns) / 1_000_000,
                book_to_detection_ms=self._audit_book_to_detection_ms(candidate_pair, provider),
                detection_to_decision_ms=(decision_started_perf_ns - scan_perf_ns) / 1_000_000,
                edge=(1.0 - candidate.net_total_cost) if candidate.net_total_cost is not None else None,
                gross_cost=candidate.gross_total_cost,
                net_cost=candidate.net_total_cost,
                size=candidate.optimal_size,
                locked_capital=candidate.optimal_required_capital,
                passed_spread_check=candidate.rejection_reason != "spread_too_wide",
                passed_depth_check=candidate.max_executable_size > 0,
                passed_edge_check=candidate.classification == "EXECUTABLE_ARBITRAGE_CANDIDATE",
            )
            if action.startswith("entered"):
                actions.append(action)
                continue
            self.missed.append(
                {
                    "timestamp": timestamp,
                    "pair_name": candidate.pair_name,
                    "reason": action,
                    "total_cost": candidate.gross_total_cost,
                    "worst_case_profit": candidate.worst_case_profit_per_unit,
                }
            )
            self._audit_record_missed_fill(
                scan_id=scan_id,
                decision_id=decision_id,
                opportunity_id=opportunity_id,
                candidate_name=candidate.pair_name,
                candidate_type="pair",
                action=action,
                expected_profit=candidate.optimal_guaranteed_profit or candidate.worst_case_profit_per_unit,
                edge=(1.0 - candidate.net_total_cost) if candidate.net_total_cost is not None else None,
                gross_cost=candidate.gross_total_cost,
                net_cost=candidate.net_total_cost,
                detected_ts_ns=scan_wall_ns,
                decision_ts_ns=decision_started_wall_ns,
                token_ids=[token for token in _pair_leg_token_ids(candidate_pair) if token],
            )
            # Once global capital/risk limits are hit, later candidates cannot enter this scan.
            if any(
                marker in action
                for marker in [
                    "max open positions reached",
                    "max locked capital reached",
                    "insufficient cash",
                ]
            ):
                actions.append(action)
                break
        if self._n_leg_trading_enabled() and best_n_leg:
            _n_leg_name = str(best_n_leg.get("name") or "")
            opportunity_id = _audit_n_leg_opportunity_id(_n_leg_name)
            decision_id = self._audit.next_id("decision") if self._audit.enabled else None
            portfolio_snapshot_id = self._audit_record_portfolio_snapshot(scan_id, timestamp, decision_id)
            decision_started_perf_ns = time.perf_counter_ns()
            decision_started_wall_ns = time.time_ns()
            if _n_leg_name in self._inflight_entries:
                n_leg_action = f"skipped: n-leg entry in-flight ({_n_leg_name})"
            else:
                # N-leg entry is ALWAYS spawned as a background asyncio task —
                # the REST recheck inside _try_enter_n_leg must never block the
                # scan loop (40 ms REST × every scan = disaster).  Capital is
                # reserved here, synchronously before any await, to prevent
                # double-spend across concurrent tasks.
                #
                # Speculative pre-fetch pipeline: the main loop kicks off a REST
                # fetch for these tokens immediately after each scan (before
                # wait_for_updates).  If that task is done by the time we get
                # here, _consume_n_leg_speculative_books() returns fresh books
                # and the background task spends ~0 ms on REST.  If not (first
                # scan or N-leg just changed), the task falls back to its own
                # REST call concurrently with the next WS wait.
                _bg_n_reserve = min(
                    self.settings.max_trade_size * int(best_n_leg.get("leg_count") or 3),
                    self._effective_cash(),
                )
                if _bg_n_reserve >= self.settings.min_trade_size and float(best_n_leg.get("gross_edge") or 0) > 0:
                    _spec_books = self._consume_n_leg_speculative_books(_n_leg_name)
                    self._inflight_entries[_n_leg_name] = _bg_n_reserve
                    with self._audit.context(scan_id=scan_id, decision_id=decision_id, opportunity_id=opportunity_id):
                        _n_leg_task = asyncio.create_task(
                            self._bg_enter_n_leg(
                                best_n_leg, books,
                                _prefetched_rest_books=_spec_books or None,
                            )
                        )
                        _n_leg_task.add_done_callback(_log_task_exception)
                    n_leg_action = f"bg n-leg entry spawned ({_n_leg_name})"
                else:
                    n_leg_action = "none"
            decision_finished_perf_ns = time.perf_counter_ns()
            self._audit_record_decision(
                scan_id=scan_id,
                decision_id=decision_id,
                opportunity_id=opportunity_id,
                candidate_type="n_leg",
                candidate_name=str(best_n_leg.get("name") or ""),
                observation_id=None,
                n_leg_candidate_id=n_leg_candidate_ids.get(str(best_n_leg.get("name") or "")),
                portfolio_snapshot_id=portfolio_snapshot_id,
                action=n_leg_action,
                decision_wall_ns=decision_started_wall_ns,
                decision_perf_ns=decision_started_perf_ns,
                decision_to_ack_ms=(decision_finished_perf_ns - decision_started_perf_ns) / 1_000_000,
                book_to_detection_ms=self._audit_n_leg_book_to_detection_ms(best_n_leg, provider),
                detection_to_decision_ms=(decision_started_perf_ns - scan_perf_ns) / 1_000_000,
                edge=best_n_leg.get("gross_edge"),
                gross_cost=best_n_leg.get("gross_cost"),
                net_cost=best_n_leg.get("gross_cost"),
                size=best_n_leg.get("optimal_size") or best_n_leg.get("max_spend_size"),
                locked_capital=best_n_leg.get("optimal_capital") or best_n_leg.get("max_spend_capital"),
                passed_spread_check=True,
                passed_depth_check=bool(best_n_leg.get("gross_cost")),
                passed_edge_check=(float(best_n_leg.get("gross_edge") or 0.0) > 0.0),
            )
            if n_leg_action != "none":
                actions.append(n_leg_action)
                if not n_leg_action.startswith(("entered", "bg n-leg entry spawned", "skipped: n-leg entry in-flight")):
                    self._audit_record_missed_fill(
                        scan_id=scan_id,
                        decision_id=decision_id,
                        opportunity_id=opportunity_id,
                        candidate_name=str(best_n_leg.get("name") or ""),
                        candidate_type="n_leg",
                        action=n_leg_action,
                        expected_profit=best_n_leg.get("optimal_profit") or best_n_leg.get("max_spend_profit") or best_n_leg.get("gross_edge"),
                        edge=best_n_leg.get("gross_edge"),
                        gross_cost=best_n_leg.get("gross_cost"),
                        net_cost=best_n_leg.get("gross_cost"),
                        detected_ts_ns=scan_wall_ns,
                        decision_ts_ns=decision_started_wall_ns,
                        token_ids=[str(token) for token in best_n_leg.get("leg_token_ids") or []],
                    )
        # ── Near-arb priority seeding ──────────────────────────────────────────
        # When there are near-arb candidates (net_total_cost ≤ near_arb_threshold)
        # we fire a background REST seed for their tokens to flush stale WS prices
        # faster than the normal fallback_cache_ms window.  Throttled by
        # near_arb_seed_interval_seconds so we don't hammer the REST endpoint every
        # scan (which runs at event-driven frequency — potentially hundreds of times
        # per second).  Limited to the top-K pairs by proximity to entry.
        if (
            self.settings.near_arb_priority_seed
            and near
            and isinstance(self._provider, WebSocketOrderBookProvider)
            and (time.monotonic() - self._last_priority_seed_at) >= self.settings.near_arb_seed_interval_seconds
        ):
            _seed_tokens: list[str] = []
            for _near_obs in sorted(near, key=lambda o: o.net_total_cost or math.inf)[: self.settings.near_arb_seed_top_k]:
                _near_pair = pair_by_name.get(_near_obs.pair_name)
                if _near_pair is not None:
                    _seed_tokens.extend(t for t in _pair_leg_token_ids(_near_pair) if t)
            _seed_tokens = list(dict.fromkeys(_seed_tokens))  # deduplicate, preserve order
            if _seed_tokens:
                _seed_task = asyncio.create_task(self._provider.priority_seed(_seed_tokens))
                _seed_task.add_done_callback(_log_task_exception)
                self._last_priority_seed_at = time.monotonic()
        _prof_t6 = time.perf_counter_ns()

        if actions:
            action = "; ".join(actions)
        elif attempted_actions:
            action = "; ".join(attempted_actions[:3])
        else:
            action = "none"

        realized = sum(pos.realized_pnl for pos in self.positions if pos.status == "closed")
        liquidation_pnl = sum(pos.liquidation_pnl for pos in self.positions if pos.status == "open")
        unrealized = liquidation_pnl
        guaranteed = sum(pos.worst_case_profit for pos in self.positions if pos.status == "open")
        best_case = sum(pos.best_case_profit for pos in self.positions if pos.status == "open")
        scan_time_ms = (time.perf_counter() - started) * 1000
        if profiling.ENABLED and (_rec := profiling.RECORDER) is not None:
            _prof_end = time.perf_counter_ns()
            _rec.record("s1_dirtyset", _prof_t1 - scan_perf_ns)
            _rec.record("s2_get_books", _prof_t2 - _prof_t1)
            _rec.record("s3_fast_observe_all", _prof_t3 - _prof_t2)
            _rec.record("s4_nleg", _prof_t4 - _prof_t3)
            _rec.record("s5_filter", _prof_t5 - _prof_t4)
            _rec.record("s6_try_enter", _prof_t6 - _prof_t5)
            _rec.record("total_scan", _prof_end - scan_perf_ns)
            _rec.maybe_record_slow_scan(
                _prof_end - scan_perf_ns,
                {
                    "dirty_tokens": len(needed_token_ids),
                    "unique_tokens": len(unique_token_ids),
                    "n_leg_dirty": bool(n_leg_dirty),
                    "entered": bool(actions),
                },
            )
        self._audit_record_timeline(
            event_kind="scan_end",
            scan_id=scan_id,
            timestamp_ns=time.time_ns(),
            perf_ns=time.perf_counter_ns(),
            metric_1=scan_time_ms,
            message=f"action={action}",
        )
        max_book_age_ms = max(
            [age for token_id in unique_token_ids if (age := provider.book_age_ms(token_id)) is not None],
            default=None,
        )
        stats = getattr(provider, "stats", None)
        # Pipeline-latency records taken once per scan, on the event-loop thread:
        #   book_age_at_decision — how stale the books we just scored were.
        #   wire_latency         — exchange-event → local-receipt for the freshest
        #                          batch (provider aggregates max over fetched tokens;
        #                          populated for WS providers, None for pure polling).
        # Both stored in ns (×1e6 from ms) so they share the percentile machinery.
        if profiling.ENABLED and (_rec := profiling.RECORDER) is not None:
            if max_book_age_ms is not None:
                _rec.record_series("book_age_at_decision", max_book_age_ms * 1e6)
            _wire_ms = getattr(stats, "update_latency_ms", None) if stats is not None else None
            if _wire_ms is not None:
                _rec.record_series("wire_latency", _wire_ms * 1e6)
        return ScanRow(
            timestamp=timestamp,
            cash_available=self.cash,
            locked_capital=sum(pos.locked_capital for pos in self.positions if pos.status == "open"),
            open_positions_count=sum(1 for pos in self.positions if pos.status == "open"),
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            liquidation_pnl=liquidation_pnl,
            guaranteed_profit_if_held=guaranteed,
            best_case_profit_if_held=best_case,
            best_pair_name=best.pair_name if best else None,
            best_total_cost=best.gross_total_cost if best else None,
            net_total_cost=best.net_total_cost if best else None,
            entry_threshold=self.settings.entry_threshold,
            distance_to_entry=best.distance_to_entry if best else None,
            best_worst_case_profit=best.worst_case_profit_per_unit if best else None,
            best_optimal_size=best.optimal_size if best else None,
            best_optimal_guaranteed_profit=best.optimal_guaranteed_profit if best else None,
            executable_candidates_count=len(executable),
            near_arb_candidates_count=len(near),
            rejected_count=len(rejected),
            books_missing_count=books_missing,
            asks_missing_count=asks_missing,
            scan_time_ms=scan_time_ms,
            book_source=self.settings.book_source,
            unique_tokens=len(unique_token_ids),
            unique_tokens_fetched=getattr(stats, "unique_tokens_fetched", 0),
            cache_hits=getattr(stats, "cache_hits", 0),
            failed_book_count=getattr(stats, "failed_book_count", 0),
            websocket_connected=getattr(stats, "websocket_connected", False),
            websocket_reconnect_count=getattr(stats, "websocket_reconnect_count", 0),
            fallback_to_polling_used=getattr(stats, "fallback_to_polling_used", False),
            token_update_count=getattr(stats, "token_update_count", 0),
            event_triggered_recomputes=getattr(stats, "event_triggered_recomputes", 0),
            max_book_age_ms=max_book_age_ms,
            update_latency_ms=getattr(stats, "update_latency_ms", None),
            action_taken=action,
            best_n_leg_name=best_n_leg.get("name") if best_n_leg else None,
            best_n_leg_leg_count=int(best_n_leg.get("leg_count")) if best_n_leg else None,
            best_n_leg_gross_cost=best_n_leg.get("gross_cost") if best_n_leg else None,
            best_n_leg_guaranteed_payout=best_n_leg.get("guaranteed_payout") if best_n_leg else None,
            best_n_leg_gross_edge=best_n_leg.get("gross_edge") if best_n_leg else None,
            best_three_leg_name=best_n_leg.get("name") if best_n_leg and best_n_leg.get("leg_count") == 3 else None,
            best_three_leg_gross_cost=best_n_leg.get("gross_cost") if best_n_leg and best_n_leg.get("leg_count") == 3 else None,
            best_three_leg_gross_edge=best_n_leg.get("gross_edge") if best_n_leg and best_n_leg.get("leg_count") == 3 else None,
        )

    def _fast_observe_all(
        self,
        timestamp: str,
        plan: ScanPlan,
        books: dict[str, OrderBook],
        dirty_idxs: frozenset[int] | None = None,
    ) -> list[PairObservation]:
        """Vectorised scan: classify all pairs at C/numpy speed, call _observe_pair
        only for the small subset that passes the near-arb threshold.

        Hot path:
          1. Fill price arrays — one Python loop, O(n) dict+attr lookups (~20 µs)
          2. scan_classify_batch — C/numba/numpy loop, all arithmetic (~2–10 µs)
          3. _observe_pair — only for near-arb or open-position pairs (~5 pairs)

        Dirty-set mode (dirty_idxs is not None):
          Only pairs in dirty_idxs are re-scored from the current books.
          All other pairs reuse their cached PairObservation from the previous scan,
          except for pairs that are currently open (which are always re-scored to
          catch exit conditions even when their tokens did not update).
        """
        arrays = self._scan_arrays
        n = len(plan.targets)
        near_t = self.settings.near_arb_threshold
        entry_t = self.settings.entry_threshold

        # ── Step 1: fill dynamic price arrays (unavoidable Python loop) ───────
        if arrays is None or arrays.n != n:
            arrays = self._build_scan_arrays(plan)
            self._scan_arrays = arrays

        # In dirty-set mode only re-fill arrays for changed pairs.
        fill_all = dirty_idxs is None
        fill_range = range(n) if fill_all else dirty_idxs

        for i in fill_range:
            target = plan.targets[i]
            t1, t2 = target.leg_token_ids
            b1 = books.get(t1 or "")
            b2 = books.get(t2 or "")
            if b1 is not None and b2 is not None:
                ba1 = b1.best_ask
                ba2 = b2.best_ask
                if ba1 is not None and ba2 is not None:
                    arrays.parent_asks[i] = ba1  # already float
                    arrays.child_asks[i]  = ba2
                    arrays.book_valid[i]  = True
                    continue
            arrays.parent_asks[i] = 2.0
            arrays.child_asks[i]  = 2.0
            arrays.book_valid[i]  = False

        # ── Step 2: C/numba/numpy classification for all pairs ────────────────
        # Skip entirely when the dirty set is empty: no prices changed (WS
        # timeout with no open positions), so class_out / gross_out / net_out
        # from the previous scan are still valid.  All pairs will return their
        # cached observation in Step 3.
        # When dirty_idxs is None (full scan) or non-empty, always classify.
        if dirty_idxs is None or dirty_idxs:
            _scan_classify_batch(
                arrays.parent_asks, arrays.child_asks,
                arrays.fee_rates_1, arrays.fee_rates_2,
                arrays.slippages, arrays.fee_buffers, arrays.min_edges,
                near_t, entry_t, n,
                arrays.gross_out, arrays.net_out, arrays.class_out,
            )

        # ── Step 3: build PairObservation objects ─────────────────────────────
        observations: list[PairObservation] = []
        for i, target in enumerate(plan.targets):
            pair = target.pair
            is_open = pair.name in self.open_pair_names
            is_dirty = fill_all or (dirty_idxs is not None and i in dirty_idxs)

            # Clean pairs that are not currently open → return cached observation.
            # Open pairs are always re-scored so exit conditions are checked even
            # when their underlying token prices did not change this scan.
            if not is_dirty and not is_open:
                cached = arrays.obs_cache[i]
                if cached is not None:
                    observations.append(cached)
                    continue
                # No cache yet (first scan in dirty-set mode) — fall through to score.

            cls = int(arrays.class_out[i])
            valid = bool(arrays.book_valid[i])

            if not valid:
                # Missing book — ask _observe_pair to return the canonical rejection
                obs = self._observe_pair(timestamp, pair, books, target.leg_token_ids)
            elif cls == _SCAN_REJECTED and not is_open:
                # Fast rejection: use pre-computed values, skip full observation
                obs = PairObservation(
                    timestamp=timestamp,
                    pair_name=pair.name,
                    parent_outcome_label=pair.parent_outcome_label,
                    child_outcome_label=pair.child_outcome_label,
                    parent_yes_ask=float(arrays.parent_asks[i]),
                    child_no_ask=float(arrays.child_asks[i]),
                    parent_yes_bid=None,
                    child_no_bid=None,
                    gross_total_cost=float(arrays.gross_out[i]),
                    estimated_fee_total_per_unit=0.0,
                    slippage_buffer=float(arrays.slippages[i]),
                    net_total_cost=float(arrays.net_out[i]),
                    entry_threshold=entry_t,
                    distance_to_entry=float(arrays.net_out[i]) - entry_t,
                    worst_case_profit_per_unit=1.0 - float(arrays.net_out[i]),
                    best_case_profit_per_unit=2.0 - float(arrays.net_out[i]),
                    max_executable_size=0.0,
                    classification="REJECTED",
                    rejection_reason="too_expensive",
                )
            else:
                # Near-arb or open-position pair — full observation (bid lookups, depth, etc.)
                obs = self._observe_pair(timestamp, pair, books, target.leg_token_ids)

            # Update cache for future dirty-set scans.
            arrays.obs_cache[i] = obs
            observations.append(obs)
        return observations

    def _observe_pair(
        self,
        timestamp: str,
        pair: PairConfig,
        books: dict[str, OrderBook],
        leg_token_ids: tuple[str | None, str | None] | None = None,
    ) -> PairObservation:
        leg_1_token, leg_2_token = leg_token_ids or _pair_leg_token_ids(pair)
        parent_book = books.get(leg_1_token or "")
        child_book = books.get(leg_2_token or "")
        slippage = self._pair_slippage_cache.get(pair.name)
        if slippage is None:
            slippage = _pair_slippage(pair, self.settings.slippage_buffer)
            self._pair_slippage_cache[pair.name] = slippage
        fee_rates = self._pair_fee_rate_cache.get(pair.name)
        if fee_rates is None:
            fee_rates = _pair_fee_rates(pair, self.settings.fee_rate)
            self._pair_fee_rate_cache[pair.name] = fee_rates
        if parent_book is None or child_book is None:
            return PairObservation(
                timestamp=timestamp,
                pair_name=pair.name,
                parent_outcome_label=pair.parent_outcome_label,
                child_outcome_label=pair.child_outcome_label,
                parent_yes_ask=None,
                child_no_ask=None,
                parent_yes_bid=None,
                child_no_bid=None,
                gross_total_cost=None,
                estimated_fee_total_per_unit=0.0,
                slippage_buffer=slippage,
                net_total_cost=None,
                entry_threshold=self.settings.entry_threshold,
                distance_to_entry=None,
                worst_case_profit_per_unit=None,
                best_case_profit_per_unit=None,
                max_executable_size=0.0,
                classification="REJECTED",
                rejection_reason="missing_order_book",
            )
        parent_ask = _float(parent_book.best_ask)
        child_ask = _float(child_book.best_ask)
        if parent_ask is None or child_ask is None:
            return PairObservation(
                timestamp=timestamp,
                pair_name=pair.name,
                parent_outcome_label=pair.parent_outcome_label,
                child_outcome_label=pair.child_outcome_label,
                parent_yes_ask=parent_ask,
                child_no_ask=child_ask,
                parent_yes_bid=None,
                child_no_bid=None,
                gross_total_cost=None,
                estimated_fee_total_per_unit=0.0,
                slippage_buffer=slippage,
                net_total_cost=None,
                entry_threshold=self.settings.entry_threshold,
                distance_to_entry=None,
                worst_case_profit_per_unit=None,
                best_case_profit_per_unit=None,
                max_executable_size=0.0,
                classification="REJECTED",
                rejection_reason="missing_ask",
            )
        gross = parent_ask + child_ask
        # Fast rejection: if clearly too expensive AND no open position needs bid-side marking,
        # skip fee computation, bid lookups, spread checks, and depth queries entirely.
        if gross > self.settings.near_arb_threshold and pair.name not in self.open_pair_names:
            return PairObservation(
                timestamp=timestamp,
                pair_name=pair.name,
                parent_outcome_label=pair.parent_outcome_label,
                child_outcome_label=pair.child_outcome_label,
                parent_yes_ask=parent_ask,
                child_no_ask=child_ask,
                parent_yes_bid=None,
                child_no_bid=None,
                gross_total_cost=gross,
                estimated_fee_total_per_unit=0.0,
                slippage_buffer=slippage,
                net_total_cost=gross,  # approximate — fees push it even higher, still rejected
                entry_threshold=self.settings.entry_threshold,
                distance_to_entry=gross - self.settings.entry_threshold,
                worst_case_profit_per_unit=1.0 - gross,
                best_case_profit_per_unit=2.0 - gross,
                max_executable_size=0.0,
                classification="REJECTED",
                rejection_reason="too_expensive",
            )
        # Full observation path: near-arb candidate or has an open position to mark.
        # Use pre-computed per-pair config values from cache.
        fee_buffer, max_spread, min_edge = self._pair_cfg_cache.get(
            pair.name, (_pair_fee_buffer(pair), _pair_max_spread(pair),
                        _pair_min_edge_threshold(pair, self.settings.min_edge_threshold))
        )
        fee = estimated_fee_per_unit(parent_ask, fee_rates[0]) + estimated_fee_per_unit(child_ask, fee_rates[1])
        net = gross + fee + slippage + fee_buffer
        parent_bid = _float(parent_book.best_bid)
        child_bid = _float(child_book.best_bid)
        max_size = float(min(parent_book.ask_depth, child_book.ask_depth))
        worst = 1.0 - net
        best = 2.0 - net
        if max_size <= 0:
            classification = "REJECTED"
            rejection_reason = "no_ask_depth"
        elif max_spread is not None and (parent_bid is None or child_bid is None):
            classification = "REJECTED"
            rejection_reason = "missing_bid_for_spread_check"
        elif max_spread is not None and (
            (parent_ask - parent_bid) > max_spread or (child_ask - child_bid) > max_spread  # type: ignore[operator]
        ):
            classification = "REJECTED"
            rejection_reason = "spread_too_wide"
        elif net < self.settings.entry_threshold and worst >= min_edge:
            classification = "EXECUTABLE_ARBITRAGE_CANDIDATE"
            rejection_reason = None
        elif net <= self.settings.near_arb_threshold:
            classification = "NEAR_ARBITRAGE"
            rejection_reason = None
        else:
            classification = "REJECTED"
            rejection_reason = "too_expensive"
        optimizer_cutoff = max(self.settings.entry_threshold, self.settings.optimizer_net_cutoff)
        optimal = (
            self._find_best_size(parent_book, child_book, pair, require_entry=False)
            if self.settings.sizing_mode == "max_profit"
            and net <= optimizer_cutoff
            and classification == "EXECUTABLE_ARBITRAGE_CANDIDATE"
            else {}
        )
        return PairObservation(
            timestamp=timestamp,
            pair_name=pair.name,
            parent_outcome_label=pair.parent_outcome_label,
            child_outcome_label=pair.child_outcome_label,
            parent_yes_ask=parent_ask,
            child_no_ask=child_ask,
            parent_yes_bid=parent_bid,
            child_no_bid=child_bid,
            gross_total_cost=gross,
            estimated_fee_total_per_unit=fee,
            slippage_buffer=slippage,
            net_total_cost=net,
            entry_threshold=self.settings.entry_threshold,
            distance_to_entry=net - self.settings.entry_threshold,
            worst_case_profit_per_unit=worst,
            best_case_profit_per_unit=best,
            max_executable_size=max_size,
            optimal_size=optimal.get("size"),
            optimal_required_capital=optimal.get("capital"),
            optimal_guaranteed_profit=optimal.get("profit"),
            optimal_net_cost_per_unit=optimal.get("net"),
            classification=classification,
            rejection_reason=rejection_reason,
        )

    async def _try_enter(
        self,
        timestamp: str,
        obs: PairObservation,
        pairs: list[PairConfig],
        books: dict[str, OrderBook],
        pair_by_name: dict[str, PairConfig] | None = None,
        _prefetched_rest_books: dict[str, OrderBook] | None = None,
    ) -> str:
        pair = (pair_by_name or {pair.name: pair for pair in pairs})[obs.pair_name]
        if not self.settings.allow_multiple_open_per_pair and pair.name in self.open_pair_names:
            return f"skipped: duplicate open pair prevented ({pair.name})"
        open_position_count = sum(self.open_count_by_pair.values())
        if self.settings.max_open_positions is not None and open_position_count >= self.settings.max_open_positions:
            return f"skipped: max open positions reached ({pair.name})"
        if (
            self.settings.max_open_positions_per_pair is not None
            and self.open_count_by_pair.get(pair.name, 0) >= self.settings.max_open_positions_per_pair
        ):
            return f"skipped: max pair positions reached ({pair.name})"
        last_trade_time = self.last_trade_time_by_pair.get(pair.name)
        if last_trade_time is not None and time.monotonic() - last_trade_time < self.settings.cooldown_seconds_per_pair:
            return f"skipped: cooldown active ({pair.name})"
        if obs.net_total_cost is None or obs.parent_yes_ask is None or obs.child_no_ask is None:
            return f"skipped: incomplete quote ({pair.name})"

        # ── Fill-rate confidence gate ─────────────────────────────────────────
        # Require a higher raw edge for pairs with a poor historical fill rate.
        # Only applied before the REST recheck so the scan uses WS-side prices;
        # the post-recheck edge check uses the confirmed REST price.
        if self.settings.fill_rate_tracking:
            _fr_confidence = self._fill_tracker.confidence(pair.name)
            _base_me = _pair_min_edge_threshold(pair, self.settings.min_edge_threshold)
            _adj_me = _base_me / max(_fr_confidence, self.settings.fill_rate_confidence_floor)
            if (1.0 - obs.net_total_cost) < _adj_me:
                return (
                    f"skipped: below fill-rate-adjusted edge "
                    f"(conf={_fr_confidence:.2f}) ({pair.name})"
                )

        if self.settings.entry_rest_recheck and self._clob_client is not None:
            if _prefetched_rest_books is not None:
                rest_books = _prefetched_rest_books
            else:
                batch = await self._clob_client.get_order_books(
                    [token for token in _pair_leg_token_ids(pair) if token],
                    max_concurrent_requests=2,
                )
                rest_books = batch.books
            rechecked_obs = self._observe_pair(timestamp, pair, rest_books)
            if rechecked_obs.net_total_cost is None or rechecked_obs.parent_yes_ask is None or rechecked_obs.child_no_ask is None:
                if self.settings.fill_rate_tracking:
                    self._fill_tracker.record_outcome(pair.name, filled=False)
                return f"skipped: entry REST recheck missing quote ({pair.name})"
            if rechecked_obs.net_total_cost >= self.settings.entry_threshold:
                if self.settings.fill_rate_tracking:
                    self._fill_tracker.record_outcome(pair.name, filled=False)
                return f"skipped: entry REST recheck no longer below threshold ({pair.name})"
            obs = rechecked_obs
            books = rest_books

        max_capital = min(
            _pair_max_trade_size(pair, self.settings.max_trade_size),
            self.cash * self.settings.capital_fraction_per_trade,
            self.cash,
        )
        if self.settings.max_total_locked_capital is not None:
            locked_now = sum(pos.locked_capital for pos in self.positions if pos.status == "open")
            remaining_lockable = self.settings.max_total_locked_capital - locked_now
            if remaining_lockable <= 0:
                return f"skipped: max locked capital reached ({pair.name})"
            max_capital = min(max_capital, remaining_lockable)
        if max_capital < self.settings.min_trade_size:
            return f"skipped: insufficient cash ({pair.name})"
        # ── Time-to-resolution sizing multiplier ──────────────────────────────
        # Scale max_capital up for fast-resolving markets, down for long-dated.
        # sqrt(target_days / days_to_resolution): 10-day market at target=30
        # gets ×1.73; 90-day market gets ×0.58.  No-op when date is unknown.
        if self.settings.resolution_time_sizing:
            _days = pair_days_to_resolution(pair.raw)
            _time_mult = resolution_size_multiplier(
                _days,
                target_days=self.settings.resolution_target_days,
                max_multiplier=self.settings.resolution_max_multiplier,
                min_multiplier=self.settings.resolution_min_multiplier,
            )
            if _time_mult != 1.0:
                max_capital = min(max_capital * _time_mult, self._effective_cash())
        entry_parent_price = obs.parent_yes_ask
        entry_child_price = obs.child_no_ask
        entry_net_cost = obs.net_total_cost
        leg_1_token, leg_2_token = _pair_leg_token_ids(pair)
        parent_book = books.get(leg_1_token or "")
        child_book = books.get(leg_2_token or "")
        if (parent_book is not None and _book_depth_untrusted(parent_book)) or (
            child_book is not None and _book_depth_untrusted(child_book)
        ):
            return f"skipped: untrusted websocket depth requires REST recheck ({pair.name})"
        if self.settings.sizing_mode == "max_profit":
            optimal = self._find_best_size(parent_book, child_book, pair, max_capital=max_capital, require_entry=True)
            if not optimal:
                return f"skipped: no profitable depth-aware size ({pair.name})"
            size = optimal["size"]
            locked_capital = optimal["capital"]
            entry_parent_price = optimal["parent_avg"]
            entry_child_price = optimal["child_avg"]
            entry_net_cost = optimal["net"]
        else:
            intended_size = max_capital / obs.net_total_cost
            size = min(intended_size, obs.max_executable_size)
            locked_capital = size * obs.net_total_cost
        if locked_capital < self.settings.min_trade_size:
            if self.settings.fill_rate_tracking:
                self._fill_tracker.record_outcome(pair.name, filled=False)
            return f"skipped: insufficient executable depth ({pair.name})"
        if entry_net_cost >= self.settings.entry_threshold:
            if self.settings.fill_rate_tracking:
                self._fill_tracker.record_outcome(pair.name, filled=False)
            return f"skipped: no longer below entry threshold ({pair.name})"
        if (1.0 - entry_net_cost) < _pair_min_edge_threshold(pair, self.settings.min_edge_threshold):
            return f"skipped: below min edge threshold ({pair.name})"
        roi = (1.0 - entry_net_cost) / max(entry_net_cost, 1e-9)
        if roi < self.settings.min_roi_threshold:
            return f"skipped: below min ROI threshold ({pair.name})"

        worst = (1.0 - entry_net_cost) * size
        best = (2.0 - entry_net_cost) * size
        if pair.boundary_ambiguity and not self.settings.allow_boundary_ambiguous_guaranteed:
            trade_type = "boundary_ambiguous_candidate"
        else:
            trade_type = "guaranteed_arbitrage" if worst > 0 else "convergence_trade"
        if self._live_mode:
            live_result = await self._execute_live_buy_bundle(
                timestamp=timestamp,
                strategy_name=pair.name,
                legs=[
                    (pair.parent_outcome_label or "leg_1", leg_1_token or "", parent_book),
                    (pair.child_outcome_label or "leg_2", leg_2_token or "", child_book),
                ],
                size=size,
                guaranteed_payout=1.0,
            )
            if not live_result.success:
                return f"live skipped: {live_result.error or 'order rejected'} ({pair.name})"
            trade_type = f"live_{trade_type}"
        position = PaperPosition(
            trade_id=self._next_trade_id,
            entry_time=timestamp,
            status="open",
            pair_name=pair.name,
            relation_subtype=pair.relation_subtype,
            entry_trade_type=trade_type,
            size=size,
            parent_outcome_label=pair.parent_outcome_label,
            child_outcome_label=pair.child_outcome_label,
            parent_yes_entry_price=entry_parent_price,
            child_no_entry_price=entry_child_price,
            gross_total_cost_per_unit=obs.gross_total_cost or 0.0,
            net_total_cost_per_unit=entry_net_cost,
            locked_capital=locked_capital,
            worst_case_profit=worst,
            best_case_profit=best,
            event_date=_optional_str(pair.overrides.get("event_date")) or _optional_str(pair.raw.get("event_date")),
        )
        self._next_trade_id += 1
        self.cash -= locked_capital
        self.positions.append(position)
        self._register_open_position(position)
        self.last_trade_time_by_pair[pair.name] = time.monotonic()
        _append_csv_row(self.settings.trades_out, position)
        if self.settings.fill_rate_tracking:
            self._fill_tracker.record_outcome(pair.name, filled=True)
        return f"entered {trade_type} {pair.name} size={size:.4f} locked={locked_capital:.4f}"

    async def _execute_live_buy_bundle(
        self,
        *,
        timestamp: str,
        strategy_name: str,
        legs: list[tuple[str, str, OrderBook | None]],
        size: float,
        guaranteed_payout: float,
    ) -> LiveOrderResult:
        if self._live_trader is None:
            return LiveOrderResult(success=False, requested_notional=0.0, error="live trader is not initialized")
        if self._clob_client is None:
            return LiveOrderResult(success=False, requested_notional=0.0, error="CLOB REST client is not initialized")

        token_ids = [token_id for _, token_id, _ in legs if token_id]
        batch = await self._clob_client.get_order_books(
            token_ids,
            max_concurrent_requests=self.settings.max_concurrent_requests,
        )
        live_legs: list[LiveOrderLeg] = []
        for label, token_id, _book in legs:
            book = batch.books.get(token_id)
            if book is None:
                result = LiveOrderResult(success=False, requested_notional=0.0, error=f"live REST recheck missing book for {label}")
                self._log_live_order(timestamp, strategy_name, result, len(legs))
                return result
            min_order_size = _book_min_order_size(book)
            if size < min_order_size:
                result = LiveOrderResult(
                    success=False,
                    requested_notional=0.0,
                    error=f"live size {size:.4f} below min order size {min_order_size:.4f} for {label}",
                )
                self._log_live_order(timestamp, strategy_name, result, len(legs))
                return result
            marginal_price = _book_marginal_ask_price_for_size(book, size)
            if marginal_price is None:
                result = LiveOrderResult(success=False, requested_notional=0.0, error=f"insufficient live REST ask depth for {label}")
                self._log_live_order(timestamp, strategy_name, result, len(legs))
                return result
            tick_size = _book_tick_size(book)
            limit_price = _round_price_up_to_tick(
                marginal_price + (max(0, self.settings.live_price_buffer_ticks) * float(tick_size)),
                tick_size,
            )
            if limit_price <= 0 or limit_price >= 1:
                result = LiveOrderResult(success=False, requested_notional=0.0, error=f"invalid live limit price {limit_price:.4f} for {label}")
                self._log_live_order(timestamp, strategy_name, result, len(legs))
                return result
            live_legs.append(
                LiveOrderLeg(
                    label=label,
                    token_id=token_id,
                    size=size,
                    limit_price=limit_price,
                    tick_size=_book_tick_size_str(book),
                    neg_risk=_book_neg_risk(book),
                )
            )

        fee_rates = await self._live_fee_rates([leg.token_id for leg in live_legs])
        unit_cost_with_fees = sum(
            leg.limit_price + estimated_fee_per_unit(leg.limit_price, fee_rates.get(leg.token_id, 0.0))
            for leg in live_legs
        )
        min_edge = _live_min_edge_threshold(self.settings.min_edge_threshold)
        if guaranteed_payout - unit_cost_with_fees < min_edge:
            result = LiveOrderResult(
                success=False,
                requested_notional=sum(leg.notional_cap for leg in live_legs),
                error=(
                    "live REST recheck below edge after limit prices/fees "
                    f"(edge={guaranteed_payout - unit_cost_with_fees:.6f})"
                ),
            )
            self._log_live_order(timestamp, strategy_name, result, len(live_legs), live_legs=live_legs)
            return result

        # Cache legs for background pre-signing of the next likely entry
        self._live_presign_queue = [
            (leg.token_id, leg.limit_price, leg.size, leg.tick_size, leg.neg_risk)
            for leg in live_legs
        ]
        result = await self._live_trader.buy_bundle_fok(live_legs)
        # Live order latency legs: signing (secp256k1, cache-dependent) and the
        # submission → ack network round-trip.  Both already timed in live_trader.
        if profiling.ENABLED and (_rec := profiling.RECORDER) is not None:
            if result.submission_ms is not None:
                _rec.record_series("order_rtt", result.submission_ms * 1e6)
            if result.signing_ms is not None:
                _rec.record_series("signing_latency", result.signing_ms * 1e6)
        self._log_live_order(timestamp, strategy_name, result, len(live_legs), live_legs=live_legs)
        return result

    async def _live_fee_rates(self, token_ids: list[str]) -> dict[str, float]:
        if self.settings.fee_rate is not None:
            return {token_id: self.settings.fee_rate for token_id in token_ids}
        missing = [token_id for token_id in token_ids if token_id not in self._token_fee_rate_cache]
        if missing and self.settings.dynamic_fee_rates and self._clob_client is not None:
            rates = await self._clob_client.get_fee_rates(missing, max_concurrent_requests=self.settings.max_concurrent_requests)
            for token_id, rate in rates.items():
                if rate is not None:
                    self._token_fee_rate_cache[token_id] = rate
        fallback = DEFAULT_TAKER_FEE_RATES["crypto"]
        return {token_id: self._token_fee_rate_cache.get(token_id, fallback) for token_id in token_ids}

    def _log_live_order(
        self,
        timestamp: str,
        strategy_name: str,
        result: LiveOrderResult,
        leg_count: int,
        *,
        live_legs: list[LiveOrderLeg] | None = None,
    ) -> None:
        if not self._live_mode:
            return
        _append_csv_row(
            self.settings.live_orders_out,
            LiveOrderLogRow(
                timestamp=timestamp,
                strategy_name=strategy_name,
                leg_count=leg_count,
                requested_notional=result.requested_notional,
                success=result.success,
                error=result.error,
                responses_json=json.dumps(_json_safe(result.responses), sort_keys=True),
            ),
        )
        if self._audit.enabled:
            ctx = self._audit.current_context()
            order_id = self._audit.next_id("order")
            self._audit.record(
                "orders",
                {
                    "order_id": order_id,
                    "decision_id": ctx.get("decision_id"),
                    "scan_id": ctx.get("scan_id"),
                    "opportunity_id": ctx.get("opportunity_id"),
                    "strategy_name": strategy_name,
                    "leg_count": leg_count,
                    "token_ids_json": json.dumps([leg.token_id for leg in live_legs or []]),
                    "requested_size": live_legs[0].size if live_legs else None,
                    "requested_notional": result.requested_notional,
                    "limit_prices_json": json.dumps([leg.limit_price for leg in live_legs or []]),
                    "order_type": "FOK",
                    "signing_started_ns": result.signing_started_ns,
                    "submission_started_ns": result.submission_started_ns,
                    "ack_received_ns": result.ack_received_ns,
                    "signing_ms": result.signing_ms,
                    "submission_ms": result.submission_ms,
                    "success": result.success,
                    "error": result.error,
                    "responses_json": json.dumps(_json_safe(result.responses), sort_keys=True),
                },
            )
            self._audit_record_timeline(
                event_kind="order_ack",
                scan_id=ctx.get("scan_id"),
                decision_id=ctx.get("decision_id"),
                opportunity_id=ctx.get("opportunity_id"),
                candidate_name=strategy_name,
                timestamp_ns=result.ack_received_ns or time.time_ns(),
                metric_1=result.requested_notional,
                metric_2=result.submission_ms,
                message="success" if result.success else result.error,
            )

    def _find_best_size(
        self,
        parent_book: OrderBook | None,
        child_book: OrderBook | None,
        pair: PairConfig,
        *,
        max_capital: float | None = None,
        require_entry: bool = True,
    ) -> dict[str, float]:
        if parent_book is None or child_book is None or not parent_book.asks or not child_book.asks:
            return {}
        if _book_depth_untrusted(parent_book) or _book_depth_untrusted(child_book):
            return {}
        slippage = self._pair_slippage_cache.get(pair.name)
        if slippage is None:
            slippage = _pair_slippage(pair, self.settings.slippage_buffer)
            self._pair_slippage_cache[pair.name] = slippage
        fee_rates = self._pair_fee_rate_cache.get(pair.name)
        if fee_rates is None:
            fee_rates = _pair_fee_rates(pair, self.settings.fee_rate)
            self._pair_fee_rate_cache[pair.name] = fee_rates
        fee_buffer = _pair_fee_buffer(pair)
        capital_limit = max_capital
        if capital_limit is None:
            open_positions = [pos for pos in self.positions if pos.status == "open"]
            capital_limit = min(
                _pair_max_trade_size(pair, self.settings.max_trade_size),
                self.cash * self.settings.capital_fraction_per_trade,
                self.cash,
            )
            if self.settings.max_total_locked_capital is not None:
                capital_limit = min(
                    capital_limit,
                    max(0.0, self.settings.max_total_locked_capital - sum(pos.locked_capital for pos in open_positions)),
                )
        if capital_limit <= 0:
            return {}

        max_depth = float(min(parent_book.ask_depth, child_book.ask_depth))
        if max_depth <= 0:
            return {}
        parent_cum = _cumulative_sizes(parent_book.asks)
        child_cum = _cumulative_sizes(child_book.asks)
        parent_float_asks = [(float(level.price), float(level.size)) for level in parent_book.asks]
        child_float_asks = [(float(level.price), float(level.size)) for level in child_book.asks]
        rough_top_net = (
            float(parent_book.best_ask or 0)
            + float(child_book.best_ask or 0)
            + estimated_fee_per_unit(float(parent_book.best_ask or 0), fee_rates[0])
            + estimated_fee_per_unit(float(child_book.best_ask or 0), fee_rates[1])
            + slippage
            + fee_buffer
        )
        cap_size = capital_limit / max(rough_top_net, 0.01)
        candidates = {size for size in parent_cum + child_cum if 0 < size <= max_depth}
        candidates.add(min(max_depth, cap_size))
        # Add a small grid so we catch capital limits that fall inside a book level.
        upper = min(max_depth, max(cap_size * 1.25, self.settings.min_trade_size))
        for index in range(1, 51):
            candidates.add(upper * index / 50)

        # Use the fast optimizer (numpy/numba) when require_entry=True (hot path).
        # The fast path applies all the same filters inside a compiled loop.
        if require_entry:
            return _fast_optimizer(
                parent_float_asks,
                child_float_asks,
                fee_rates[0],
                fee_rates[1],
                slippage,
                fee_buffer,
                self.settings.entry_threshold,
                _pair_min_edge_threshold(pair, self.settings.min_edge_threshold),
                capital_limit,
                self.settings.min_trade_size,
                candidates,
            )

        # Non-entry path (e.g. N-leg watchlist sizing): use Python loop so we
        # can omit the entry_threshold / min_edge guards when require_entry=False.
        best: dict[str, float] = {}
        for size in sorted(candidates):
            if size <= 0:
                continue
            parent_price = _float_avg_price_for_size(parent_float_asks, size)
            child_price = _float_avg_price_for_size(child_float_asks, size)
            if parent_price is None or child_price is None:
                continue
            fee = estimated_fee_per_unit(parent_price, fee_rates[0]) + estimated_fee_per_unit(child_price, fee_rates[1])
            net = parent_price + child_price + fee + slippage + fee_buffer
            capital = size * net
            profit = size * (1.0 - net)
            if size < self.settings.min_trade_size / max(net, 0.01):
                continue
            if capital > capital_limit + 1e-9:
                continue
            if profit <= 0:
                continue
            if not best or profit > best["profit"]:
                best = {
                    "size": size,
                    "capital": capital,
                    "profit": profit,
                    "net": net,
                    "parent_avg": parent_price,
                    "child_avg": child_price,
                }
        return best

    def _mark_positions(self, timestamp: str, observations: list[PairObservation]) -> None:
        by_pair = {obs.pair_name: obs for obs in observations}
        now = datetime.fromisoformat(timestamp)
        for position in self.positions:
            if position.status != "open":
                continue
            obs = by_pair.get(position.pair_name)
            if obs and obs.parent_yes_bid is not None and obs.child_no_bid is not None:
                gross_exit_value = (obs.parent_yes_bid + obs.child_no_bid) * position.size
                fee_rate_1, fee_rate_2 = self._pair_fee_rate_cache.get(position.pair_name, (0.0, 0.0))
                exit_fee_per_share = estimated_fee_per_unit(obs.parent_yes_bid, fee_rate_1) + estimated_fee_per_unit(
                    obs.child_no_bid, fee_rate_2
                )
                exit_fee_total = exit_fee_per_share * position.size
                net_exit_value = gross_exit_value - exit_fee_total
                position.exit_parent_yes_bid = obs.parent_yes_bid
                position.exit_child_no_bid = obs.child_no_bid
                position.exit_total_value = gross_exit_value
                position.liquidation_value_gross = gross_exit_value
                position.exit_fee_total = exit_fee_total
                position.liquidation_value_net = net_exit_value
                position.liquidation_pnl = net_exit_value - position.locked_capital
                position.mtm_value = net_exit_value
                position.unrealized_pnl = position.liquidation_pnl
                if self.settings.exit_mode == "close_when_edge_converges":
                    self._maybe_close_position(position, timestamp)
            entry = datetime.fromisoformat(position.entry_time)
            position.hold_minutes = max(0.0, (now - entry).total_seconds() / 60)

    def _maybe_close_position(self, position: PaperPosition, timestamp: str) -> None:
        if position.status != "open" or position.liquidation_value_net is None:
            return
        pnl = position.liquidation_pnl
        pnl_pct = pnl / position.locked_capital if position.locked_capital else 0.0
        reason: str | None = None
        if self.settings.take_profit_absolute is not None and pnl >= self.settings.take_profit_absolute:
            reason = "take_profit_absolute"
        elif pnl_pct >= self.settings.take_profit_pct:
            reason = "take_profit_pct"
        elif self.settings.stop_loss_absolute is not None and pnl <= -self.settings.stop_loss_absolute:
            reason = "stop_loss_absolute"
        elif pnl_pct <= -self.settings.stop_loss_pct:
            reason = "stop_loss_pct"
        if reason is None:
            return
        position.status = "closed"
        position.exit_time = timestamp
        position.exit_reason = reason
        position.realized_pnl = pnl
        position.unrealized_pnl = 0.0
        self.cash += position.liquidation_value_net
        self._register_closed_position(position)

    def _register_open_position(self, position: PaperPosition) -> None:
        self.open_pair_names.add(position.pair_name)
        self.open_count_by_pair[position.pair_name] = self.open_count_by_pair.get(position.pair_name, 0) + 1

    def _register_closed_position(self, position: PaperPosition) -> None:
        current = self.open_count_by_pair.get(position.pair_name, 0)
        if current <= 1:
            self.open_count_by_pair.pop(position.pair_name, None)
            self.open_pair_names.discard(position.pair_name)
        else:
            self.open_count_by_pair[position.pair_name] = current - 1

    def _print_startup(self, pairs: list[PairConfig]) -> None:
        if self.settings.headless:
            return
        mode_line = (
            "LIVE TRADING MODE ENABLED. Real orders may be placed."
            if self._live_mode
            else "Paper-only clean simulator starting. No real orders can be placed."
        )
        msg = (
            f"{mode_line}\n"
            f"Pairs loaded: {len(pairs)} | Budget: {self.settings.budget:.2f} | "
            f"Entry threshold: {self.settings.entry_threshold:.4f} | "
            f"Min edge: {_fmt(self.settings.min_edge_threshold)} | Near threshold: {self.settings.near_arb_threshold:.4f}\n"
            f"Book source: {self.settings.book_source} | Sizing: {self.settings.sizing_mode} | Exit: {self.settings.exit_mode}"
        )
        if self.settings.live_universe:
            print(
                f"Live universe mode: continuously discovering {self.settings.live_universe_assets} pairs across "
                f"{self.settings.live_universe_horizon_days} day(s) ahead, refreshing every "
                f"{self.settings.live_universe_refresh_seconds:.0f}s."
            )
        venue_name = "Poly" if self.settings.exchange == "polymarket" else "Kalshi"
        if self.console and Panel:
            self.console.print(Panel(msg, title=f"Clean {venue_name} Paper Bot"))
        else:
            print(msg)
        if self.settings.fee_rate == 0.0:
            print("WARNING: fees are forced to zero by --fee-rate 0; PnL may be overstated.")
        elif self.settings.fee_rate is not None:
            print(f"Fee mode: using explicit fee rate {self.settings.fee_rate}.")
        elif self.settings.exchange == "kalshi":
            print("Fee mode: Kalshi dynamic fee lookup is not configured; using YAML/category conservative estimates.")
        elif self.settings.dynamic_fee_rates:
            print("Fee mode: dynamically fetching token fee rates from public CLOB /fee-rate with conservative fallback.")
        elif self.settings.fee_rate is None:
            print("Fee mode: using pair/default conservative taker fee estimates where configured.")
        if self._audit.enabled:
            print(f"Forensic audit: writing partitioned Parquet datasets to {self._audit.run_dir}")

    def _start_live_dashboard(self) -> None:
        if (
            self.settings.live_dashboard
            and not self.settings.headless
            and not self.settings.once
            and self.console is not None
            and Live is not None
        ):
            self._live = Live(
                self._dashboard_renderable(None),
                console=self.console,
                refresh_per_second=8,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()

    def _stop_live_dashboard(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._dashboard_line_count = 0

    def _dashboard_renderable(self, row: ScanRow | None) -> Any:
        if row is None:
            text = "Starting live paper simulator...\nWaiting for first market-data scan."
            return Panel(text, title="Paper Arb Sim") if Panel else text
        best_line = "Best pair: none"
        if row.best_pair_name:
            best_line = (
                f"Best pair: {row.best_pair_name}\n"
                f"Gross cost: {_fmt(row.best_total_cost)} | Net cost: {_fmt(row.net_total_cost)} | "
                f"Entry threshold: {row.entry_threshold:.4f} | Distance: {_fmt(row.distance_to_entry)}\n"
                f"Worst/unit: {_fmt(row.best_worst_case_profit)} | "
                f"Opt size: {_fmt(row.best_optimal_size)} | Opt profit: {_fmt(row.best_optimal_guaranteed_profit)}"
            )
        n_leg_line = "Best N-leg: none"
        if row.best_n_leg_name:
            n_leg_line = (
                f"Best N-leg ({row.best_n_leg_leg_count} legs): {row.best_n_leg_name}\n"
                f"Gross cost: {_fmt(row.best_n_leg_gross_cost)} | "
                f"Guaranteed payout: {_fmt(row.best_n_leg_guaranteed_payout)} | "
                f"Gross edge: {_fmt(row.best_n_leg_gross_edge)}"
            )
        text = (
            f"{row.timestamp}\n"
            f"Cash: {row.cash_available:.4f} | Locked: {row.locked_capital:.4f} | Open: {row.open_positions_count}\n"
            f"Liquidation PnL: {row.liquidation_pnl:.4f} | Guaranteed hold PnL: {row.guaranteed_profit_if_held:.4f}\n"
            f"Executable: {row.executable_candidates_count} | Near: {row.near_arb_candidates_count} | "
            f"Rejected: {row.rejected_count} | Missing books: {row.books_missing_count} | Missing asks: {row.asks_missing_count}\n"
            f"{best_line}\n"
            f"{n_leg_line}\n"
            f"Action: {row.action_taken}\n"
            f"Scan: {row.scan_time_ms:.0f} ms | Source: {row.book_source} | Tokens: {row.unique_tokens} "
            f"| fetched={row.unique_tokens_fetched} cache_hits={row.cache_hits}\n"
            f"WS: connected={row.websocket_connected} fallback={row.fallback_to_polling_used} reconnects={row.websocket_reconnect_count}\n"
            f"CSV: {self.settings.out} | Report: {self.settings.save_markdown}"
        )
        if self._latest_top_lines:
            text += "\nTop pairs:\n" + "\n".join(self._latest_top_lines)
        return Panel(text, title="Paper Arb Sim") if Panel else text

    def _best_three_leg_opportunity(self, pairs: list[PairConfig], books: dict[str, OrderBook]) -> dict[str, Any]:
        specs = _derive_range_threshold_n_leg_specs(pairs, max_ranges=1)
        best = self._best_n_leg_opportunity(specs, books)
        return best if best.get("leg_count") == 3 else {}

    def _best_n_leg_opportunity(
        self, specs: list[NLegOpportunitySpec], books: dict[str, OrderBook]
    ) -> dict[str, Any]:
        """Find the best N-leg opportunity by gross ROI.

        Uses a pre-allocated numpy matrix for vectorised scoring when the scan
        plan's cached ``_n_leg_arrays`` matches the provided specs.  Falls back
        to building a temporary matrix for direct calls (e.g., from tests).

        **Performance contract**: no depth optimizer runs here.  Only a fast
        best-ask lookup + vectorised arithmetic.  _try_enter_n_leg runs the
        expensive VWAP optimizer at most once per scan cycle.
        """
        if not specs:
            return {}
        # Use cached arrays when specs match the active plan (hot path)
        arrays = self._n_leg_arrays
        if arrays is not None and arrays.spec_list == specs:
            return self._best_n_leg_vectorized(arrays, books)
        # Build temporary arrays for direct / test calls (cold path)
        return self._best_n_leg_vectorized(
            _NLegScanArrays(specs, self.settings.min_trade_size), books
        )

    def _best_n_leg_vectorized(
        self, arrays: _NLegScanArrays, books: dict[str, OrderBook]
    ) -> dict[str, Any]:
        """Vectorised N-leg scan using pre-allocated numpy arrays.

        Step 1 — fill ask/depth matrices (Python loop, unavoidable dict access).
        Step 2 — numpy arithmetic: has_all, gross_cost, min_depth, valid mask.
        Step 3 — argmax over valid ROIs → single best candidate.

        Complexity: O(n_specs × n_legs) dict lookups + O(n_specs) numpy ops.
        For 100 specs the numpy step is ~50× faster than the equivalent Python
        arithmetic loop.
        """
        n = arrays.n_specs
        if n == 0:
            return {}

        am = arrays.ask_matrix
        dm = arrays.depth_matrix

        # ── Step 1: fill dynamic price/depth matrices ──────────────────────────
        for spec_i, token_ids in enumerate(arrays.token_id_rows):
            n_legs = len(token_ids)
            for leg_j, token_id in enumerate(token_ids):
                book = books.get(token_id)
                if book is not None:
                    ba = book.best_ask
                    if ba is not None:
                        am[spec_i, leg_j] = ba
                        dm[spec_i, leg_j] = book.ask_depth
                    else:
                        am[spec_i, leg_j] = _np.nan
                        dm[spec_i, leg_j] = 0.0
                else:
                    am[spec_i, leg_j] = _np.nan
                    dm[spec_i, leg_j] = 0.0
            # Zero-pad unused leg columns
            for leg_j in range(n_legs, arrays.max_legs):
                am[spec_i, leg_j] = 0.0   # not NaN — nansum shouldn't see these
                dm[spec_i, leg_j] = 1e9   # large depth so min doesn't kill valid specs

        # ── Step 2: vectorised filtering and scoring ──────────────────────────
        # Mask specs where any leg is missing a book / best_ask
        has_all = ~_np.any(_np.isnan(am[:, :]), axis=1)  # (n,) bool

        # Gross cost = sum of best asks over all legs (NaN = missing → handled by has_all)
        gross_cost = _np.nansum(am, axis=1)                  # (n,) float64

        # Minimum depth across all legs (only real legs, not padded zeros)
        min_depth = dm.min(axis=1)                            # (n,) float64
        depth_ok = min_depth >= arrays.min_fill_vec           # (n,) bool

        gross_edge = arrays.payout_vec - gross_cost           # (n,) float64
        valid = has_all & depth_ok & (gross_edge > 0.0)       # (n,) bool

        if not valid.any():
            return {}

        gross_roi = gross_edge / _np.maximum(gross_cost, 1e-9)
        best_idx = int(_np.argmax(_np.where(valid, gross_roi, -_np.inf)))

        if not valid[best_idx]:
            return {}

        spec = arrays.spec_list[best_idx]
        gc = float(gross_cost[best_idx])
        ge = float(gross_edge[best_idx])
        return {
            "name": spec.name,
            "legs": " + ".join(leg.label for leg in spec.legs),
            "leg_labels": [leg.label for leg in spec.legs],
            "leg_token_ids": [leg.token_id for leg in spec.legs],
            "leg_count": len(spec.legs),
            "relation_subtype": spec.relation_subtype,
            "guaranteed_payout": spec.guaranteed_payout,
            "event_date": spec.event_date,  # from main: needed by the N-day live universe
            "gross_cost": gc,
            "gross_edge": ge,
            "score_edge": float(gross_roi[best_idx]),
        }

    def _try_enter_three_leg(self, best_three_leg: dict[str, float | str]) -> str:
        return "skipped: use _try_enter_n_leg async path"

    async def _try_enter_n_leg(
        self,
        best_n_leg: dict[str, Any],
        books: dict[str, OrderBook],
        _prefetched_rest_books: dict[str, OrderBook] | None = None,
    ) -> str:
        name = str(best_n_leg.get("name") or "")
        gross_cost = float(best_n_leg.get("gross_cost") or 0.0)
        gross_edge = float(best_n_leg.get("gross_edge") or 0.0)
        guaranteed_payout = float(best_n_leg.get("guaranteed_payout") or 0.0)
        leg_count = int(best_n_leg.get("leg_count") or 0)
        leg_labels = [str(label) for label in best_n_leg.get("leg_labels") or []]
        leg_token_ids = [str(token_id) for token_id in best_n_leg.get("leg_token_ids") or []]
        if not name or leg_count < 3 or gross_cost <= 0 or guaranteed_payout <= 0 or gross_edge <= 0:
            return "none"
        if len(leg_token_ids) != leg_count or len(leg_labels) != leg_count or any(not token for token in leg_token_ids):
            return f"skipped: incomplete N-leg token map ({name})"
        # Track whether this is an incremental top-up of an existing position.
        _incremental_mode = name in self._open_n_leg_names
        if _incremental_mode and not self.settings.allow_incremental_n_leg:
            return f"skipped: duplicate open N-leg prevented ({name})"
        # For incremental entries, note the existing total size so we can
        # compute how much new depth this entry would add.
        _existing_size: float = (
            sum(p.size for p in self.positions if p.status == "open" and p.pair_name == f"N_LEG::{name}")
            if _incremental_mode
            else 0.0
        )
        max_capital = self._n_leg_capital_limit()
        if max_capital < self.settings.min_trade_size:
            return f"skipped: insufficient N-leg cash/capital ({name})"

        entry_books = books
        leg_books = [entry_books.get(token_id) for token_id in leg_token_ids]
        if self.settings.entry_rest_recheck and self._clob_client is not None:
            if _prefetched_rest_books is not None:
                entry_books = {**books, **_prefetched_rest_books}
            else:
                batch = await self._clob_client.get_order_books(
                    leg_token_ids,
                    max_concurrent_requests=self.settings.max_concurrent_requests,
                )
                entry_books = {**books, **batch.books}
            leg_books = [entry_books.get(token_id) for token_id in leg_token_ids]
            if any(book is None or book.best_ask is None for book in leg_books):
                if self.settings.fill_rate_tracking:
                    self._fill_tracker.record_outcome(name, filled=False)
                return f"skipped: N-leg entry REST recheck missing quote ({name})"
            typed_books = [book for book in leg_books if book is not None]
            gross_cost = sum(float(book.best_ask or 0.0) for book in typed_books)
            gross_edge = guaranteed_payout - gross_cost
            if gross_cost <= 0 or gross_edge <= 0:
                if self.settings.fill_rate_tracking:
                    self._fill_tracker.record_outcome(name, filled=False)
                return f"skipped: N-leg entry REST recheck no longer positive ({name})"
        else:
            if any(book is None or book.best_ask is None for book in leg_books):
                return f"skipped: missing N-leg quote ({name})"
            typed_books = [book for book in leg_books if book is not None]

        if any(_book_depth_untrusted(book) for book in typed_books):
            return f"skipped: untrusted websocket N-leg depth requires REST recheck ({name})"

        sizing_mode = self._n_leg_sizing_mode()
        if sizing_mode == "optimized":
            optimal = self._find_best_n_leg_size(typed_books, guaranteed_payout)
            if not optimal:
                return f"skipped: no profitable depth-aware N-leg size ({name})"
            if float(optimal.get("roi") or 0.0) < self.settings.min_roi_threshold:
                return f"skipped: below min ROI threshold ({name})"
            suggested_size = float(optimal.get("size") or 0.0)
            suggested_capital = float(optimal.get("capital") or 0.0)
            suggested_profit = float(optimal.get("profit") or 0.0)
            scale = min(1.0, max_capital / suggested_capital)
            size = suggested_size * scale
            locked_capital = suggested_capital * scale
            worst = suggested_profit * scale
        elif sizing_mode == "max_trade":
            max_spend = self._find_n_leg_max_spend_size(typed_books, guaranteed_payout)
            if not max_spend:
                return f"skipped: no depth-aware max-spend N-leg size ({name})"
            if float(max_spend.get("roi") or 0.0) < self.settings.min_roi_threshold:
                return f"skipped: below min ROI threshold ({name})"
            max_spend_size = float(max_spend.get("size") or 0.0)
            max_spend_capital = float(max_spend.get("capital") or 0.0)
            max_spend_profit = float(max_spend.get("profit") or 0.0)
            scale = min(1.0, max_capital / max_spend_capital)
            size = max_spend_size * scale
            locked_capital = max_spend_capital * scale
            worst = max_spend_profit * scale
        else:
            return f"skipped: unsupported N-leg sizing mode {sizing_mode} ({name})"
        if locked_capital < self.settings.min_trade_size or worst <= 0:
            return f"skipped: insufficient profitable N-leg depth ({name})"

        # ── Incremental entry check ───────────────────────────────────────────
        if _incremental_mode:
            # `size` is the depth-optimizer's recommended TOTAL position size.
            # The incremental portion is the new depth on top of what we hold.
            min_increment = self.settings.min_trade_size / max(guaranteed_payout, 1.0)
            incremental_size = size - _existing_size
            if incremental_size < min_increment:
                return (
                    f"skipped: no incremental N-leg depth beyond existing "
                    f"{_existing_size:.2f} shares (optimal={size:.2f}) ({name})"
                )
            # Per-unit cost derived from the full-book VWAP (slightly optimistic
            # because we've "used" the cheapest depth, but acceptable for paper).
            unit_cost = locked_capital / max(size, 1e-9)
            incremental_capital = incremental_size * unit_cost
            incremental_profit = incremental_size * (guaranteed_payout - unit_cost)
            if incremental_profit < self.settings.incremental_n_leg_min_profit:
                return (
                    f"skipped: incremental N-leg profit {incremental_profit:.4f} "
                    f"below min {self.settings.incremental_n_leg_min_profit:.4f} ({name})"
                )
            if incremental_capital > max_capital:
                incremental_capital = max_capital
                incremental_size = incremental_capital / max(unit_cost, 1e-9)
                incremental_profit = incremental_size * (guaranteed_payout - unit_cost)
            size = incremental_size
            locked_capital = incremental_capital
            worst = incremental_profit

        if self._live_mode:
            live_result = await self._execute_live_buy_bundle(
                timestamp=now_iso(),
                strategy_name=name,
                legs=[
                    (label, token_id, entry_books.get(token_id))
                    for label, token_id in zip(leg_labels, leg_token_ids, strict=False)
                ],
                size=size,
                guaranteed_payout=guaranteed_payout,
            )
            if not live_result.success:
                return f"live skipped: {live_result.error or 'order rejected'} ({name})"
        position = PaperPosition(
            trade_id=self._next_trade_id,
            entry_time=now_iso(),
            status="open",
            pair_name=f"N_LEG::{name}",
            relation_subtype=str(best_n_leg.get("relation_subtype") or "n_leg_range_threshold"),
            entry_trade_type="live_guaranteed_arbitrage" if self._live_mode else "guaranteed_arbitrage",
            size=size,
            parent_outcome_label=f"{leg_count}_LEG",
            child_outcome_label=str(best_n_leg.get("legs") or name),
            parent_yes_entry_price=gross_cost,
            child_no_entry_price=0.0,
            gross_total_cost_per_unit=gross_cost,
            net_total_cost_per_unit=gross_cost,
            locked_capital=locked_capital,
            worst_case_profit=worst,
            best_case_profit=worst,
            event_date=_optional_str(best_n_leg.get("event_date")),
        )
        self._next_trade_id += 1
        self.cash -= locked_capital
        self.positions.append(position)
        self._open_n_leg_names.add(name)
        _append_csv_row(self.settings.trades_out, position)
        if self.settings.fill_rate_tracking:
            self._fill_tracker.record_outcome(name, filled=True)
        entry_kind = "incremental" if _incremental_mode else "entered"
        return f"{entry_kind} {leg_count}-leg {name} size={size:.4f} locked={locked_capital:.4f}"

    def _find_best_three_leg_size(
        self, low_yes_book: OrderBook, high_no_book: OrderBook, range_no_book: OrderBook
    ) -> dict[str, float]:
        return self._find_best_n_leg_size([low_yes_book, high_no_book, range_no_book], guaranteed_payout=2.0)

    def _find_best_n_leg_size(self, leg_books: list[OrderBook], guaranteed_payout: float) -> dict[str, float]:
        if len(leg_books) < 3 or any(not book.asks or _book_depth_untrusted(book) for book in leg_books):
            return {}
        max_capital = self._n_leg_capital_limit()
        if max_capital <= 0:
            return {}
        max_depth = float(min(book.ask_depth for book in leg_books))
        if max_depth <= 0:
            return {}
        candidates: set[float] = set()
        for book in leg_books:
            candidates.update(_cumulative_sizes(book.asks))
        top_unit_cost = sum(float(book.best_ask or 0) for book in leg_books)
        cap_size = max_capital / max(top_unit_cost, 0.01)
        upper = min(max_depth, max(cap_size * 1.25, self.settings.min_trade_size / max(top_unit_cost, 0.01)))
        for idx in range(1, 51):
            candidates.add(upper * idx / 50)
        # Pre-convert to float tuples once — eliminates Decimal(str(size)) in the hot loop
        leg_float_asks = [
            [(float(level.price), float(level.size)) for level in book.asks]
            for book in leg_books
        ]
        best: dict[str, float] = {}
        for size in sorted(candidates):
            if size <= 0 or size > max_depth:
                continue
            prices_f = [_float_avg_price_for_size(asks, size) for asks in leg_float_asks]
            if any(p is None for p in prices_f):
                continue
            unit_cost = sum(prices_f)  # type: ignore[arg-type]
            cost = size * unit_cost
            profit = size * (guaranteed_payout - unit_cost)
            if cost > max_capital + 1e-9 or cost < self.settings.min_trade_size or profit <= 0:
                continue
            roi = profit / max(cost, 1e-9)
            if (
                not best
                or roi > best["roi"] + 1e-12
                or (abs(roi - best["roi"]) <= 1e-12 and profit > best["profit"])
            ):
                best = {"size": size, "capital": cost, "profit": profit, "roi": roi}
        return best

    def _find_three_leg_max_spend_size(
        self, low_yes_book: OrderBook, high_no_book: OrderBook, range_no_book: OrderBook
    ) -> dict[str, float]:
        return self._find_n_leg_max_spend_size([low_yes_book, high_no_book, range_no_book], guaranteed_payout=2.0)

    def _find_n_leg_max_spend_size(self, leg_books: list[OrderBook], guaranteed_payout: float) -> dict[str, float]:
        if len(leg_books) < 3 or any(not book.asks or _book_depth_untrusted(book) for book in leg_books):
            return {}
        max_capital = self._n_leg_capital_limit()
        if max_capital <= 0:
            return {}
        max_depth = float(min(book.ask_depth for book in leg_books))
        if max_depth <= 0:
            return {}
        candidates: set[float] = set()
        for book in leg_books:
            candidates.update(_cumulative_sizes(book.asks))
        top_unit_cost = sum(float(book.best_ask or 0) for book in leg_books)
        cap_size = max_capital / max(top_unit_cost, 0.01)
        upper = min(max_depth, max(cap_size * 1.25, self.settings.min_trade_size / max(top_unit_cost, 0.01)))
        for idx in range(1, 51):
            candidates.add(upper * idx / 50)
        leg_float_asks = [
            [(float(level.price), float(level.size)) for level in book.asks]
            for book in leg_books
        ]
        best: dict[str, float] = {}
        for size in sorted(candidates):
            if size <= 0 or size > max_depth:
                continue
            prices_f = [_float_avg_price_for_size(asks, size) for asks in leg_float_asks]
            if any(p is None for p in prices_f):
                continue
            unit_cost = sum(prices_f)  # type: ignore[arg-type]
            cost = size * unit_cost
            profit = size * (guaranteed_payout - unit_cost)
            if cost > max_capital + 1e-9 or cost < self.settings.min_trade_size or profit <= 0:
                continue
            roi = profit / max(cost, 1e-9)
            if not best or cost > best["capital"]:
                best = {"size": size, "capital": cost, "profit": profit, "roi": roi}
        return best

    def _n_leg_capital_limit(self) -> float:
        max_capital = min(self.settings.max_trade_size, self.cash * self.settings.capital_fraction_per_trade, self.cash)
        if self.settings.max_total_locked_capital is not None:
            locked_now = sum(pos.locked_capital for pos in self.positions if pos.status == "open")
            max_capital = min(max_capital, max(0.0, self.settings.max_total_locked_capital - locked_now))
        return max_capital

    def _n_leg_trading_enabled(self) -> bool:
        return self.settings.enable_n_leg_trading or self.settings.enable_three_leg_trading

    def _n_leg_sizing_mode(self) -> str:
        if self.settings.three_leg_sizing_mode != "optimized":
            return self.settings.three_leg_sizing_mode
        return self.settings.n_leg_sizing_mode

    def _n_leg_max_ranges(self) -> int | None:
        value = self.settings.n_leg_max_ranges
        if value is None or value <= 0:
            return None
        return max(1, value)

    def _audit_record_timeline(
        self,
        *,
        event_kind: str,
        scan_id: str | None = None,
        decision_id: str | None = None,
        opportunity_id: str | None = None,
        token_id: str | None = None,
        candidate_name: str | None = None,
        timestamp_ns: int | None = None,
        perf_ns: int | None = None,
        metric_1: float | None = None,
        metric_2: float | None = None,
        message: str | None = None,
    ) -> None:
        if not self._audit.enabled:
            return
        self._audit.record(
            "timeline_events",
            {
                "timeline_event_id": self._audit.next_id("timeline"),
                "event_kind": event_kind,
                "scan_id": scan_id,
                "decision_id": decision_id,
                "opportunity_id": opportunity_id,
                "token_id": token_id,
                "candidate_name": candidate_name,
                "timestamp_ns": timestamp_ns or time.time_ns(),
                "perf_ns": perf_ns or time.perf_counter_ns(),
                "metric_1": metric_1,
                "metric_2": metric_2,
                "message": message,
            },
        )

    def _audit_record_books(
        self,
        scan_id: str | None,
        books: dict[str, OrderBook | None],
        provider: OrderBookProvider,
        token_ids: list[str],
    ) -> None:
        if not self._audit.enabled:
            return
        local_ts_ns = time.time_ns()
        for token_id in token_ids:
            book = books.get(token_id)
            if book is None:
                continue
            raw = book.raw_json if isinstance(book.raw_json, dict) else {}
            snapshot_id = self._audit.next_id("book")
            best_bid = _float(book.best_bid)
            best_ask = _float(book.best_ask)
            spread = _float(book.spread)
            self._audit.record(
                "book_snapshots",
                {
                    "book_snapshot_id": snapshot_id,
                    "scan_id": scan_id,
                    "token_id": token_id,
                    "source": raw.get("_audit_source") or self.settings.book_source,
                    "source_event_id": raw.get("_audit_event_id"),
                    "exchange_ts_ns": _audit_datetime_ns(book.timestamp),
                    "local_ts_ns": local_ts_ns,
                    "book_age_ms": provider.book_age_ms(token_id),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "bid_depth": _float(book.bid_depth),
                    "ask_depth": _float(book.ask_depth),
                    "bid_level_count": len(book.bids),
                    "ask_level_count": len(book.asks),
                    "depth_untrusted": _book_depth_untrusted(book),
                },
            )
            for side, levels in [("bid", book.bids), ("ask", book.asks)]:
                cumulative = Decimal("0")
                for level_index, level in enumerate(levels):
                    cumulative += level.size
                    self._audit.record(
                        "book_levels",
                        {
                            "book_snapshot_id": snapshot_id,
                            "scan_id": scan_id,
                            "token_id": token_id,
                            "side": side,
                            "level_index": level_index,
                            "price": _float(level.price),
                            "size": _float(level.size),
                            "cumulative_size": _float(cumulative),
                            "local_ts_ns": local_ts_ns,
                        },
                    )

    def _audit_record_pair_observations(
        self,
        scan_id: str | None,
        timestamp: str,
        targets: tuple[PairScanTarget, ...],
        observations: list[PairObservation],
    ) -> dict[str, str]:
        if not self._audit.enabled:
            return {}
        out: dict[str, str] = {}
        token_by_pair = {target.pair.name: target.leg_token_ids for target in targets}
        for obs in observations:
            observation_id = self._audit.next_id("obs")
            opportunity_id = _audit_pair_opportunity_id(obs.pair_name)
            leg_1, leg_2 = token_by_pair.get(obs.pair_name, (None, None))
            out[obs.pair_name] = observation_id
            self._audit.record(
                "pair_observations",
                {
                    "observation_id": observation_id,
                    "scan_id": scan_id,
                    "opportunity_id": opportunity_id,
                    "timestamp": timestamp,
                    "pair_name": obs.pair_name,
                    "candidate_type": "pair",
                    "parent_token_id": leg_1,
                    "child_token_id": leg_2,
                    "parent_yes_ask": obs.parent_yes_ask,
                    "child_no_ask": obs.child_no_ask,
                    "parent_yes_bid": obs.parent_yes_bid,
                    "child_no_bid": obs.child_no_bid,
                    "gross_total_cost": obs.gross_total_cost,
                    "estimated_fee_total_per_unit": obs.estimated_fee_total_per_unit,
                    "slippage_buffer": obs.slippage_buffer,
                    "net_total_cost": obs.net_total_cost,
                    "entry_threshold": obs.entry_threshold,
                    "distance_to_entry": obs.distance_to_entry,
                    "worst_case_profit_per_unit": obs.worst_case_profit_per_unit,
                    "best_case_profit_per_unit": obs.best_case_profit_per_unit,
                    "max_executable_size": obs.max_executable_size,
                    "classification": obs.classification,
                    "rejection_reason": obs.rejection_reason,
                    "optimal_size": obs.optimal_size,
                    "optimal_required_capital": obs.optimal_required_capital,
                    "optimal_guaranteed_profit": obs.optimal_guaranteed_profit,
                    "optimal_net_cost_per_unit": obs.optimal_net_cost_per_unit,
                    "spread_check_passed": obs.rejection_reason != "spread_too_wide",
                    "depth_check_passed": obs.max_executable_size > 0,
                    "edge_check_passed": obs.classification == "EXECUTABLE_ARBITRAGE_CANDIDATE",
                },
            )
            self._audit_record_timeline(
                event_kind="pair_observation",
                scan_id=scan_id,
                opportunity_id=opportunity_id,
                candidate_name=obs.pair_name,
                metric_1=obs.net_total_cost,
                metric_2=obs.max_executable_size,
                message=obs.classification,
            )
        return out

    def _audit_record_n_leg_candidates(
        self,
        scan_id: str | None,
        specs: tuple[NLegOpportunitySpec, ...],
        books: dict[str, OrderBook | None],
        best_n_leg: dict[str, Any],
    ) -> dict[str, str]:
        if not self._audit.enabled:
            return {}
        out: dict[str, str] = {}
        best_name = str(best_n_leg.get("name") or "")
        for spec in specs:
            candidate_id = self._audit.next_id("nleg")
            opportunity_id = _audit_n_leg_opportunity_id(spec.name)
            out[spec.name] = candidate_id
            leg_books = [books.get(leg.token_id) for leg in spec.legs]
            typed_books = [book for book in leg_books if book is not None]
            missing = len(typed_books) != len(spec.legs)
            asks = [_float(book.best_ask) for book in typed_books]
            missing_ask = missing or any(price is None for price in asks)
            gross_cost = sum(price for price in asks if price is not None) if not missing_ask else None
            gross_edge = spec.guaranteed_payout - gross_cost if gross_cost is not None else None
            merged = best_n_leg if spec.name == best_name else {}
            classification = (
                "REJECTED"
                if missing or missing_ask or gross_edge is None or gross_edge <= 0
                else "EXECUTABLE_ARBITRAGE_CANDIDATE"
            )
            rejection_reason = "missing_order_book" if missing else "missing_ask" if missing_ask else "not_positive_edge" if classification == "REJECTED" else None
            self._audit.record(
                "n_leg_candidates",
                {
                    "n_leg_candidate_id": candidate_id,
                    "scan_id": scan_id,
                    "opportunity_id": opportunity_id,
                    "name": spec.name,
                    "event_date": spec.event_date,
                    "leg_count": len(spec.legs),
                    "leg_token_ids_json": json.dumps([leg.token_id for leg in spec.legs]),
                    "leg_labels_json": json.dumps([leg.label for leg in spec.legs]),
                    "guaranteed_payout": spec.guaranteed_payout,
                    "gross_cost": gross_cost,
                    "gross_edge": gross_edge,
                    "score_edge": merged.get("score_edge"),
                    "optimal_size": merged.get("optimal_size"),
                    "optimal_capital": merged.get("optimal_capital"),
                    "optimal_profit": merged.get("optimal_profit"),
                    "max_spend_size": merged.get("max_spend_size"),
                    "max_spend_capital": merged.get("max_spend_capital"),
                    "max_spend_profit": merged.get("max_spend_profit"),
                    "classification": classification,
                    "rejection_reason": rejection_reason,
                    "optimizer_ms": None,
                },
            )
        return out

    def _audit_update_opportunity_windows(
        self,
        *,
        scan_id: str | None,
        timestamp_ns: int,
        observations: list[PairObservation],
        best_n_leg: dict[str, Any],
    ) -> None:
        if not self._audit.enabled:
            return
        current: dict[str, tuple[str, str, float]] = {}
        for obs in observations:
            if obs.classification == "EXECUTABLE_ARBITRAGE_CANDIDATE" and obs.net_total_cost is not None:
                current[_audit_pair_opportunity_id(obs.pair_name)] = (
                    obs.pair_name,
                    "pair",
                    1.0 - obs.net_total_cost,
                )
        if best_n_leg and float(best_n_leg.get("gross_edge") or 0.0) > 0.0:
            name = str(best_n_leg.get("name") or "")
            current[_audit_n_leg_opportunity_id(name)] = (name, "n_leg", float(best_n_leg.get("gross_edge") or 0.0))

        for opportunity_id, (name, candidate_type, edge) in current.items():
            state = self._audit_windows.get(opportunity_id)
            if state is None:
                state = {
                    "window_id": self._audit.next_id("window"),
                    "candidate_name": name,
                    "candidate_type": candidate_type,
                    "first_seen_scan_id": scan_id,
                    "first_seen_ns": timestamp_ns,
                    "scan_count": 0,
                    "best_edge": edge,
                }
                self._audit_windows[opportunity_id] = state
            state["last_seen_scan_id"] = scan_id
            state["last_seen_ns"] = timestamp_ns
            state["scan_count"] = int(state.get("scan_count") or 0) + 1
            state["best_edge"] = max(float(state.get("best_edge") or edge), edge)
            self._audit_write_window(opportunity_id, state, status="active", close_reason=None)

        for opportunity_id in list(self._audit_windows):
            if opportunity_id in current:
                continue
            state = self._audit_windows.pop(opportunity_id)
            self._audit_write_window(opportunity_id, state, status="closed", close_reason="no_longer_executable")

    def _audit_finalize_open_windows(self) -> None:
        if not self._audit.enabled:
            return
        for opportunity_id, state in list(self._audit_windows.items()):
            self._audit_write_window(opportunity_id, state, status="closed", close_reason="run_finished")
        self._audit_windows.clear()

    def _audit_write_window(self, opportunity_id: str, state: dict[str, Any], *, status: str, close_reason: str | None) -> None:
        first_seen_ns = int(state.get("first_seen_ns") or 0)
        last_seen_ns = int(state.get("last_seen_ns") or first_seen_ns)
        self._audit.record(
            "opportunity_windows",
            {
                "window_id": state.get("window_id"),
                "opportunity_id": opportunity_id,
                "candidate_name": state.get("candidate_name"),
                "candidate_type": state.get("candidate_type"),
                "first_seen_scan_id": state.get("first_seen_scan_id"),
                "last_seen_scan_id": state.get("last_seen_scan_id"),
                "first_seen_ns": first_seen_ns,
                "last_seen_ns": last_seen_ns,
                "duration_ms": max(0.0, (last_seen_ns - first_seen_ns) / 1_000_000),
                "scan_count": state.get("scan_count"),
                "best_edge": state.get("best_edge"),
                "status": status,
                "close_reason": close_reason,
                "polling_uncertainty_ms": self.settings.poll_seconds * 1000 if self.settings.book_source != "websocket" else None,
            },
        )

    def _audit_record_portfolio_snapshot(self, scan_id: str | None, timestamp: str, decision_id: str | None) -> str | None:
        if not self._audit.enabled:
            return None
        snapshot_id = self._audit.next_id("portfolio")
        open_positions = [pos for pos in self.positions if pos.status == "open"]
        exposure: dict[str, float] = {}
        for pos in open_positions:
            exposure[pos.pair_name] = exposure.get(pos.pair_name, 0.0) + pos.locked_capital
        live_spend = self._live_trader.session_requested_spend if self._live_trader is not None else 0.0
        self._audit.record(
            "portfolio_snapshots",
            {
                "portfolio_snapshot_id": snapshot_id,
                "scan_id": scan_id,
                "decision_id": decision_id,
                "timestamp": timestamp,
                "cash_available": self.cash,
                "locked_capital": sum(pos.locked_capital for pos in open_positions),
                "open_positions_count": len(open_positions),
                "live_session_spend": live_spend,
                "max_trade_size": self.settings.max_trade_size,
                "min_trade_size": self.settings.min_trade_size,
                "max_total_locked_capital": self.settings.max_total_locked_capital,
                "capital_fraction_per_trade": self.settings.capital_fraction_per_trade,
                "cooldown_seconds_per_pair": self.settings.cooldown_seconds_per_pair,
                "open_positions_json": json.dumps([asdict(pos) for pos in open_positions], sort_keys=True),
                "exposure_by_pair_json": json.dumps(exposure, sort_keys=True),
            },
        )
        return snapshot_id

    def _audit_record_decision(
        self,
        *,
        scan_id: str | None,
        decision_id: str | None,
        opportunity_id: str,
        candidate_type: str,
        candidate_name: str,
        observation_id: str | None,
        n_leg_candidate_id: str | None,
        portfolio_snapshot_id: str | None,
        action: str,
        decision_wall_ns: int,
        decision_perf_ns: int,
        decision_to_ack_ms: float | None,
        book_to_detection_ms: float | None,
        detection_to_decision_ms: float | None,
        edge: float | None,
        gross_cost: float | None,
        net_cost: float | None,
        size: float | None,
        locked_capital: float | None,
        passed_spread_check: bool,
        passed_depth_check: bool,
        passed_edge_check: bool,
    ) -> None:
        if not self._audit.enabled or decision_id is None:
            return
        entered = action.startswith("entered")
        submitted = entered or action.startswith("live skipped")
        lower_action = action.lower()
        passed_capital = not any(marker in lower_action for marker in ["capital", "cash", "max locked"])
        parsed = _audit_parse_action_numbers(action)
        self._audit.record(
            "decisions",
            {
                "decision_id": decision_id,
                "scan_id": scan_id,
                "opportunity_id": opportunity_id,
                "candidate_type": candidate_type,
                "candidate_name": candidate_name,
                "observation_id": observation_id,
                "n_leg_candidate_id": n_leg_candidate_id,
                "portfolio_snapshot_id": portfolio_snapshot_id,
                "order_ids_json": None,
                "timestamp": datetime.fromtimestamp(decision_wall_ns / 1_000_000_000, tz=UTC).isoformat(),
                "decision_wall_ns": decision_wall_ns,
                "decision_perf_ns": decision_perf_ns,
                "outcome": "filled" if entered else "skipped",
                "skip_reason": None if entered else action,
                "action": action,
                "passed_spread_check": passed_spread_check,
                "passed_depth_check": passed_depth_check,
                "passed_edge_check": passed_edge_check,
                "passed_capital_check": passed_capital,
                "submitted": submitted,
                "filled": entered,
                "book_to_detection_ms": book_to_detection_ms,
                "detection_to_decision_ms": detection_to_decision_ms,
                "decision_to_ack_ms": decision_to_ack_ms,
                "edge": edge,
                "gross_cost": gross_cost,
                "net_cost": net_cost,
                "size": parsed.get("size", size),
                "locked_capital": parsed.get("locked", locked_capital),
            },
        )
        self._audit_record_timeline(
            event_kind="decision",
            scan_id=scan_id,
            decision_id=decision_id,
            opportunity_id=opportunity_id,
            candidate_name=candidate_name,
            timestamp_ns=decision_wall_ns,
            perf_ns=decision_perf_ns,
            metric_1=edge,
            metric_2=decision_to_ack_ms,
            message=action,
        )

    def _audit_record_missed_fill(
        self,
        *,
        scan_id: str | None,
        decision_id: str | None,
        opportunity_id: str,
        candidate_name: str,
        candidate_type: str,
        action: str,
        expected_profit: float | None,
        edge: float | None,
        gross_cost: float | None,
        net_cost: float | None,
        detected_ts_ns: int,
        decision_ts_ns: int,
        token_ids: list[str],
    ) -> None:
        if not self._audit.enabled:
            return
        classification = _audit_miss_classification(action)
        activity_score = self._audit.recent_activity_score(token_ids)
        self._audit.record(
            "missed_fills",
            {
                "missed_fill_id": self._audit.next_id("miss"),
                "decision_id": decision_id,
                "scan_id": scan_id,
                "opportunity_id": opportunity_id,
                "candidate_name": candidate_name,
                "candidate_type": candidate_type,
                "classification": classification,
                "reason": action,
                "expected_profit": expected_profit,
                "edge": edge,
                "gross_cost": gross_cost,
                "net_cost": net_cost,
                "detected_ts_ns": detected_ts_ns,
                "decision_ts_ns": decision_ts_ns,
                "gone_ts_ns": None,
                "market_activity_score": activity_score,
            },
        )
        self._audit_record_timeline(
            event_kind="missed_fill",
            scan_id=scan_id,
            decision_id=decision_id,
            opportunity_id=opportunity_id,
            candidate_name=candidate_name,
            timestamp_ns=decision_ts_ns,
            metric_1=edge,
            metric_2=activity_score,
            message=classification,
        )

    def _audit_book_to_detection_ms(self, pair: PairConfig, provider: OrderBookProvider) -> float | None:
        if not self._audit.enabled:
            return None
        ages = [provider.book_age_ms(token) for token in _pair_leg_token_ids(pair) if token]
        clean = [age for age in ages if age is not None]
        return max(clean) if clean else None

    def _audit_n_leg_book_to_detection_ms(self, best_n_leg: dict[str, Any], provider: OrderBookProvider) -> float | None:
        if not self._audit.enabled:
            return None
        ages = [provider.book_age_ms(str(token)) for token in best_n_leg.get("leg_token_ids") or []]
        clean = [age for age in ages if age is not None]
        return max(clean) if clean else None

    def _render_dashboard(self, row: ScanRow) -> None:
        renderable = self._dashboard_renderable(row)
        if self._live is not None:
            self._live.update(renderable)
            return
        if self.settings.clear_screen:
            os.system("clear")
            self._dashboard_line_count = 0
        if self.console and Panel:
            if self.settings.live_dashboard:
                self._print_dashboard_in_place(self._capture_renderable(renderable))
            else:
                self.console.print(renderable)
        else:
            text = str(renderable)
            if self.settings.live_dashboard:
                self._print_dashboard_in_place(text)
            else:
                print("\n" + text)

    def _capture_renderable(self, renderable: Any) -> str:
        if Console is None:
            return str(renderable)
        width = self.console.width if self.console is not None else 100
        buffer = io.StringIO()
        capture_console = Console(file=buffer, force_terminal=False, color_system=None, width=width)
        capture_console.print(renderable)
        return buffer.getvalue()

    def _print_dashboard_in_place(self, text: str) -> None:
        if self._dashboard_line_count:
            print(f"\x1b[{self._dashboard_line_count}F\x1b[J", end="")
        if not text.endswith("\n"):
            text += "\n"
        print(text, end="", flush=True)
        self._dashboard_line_count = text.count("\n")

    def _write_markdown(self) -> None:
        open_positions = [pos for pos in self.positions if pos.status == "open"]
        realized = sum(pos.realized_pnl for pos in self.positions if pos.status == "closed")
        liquidation_pnl = sum(pos.liquidation_pnl for pos in open_positions)
        unrealized = liquidation_pnl
        locked = sum(pos.locked_capital for pos in open_positions)
        guaranteed = sum(pos.worst_case_profit for pos in open_positions)
        best_case = sum(pos.best_case_profit for pos in open_positions)
        lines = [
            "# Paper Arbitrage Simulation",
            "",
            "_Research/paper only. No live orders were placed._",
            "",
            f"- Starting budget: {self.settings.budget:.4f}",
            f"- Ending cash: {self.cash:.4f}",
            f"- Locked capital: {locked:.4f}",
            f"- Number of scans: {self._total_scans}",
            f"- Number of executable opportunities seen: {sum(row.executable_candidates_count for row in self.scan_rows)}",
            f"- Number of paper trades entered: {len(self.positions)}",
            f"- Number of open positions: {len(open_positions)}",
            f"- Number of closed positions: {sum(1 for pos in self.positions if pos.status == 'closed')}",
            f"- Realized PnL: {realized:.4f}",
            f"- Liquidation / exit PnL for open positions: {liquidation_pnl:.4f}",
            f"- Backward-compatible `unrealized_pnl` alias: {unrealized:.4f}",
            f"- Guaranteed worst-case PnL if held: {guaranteed:.4f}",
            f"- Best-case PnL if held: {best_case:.4f}",
            "",
            "## Trades Entered",
            "",
        ]
        lines += markdown_table([asdict(pos) for pos in self.positions])
        lines += ["", "## Top Opportunities Missed", ""]
        lines += markdown_table(self.missed[-20:])
        lines += ["", "## Scan Summary", ""]
        # scan_rows is a collections.deque (no slice support); materialize the
        # tail via list() before slicing.
        lines += markdown_table([asdict(row) for row in list(self.scan_rows)[-50:]])
        market_data_warning = (
            "- Kalshi mode uses public REST market data only; live Kalshi order placement is intentionally disabled."
            if self.settings.exchange == "kalshi"
            else "- WebSocket mode consumes public market data only; live order placement is not implemented."
        )
        fee_warning = (
            "- Kalshi dynamic fee lookup is not configured; fees use conservative YAML/category fallback."
            if self.settings.exchange == "kalshi"
            else "- Fees are fetched from the public CLOB `/fee-rate` endpoint by default, with conservative pair/category fallback."
        )
        lines += [
            "",
            "## Warnings / Limitations",
            "",
            "- This simulation is paper-only and never places real trades.",
            market_data_warning,
            "- Depth-aware sizing is skipped for pairs whose top-of-book net cost is above `--optimizer-net-cutoff` because deeper asks cannot improve buy prices.",
            fee_warning,
            "- Liquidation PnL uses executable bid-side exits and is not the same as hold-to-resolution guaranteed PnL.",
            "- Capital remains locked under hold-until-resolution; there is no settlement ingestion layer here.",
        ]
        self.settings.save_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def load_pairs(path: Path, *, include_disabled: bool = False) -> list[PairConfig]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_SAFE_LOADER) or {}
    pairs = [PairConfig.from_dict(item) for item in payload.get("pairs") or []]
    if not include_disabled:
        pairs = [pair for pair in pairs if pair.enabled]
    return pairs


async def resolve_missing_tokens(pairs: list[PairConfig], *, exchange: str = "polymarket") -> None:
    if exchange == "kalshi":
        _resolve_kalshi_tokens(pairs)
        return
    unresolved = [
        pair
        for pair in pairs
        if (pair.parent_market_slug and (not pair.parent_yes_token_id or not pair.parent_no_token_id))
        or (pair.child_market_slug and (not pair.child_yes_token_id or not pair.child_no_token_id))
    ]
    if not unresolved:
        return
    market_cache: dict[str, dict[str, Any] | None] = {}

    async def cached_market(gamma: GammaClient, slug: str) -> dict[str, Any] | None:
        if slug not in market_cache:
            market_cache[slug] = await gamma.market_by_slug(slug)
        return market_cache[slug]

    async with GammaClient() as gamma:
        for pair in unresolved:
            if not pair.parent_yes_token_id and pair.parent_market_slug:
                market = await cached_market(gamma, pair.parent_market_slug)
                if market:
                    yes, no = resolve_binary_token_ids_from_market(market)
                    pair.parent_yes_token_id = yes
                    pair.parent_no_token_id = no
            if not pair.parent_no_token_id and pair.parent_market_slug:
                market = await cached_market(gamma, pair.parent_market_slug)
                if market:
                    yes, no = resolve_binary_token_ids_from_market(market)
                    pair.parent_yes_token_id = pair.parent_yes_token_id or yes
                    pair.parent_no_token_id = no
            if not pair.child_yes_token_id and pair.child_market_slug:
                market = await cached_market(gamma, pair.child_market_slug)
                if market:
                    yes, no = resolve_binary_token_ids_from_market(market)
                    pair.child_yes_token_id = yes
                    pair.child_no_token_id = pair.child_no_token_id or no
            if not pair.child_no_token_id and pair.child_market_slug:
                market = await cached_market(gamma, pair.child_market_slug)
                if market:
                    yes, no = resolve_binary_token_ids_from_market(market)
                    pair.child_yes_token_id = pair.child_yes_token_id or yes
                    pair.child_no_token_id = no


def _resolve_kalshi_tokens(pairs: list[PairConfig]) -> None:
    for pair in pairs:
        parent_ticker = (pair.parent_market_ticker or pair.parent_market_slug or "").strip()
        child_ticker = (pair.child_market_ticker or pair.child_market_slug or "").strip()
        if parent_ticker:
            pair.parent_market_ticker = parent_ticker.upper()
            pair.parent_yes_token_id = pair.parent_yes_token_id or kalshi_token_id(parent_ticker, "yes")
            pair.parent_no_token_id = pair.parent_no_token_id or kalshi_token_id(parent_ticker, "no")
        else:
            _normalize_existing_kalshi_tokens(pair, market="parent")
        if child_ticker:
            pair.child_market_ticker = child_ticker.upper()
            pair.child_yes_token_id = pair.child_yes_token_id or kalshi_token_id(child_ticker, "yes")
            pair.child_no_token_id = pair.child_no_token_id or kalshi_token_id(child_ticker, "no")
        else:
            _normalize_existing_kalshi_tokens(pair, market="child")


def _normalize_existing_kalshi_tokens(pair: PairConfig, *, market: str) -> None:
    yes_attr = f"{market}_yes_token_id"
    no_attr = f"{market}_no_token_id"
    yes_token = getattr(pair, yes_attr)
    no_token = getattr(pair, no_attr)
    parsed_yes = parse_kalshi_token_id(yes_token or "")
    parsed_no = parse_kalshi_token_id(no_token or "")
    if parsed_yes is not None and parsed_yes.side == "yes" and not no_token:
        setattr(pair, no_attr, kalshi_token_id(parsed_yes.ticker, "no"))
    if parsed_no is not None and parsed_no.side == "no" and not yes_token:
        setattr(pair, yes_attr, kalshi_token_id(parsed_no.ticker, "yes"))


def _float_avg_price_for_size(levels_float: list[tuple[float, float]], size: float) -> float | None:
    """Float-only VWAP computation for the depth-optimizer hot path.

    Equivalent to OrderBook.avg_ask_price_for_size but operates on pre-converted
    (price, size) float tuples, avoiding per-iteration Decimal construction.
    """
    remaining = size
    total = 0.0
    filled = 0.0
    for price, lsize in levels_float:
        take = min(lsize, remaining)
        if take <= 1e-12:
            continue
        total += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    if filled < size - 1e-9:
        return None
    return total / size if size > 0 else None


def estimated_fee_per_unit(price: float, fee_rate: float) -> float:
    if fee_rate <= 0:
        return 0.0
    return fee_rate * price * (1.0 - price)


async def build_watchlist_once(
    *,
    pairs_path: Path,
    out_path: Path,
    ranking_csv_path: Path,
    top_n: int = 10,
    max_net_cost: float = 1.05,
    min_depth: float = 1.0,
    include_disabled: bool = True,
    entry_threshold: float = 1.0,
    min_edge_threshold: float | None = 0.0025,
    near_arb_threshold: float = 1.02,
    slippage_buffer: float | None = None,
    fee_rate: float | None = None,
    max_concurrent_requests: int = 10,
    relation_safety: str = "all",
    min_display_price: float | None = None,
    max_threshold_distance_pct: float | None = None,
    spot_prices: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a small liquid paper watchlist from a broad generated candidate file."""
    pairs = await load_pairs(pairs_path, include_disabled=include_disabled)
    await resolve_missing_tokens(pairs)
    usable = [
        pair
        for pair in pairs
        if all(_pair_leg_token_ids(pair))
        and _passes_relation_safety(pair, relation_safety)
        and _passes_display_price(pair, min_display_price)
        and _passes_threshold_distance(pair, spot_prices or {}, max_threshold_distance_pct)
    ]
    settings = SimulatorSettings(
        pairs_path=pairs_path,
        entry_threshold=entry_threshold,
        min_edge_threshold=min_edge_threshold,
        near_arb_threshold=near_arb_threshold,
        slippage_buffer=slippage_buffer,
        fee_rate=fee_rate,
        max_concurrent_requests=max_concurrent_requests,
    )
    observer = PaperArbSimulator(settings)
    timestamp = now_iso()
    token_ids: list[str] = []
    for pair in usable:
        token_ids.extend([token for token in _pair_leg_token_ids(pair) if token])

    async with ClobClient() as clob:
        await observer._prepare_pair_caches(usable, clob)
        batch = await clob.get_order_books(token_ids, max_concurrent_requests=max_concurrent_requests)

    observations = [observer._observe_pair(timestamp, pair, batch.books) for pair in usable]
    by_name = {pair.name: pair for pair in usable}
    rows: list[dict[str, Any]] = []
    for obs in observations:
        pair = by_name[obs.pair_name]
        passed = (
            obs.net_total_cost is not None
            and obs.max_executable_size >= min_depth
            and obs.net_total_cost <= max_net_cost
        )
        row = asdict(obs)
        row.update(
            {
                "passed_watchlist_filter": passed,
                "parent_market_slug": pair.parent_market_slug,
                "child_market_slug": pair.child_market_slug,
                "parent_yes_token_id": pair.parent_yes_token_id,
                "parent_no_token_id": pair.parent_no_token_id,
                "child_yes_token_id": pair.child_yes_token_id,
                "child_no_token_id": pair.child_no_token_id,
                "rank_score": _watchlist_rank_score(obs),
            }
        )
        rows.append(row)

    rows.sort(key=lambda row: (not row["passed_watchlist_filter"], row["rank_score"]))
    _ensure_parent(ranking_csv_path)
    if rows:
        with ranking_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    selected_rows = [row for row in rows if row["passed_watchlist_filter"]][:top_n]
    selected_names = {row["pair_name"] for row in selected_rows}
    selected_pairs: list[dict[str, Any]] = []
    for pair in usable:
        if pair.name not in selected_names:
            continue
        raw = dict(pair.raw)
        raw["enabled"] = True
        raw["parent_yes_token_id"] = pair.parent_yes_token_id
        raw["parent_no_token_id"] = pair.parent_no_token_id
        raw["child_yes_token_id"] = pair.child_yes_token_id
        raw["child_no_token_id"] = pair.child_no_token_id
        raw["paper_watchlist_generated_at"] = timestamp
        matching = next(row for row in selected_rows if row["pair_name"] == pair.name)
        raw["paper_watchlist_metrics"] = {
            "parent_yes_ask": matching["parent_yes_ask"],
            "child_no_ask": matching["child_no_ask"],
            "gross_total_cost": matching["gross_total_cost"],
            "net_total_cost": matching["net_total_cost"],
            "distance_to_entry": matching["distance_to_entry"],
            "max_executable_size": matching["max_executable_size"],
            "classification": matching["classification"],
        }
        selected_pairs.append(raw)

    # Preserve ranking order in the emitted YAML.
    order = {row["pair_name"]: index for index, row in enumerate(selected_rows)}
    selected_pairs.sort(key=lambda item: order.get(item.get("name"), 999999))
    _ensure_parent(out_path)
    out_path.write_text(
        yaml.safe_dump(
            {
                "notes": "Paper-only executable watchlist generated from public CLOB books. No live trading.",
                "pairs": selected_pairs,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {
        "total_pairs_loaded": len(pairs),
        "usable_pairs": len(usable),
        "valid_quotes": sum(1 for row in rows if row["net_total_cost"] is not None),
        "passed_filter": len([row for row in rows if row["passed_watchlist_filter"]]),
        "written": len(selected_pairs),
        "out": str(out_path),
        "ranking_csv": str(ranking_csv_path),
        "best_net_cost": selected_rows[0]["net_total_cost"] if selected_rows else None,
        "best_pair": selected_rows[0]["pair_name"] if selected_rows else None,
    }


async def benchmark_scan(
    *,
    pairs_path: Path,
    iterations: int = 5,
    include_disabled: bool = True,
    entry_threshold: float = 1.0,
    min_edge_threshold: float | None = 0.0025,
    near_arb_threshold: float = 1.02,
    slippage_buffer: float | None = None,
    fee_rate: float | None = None,
    sizing_mode: str = "max_profit",
    optimizer_net_cutoff: float = 1.05,
    max_concurrent_requests: int = 10,
) -> dict[str, Any]:
    load_started = time.perf_counter()
    pairs = await load_pairs(pairs_path, include_disabled=include_disabled)
    await resolve_missing_tokens(pairs)
    load_resolve_ms = (time.perf_counter() - load_started) * 1000
    usable = [pair for pair in pairs if all(_pair_leg_token_ids(pair))]
    observer = PaperArbSimulator(
        SimulatorSettings(
            pairs_path=pairs_path,
            entry_threshold=entry_threshold,
            min_edge_threshold=min_edge_threshold,
            near_arb_threshold=near_arb_threshold,
            slippage_buffer=slippage_buffer,
            fee_rate=fee_rate,
            sizing_mode=sizing_mode,
            optimizer_net_cutoff=optimizer_net_cutoff,
            max_concurrent_requests=max_concurrent_requests,
        )
    )
    token_ids: list[str] = []
    for pair in usable:
        token_ids.extend([token for token in _pair_leg_token_ids(pair) if token])
    unique_tokens = len(set(token_ids))
    fetch_times: list[float] = []
    observe_times: list[float] = []
    total_times: list[float] = []
    last_counts: dict[str, int] = {}
    async with ClobClient() as clob:
        await observer._prepare_pair_caches(usable, clob)
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            fetch_started = time.perf_counter()
            batch = await clob.get_order_books(token_ids, max_concurrent_requests=max_concurrent_requests)
            fetch_times.append((time.perf_counter() - fetch_started) * 1000)
            observe_started = time.perf_counter()
            observations = [observer._observe_pair(now_iso(), pair, batch.books) for pair in usable]
            observe_times.append((time.perf_counter() - observe_started) * 1000)
            total_times.append((time.perf_counter() - started) * 1000)
            last_counts = {
                "valid_quotes": sum(1 for obs in observations if obs.net_total_cost is not None),
                "executable": sum(1 for obs in observations if obs.classification == "EXECUTABLE_ARBITRAGE_CANDIDATE"),
                "near": sum(1 for obs in observations if obs.classification == "NEAR_ARBITRAGE"),
                "rejected": sum(1 for obs in observations if obs.classification == "REJECTED"),
                "missing_books": sum(1 for obs in observations if obs.rejection_reason == "missing_order_book"),
                "missing_asks": sum(1 for obs in observations if obs.rejection_reason == "missing_ask"),
            }
    csv_write_ms, markdown_write_ms = _profile_report_io()
    return {
        "pairs_loaded": len(pairs),
        "usable_pairs": len(usable),
        "unique_tokens": unique_tokens,
        "iterations": max(1, iterations),
        "load_and_resolve_tokens_ms": load_resolve_ms,
        "avg_order_book_fetch_ms": _avg(fetch_times),
        "max_order_book_fetch_ms": max(fetch_times) if fetch_times else None,
        "avg_observation_compute_ms": _avg(observe_times),
        "max_observation_compute_ms": max(observe_times) if observe_times else None,
        "avg_total_scan_time_ms": _avg(total_times),
        "min_total_scan_time_ms": min(total_times) if total_times else None,
        "max_total_scan_time_ms": max(total_times) if total_times else None,
        "csv_single_row_write_ms": csv_write_ms,
        "markdown_small_report_write_ms": markdown_write_ms,
        **last_counts,
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_No data._"]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_md(row.get(header)) for header in headers) + " |")
    return lines


def _watchlist_rank_score(obs: PairObservation) -> float:
    if obs.net_total_cost is None:
        return math.inf
    # Lower is better. Depth helps break ties, but cost remains dominant.
    depth_bonus = min(obs.max_executable_size, 100.0) / 100_000
    return obs.net_total_cost - depth_bonus


def _cumulative_sizes(levels: list[Any]) -> list[float]:
    total = 0.0
    out: list[float] = []
    for level in levels:
        total += float(level.size)
        out.append(total)
    return out


def _book_marginal_ask_price_for_size(book: OrderBook, size: float) -> float | None:
    remaining = Decimal(str(size))
    if remaining <= 0:
        return None
    marginal: Decimal | None = None
    for level in book.asks:
        take = min(level.size, remaining)
        if take <= 0:
            continue
        marginal = level.price
        remaining -= take
        if remaining <= 0:
            return float(marginal)
    return None


def _book_tick_size(book: OrderBook) -> Decimal:
    raw = book.raw_json.get("tick_size") if isinstance(book.raw_json, dict) else None
    try:
        tick = Decimal(str(raw or "0.001"))
    except Exception:
        tick = Decimal("0.001")
    return tick if tick > 0 else Decimal("0.001")


def _book_tick_size_str(book: OrderBook) -> str:
    return str(_book_tick_size(book).normalize())


def _book_neg_risk(book: OrderBook) -> bool:
    return bool(book.raw_json.get("neg_risk", False)) if isinstance(book.raw_json, dict) else False


def _book_min_order_size(book: OrderBook) -> float:
    raw = book.raw_json.get("min_order_size") if isinstance(book.raw_json, dict) else None
    try:
        return max(0.0, float(raw or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _round_price_up_to_tick(price: float, tick_size: Decimal) -> float:
    price_decimal = Decimal(str(price))
    ticks = (price_decimal / tick_size).to_integral_value(rounding=ROUND_CEILING)
    rounded = ticks * tick_size
    return float(min(Decimal("1") - tick_size, max(tick_size, rounded)))


def _live_min_edge_threshold(cli_value: float | None) -> float:
    return max(0.0, cli_value if cli_value is not None else 0.0)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items() if "private" not in str(key).lower()}
    return str(value)


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _profile_report_io() -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        csv_path = tmp_dir / "scan.csv"
        row = ScanRow(
            timestamp=now_iso(),
            cash_available=100.0,
            locked_capital=0.0,
            open_positions_count=0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            liquidation_pnl=0.0,
            guaranteed_profit_if_held=0.0,
            best_case_profit_if_held=0.0,
            best_pair_name="PROFILE",
            best_total_cost=1.0,
            net_total_cost=1.0,
            entry_threshold=1.0,
            distance_to_entry=0.0,
            best_worst_case_profit=0.0,
            best_optimal_size=1.0,
            best_optimal_guaranteed_profit=0.0,
            executable_candidates_count=0,
            near_arb_candidates_count=0,
            rejected_count=0,
            books_missing_count=0,
            asks_missing_count=0,
            scan_time_ms=0.0,
            book_source="profile",
            unique_tokens=0,
            unique_tokens_fetched=0,
            cache_hits=0,
            failed_book_count=0,
            websocket_connected=False,
            websocket_reconnect_count=0,
            fallback_to_polling_used=False,
            token_update_count=0,
            event_triggered_recomputes=0,
            max_book_age_ms=None,
            update_latency_ms=None,
            action_taken="none",
        )
        started = time.perf_counter()
        _write_csv_header(csv_path, ScanRow)
        _append_csv_row(csv_path, row)
        csv_ms = (time.perf_counter() - started) * 1000
        report_path = tmp_dir / "report.md"
        report_body = "\n".join(markdown_table([asdict(row) for _ in range(50)]))
        started = time.perf_counter()
        report_path.write_text(report_body, encoding="utf-8")
        markdown_ms = (time.perf_counter() - started) * 1000
    return csv_ms, markdown_ms


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _audit_pair_opportunity_id(pair_name: str) -> str:
    return f"pair:{pair_name}"


def _audit_n_leg_opportunity_id(name: str) -> str:
    return f"n_leg:{name}"


def _audit_datetime_ns(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp() * 1_000_000_000)


def _audit_parse_action_numbers(action: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ["size", "locked"]:
        match = re.search(rf"{key}=([-+]?[0-9]*\.?[0-9]+)", action)
        if match:
            try:
                out[key] = float(match.group(1))
            except ValueError:
                pass
    return out


def _audit_miss_classification(action: str) -> str:
    lowered = action.lower()
    if "rest recheck" in lowered and ("no longer" in lowered or "positive" in lowered):
        return "rest_recheck_lost"
    if "rest recheck missing" in lowered or "missing quote" in lowered or "missing book" in lowered:
        return "rest_recheck_missing"
    if "depth" in lowered:
        return "depth_depleted"
    if "capital" in lowered or "cash" in lowered or "max locked" in lowered:
        return "capital_blocked"
    if "duplicate" in lowered:
        return "duplicate_guard_blocked"
    if "cooldown" in lowered:
        return "cooldown_guard_blocked"
    if "live skipped" in lowered or "order rejected" in lowered:
        return "order_failed_or_rejected"
    if "threshold" in lowered or "edge" in lowered:
        return "edge_or_threshold_lost"
    return "unknown"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def _pair_slippage(pair: PairConfig, cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    return float(pair.overrides.get("slippage_buffer", 0.0025))


def _pair_fee_rates(pair: PairConfig, cli_value: float | None) -> tuple[float, float]:
    if cli_value is not None:
        return cli_value, cli_value
    if "estimated_fee_rate" in pair.overrides:
        rate = float(pair.overrides.get("estimated_fee_rate") or 0.0)
        return rate, rate
    fee_mode = str(pair.overrides.get("fee_mode") or "conservative").lower()
    if fee_mode == "none":
        return 0.0, 0.0
    order_role = str(pair.overrides.get("order_role") or "taker").lower()
    if order_role == "maker":
        return 0.0, 0.0
    category = str(pair.overrides.get("fee_category") or _infer_fee_category(pair)).lower()
    rate = DEFAULT_TAKER_FEE_RATES.get(category, DEFAULT_TAKER_FEE_RATES["other"])
    return rate, rate


def _pair_fee_rate(pair: PairConfig, cli_value: float | None) -> float:
    return max(_pair_fee_rates(pair, cli_value))


def _pair_fee_buffer(pair: PairConfig) -> float:
    return float(pair.overrides.get("fee_buffer_usd") or 0.0)


def _pair_min_edge_threshold(pair: PairConfig, cli_value: float | None) -> float:
    if cli_value is not None:
        return max(0.0, cli_value)
    return max(0.0, float(pair.overrides.get("min_edge_threshold") or 0.0))


def _pair_max_spread(pair: PairConfig) -> float | None:
    value = pair.overrides.get("max_spread_per_leg")
    return None if value in (None, "") else float(value)


def _pair_max_trade_size(pair: PairConfig, cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    value = pair.overrides.get("max_trade_size_usd")
    return float(value) if value not in (None, "") else float("inf")


def _derive_range_threshold_n_leg_specs(
    pairs: list[PairConfig], *, max_ranges: int | None = 1
) -> list[NLegOpportunitySpec]:
    threshold_yes: dict[tuple[str, str, str], str] = {}
    threshold_no: dict[tuple[str, str, str], str] = {}
    ranges: dict[tuple[str, str], dict[tuple[str, str], str]] = {}
    event_dates: dict[tuple[str, str], str | None] = {}
    for pair in pairs:
        parent_event = _pair_raw_str(pair, "parent_event_slug") or pair.parent_market_slug
        child_event = _pair_raw_str(pair, "child_event_slug") or pair.child_market_slug
        if not parent_event or not child_event:
            continue
        event_dates.setdefault(
            (parent_event, child_event),
            _optional_str(pair.overrides.get("event_date")) or _optional_str(pair.raw.get("event_date")),
        )
        threshold = _normalize_threshold_label(pair.parent_outcome_label)
        if threshold:
            threshold_key = (parent_event, child_event, threshold)
            if pair.parent_yes_token_id:
                threshold_yes[threshold_key] = pair.parent_yes_token_id
            if pair.parent_no_token_id:
                threshold_no[threshold_key] = pair.parent_no_token_id
        bounds = _parse_range_bounds(pair.child_outcome_label)
        if bounds and pair.child_no_token_id:
            ranges.setdefault((parent_event, child_event), {})[bounds] = pair.child_no_token_id

    specs: list[NLegOpportunitySpec] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for (parent_event, child_event), range_tokens in ranges.items():
        ordered = sorted(
            [
                (low, high, token)
                for (low, high), token in range_tokens.items()
                if _label_number(low) is not None and _label_number(high) is not None
            ],
            key=lambda item: (_label_number(item[0]) or 0.0, _label_number(item[1]) or 0.0),
        )
        for start in range(len(ordered)):
            window: list[tuple[str, str, str]] = []
            for end in range(start, len(ordered)):
                current = ordered[end]
                if window and window[-1][1] != current[0]:
                    break
                window.append(current)
                range_count = len(window)
                if max_ranges is not None and range_count > max_ranges:
                    break
                low_label = window[0][0]
                high_label = window[-1][1]
                low_yes = threshold_yes.get((parent_event, child_event, low_label))
                high_no = threshold_no.get((parent_event, child_event, high_label))
                if not low_yes or not high_no:
                    continue
                legs = [
                    NLeg(label=f"{low_label}_YES", token_id=low_yes),
                    NLeg(label=f"{high_label}_NO", token_id=high_no),
                ]
                legs.extend(NLeg(label=f"{low}_{high}_NO", token_id=token) for low, high, token in window)
                token_signature = tuple(leg.token_id for leg in legs)
                name = f"{child_event} {low_label}-{high_label} via {range_count} range{'s' if range_count != 1 else ''}"
                seen_key = (name, token_signature)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                specs.append(
                    NLegOpportunitySpec(
                        name=name,
                        legs=tuple(legs),
                        guaranteed_payout=float(range_count + 1),
                        event_date=event_dates.get((parent_event, child_event)),
                    )
                )
    return specs


def _pair_raw_str(pair: PairConfig, key: str) -> str:
    value = pair.raw.get(key) if isinstance(pair.raw, dict) else None
    return str(value or "").strip()


def _trade_template_signature(pair: PairConfig) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    if not isinstance(pair.trade_template, dict):
        return ()
    out: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for key, value in sorted(pair.trade_template.items()):
        if isinstance(value, dict):
            out.append((str(key), tuple(sorted((str(item_key), str(item_value)) for item_key, item_value in value.items()))))
        else:
            out.append((str(key), (("value", str(value)),)))
    return tuple(out)


def _normalize_threshold_label(label: str) -> str | None:
    cleaned = "".join(ch for ch in label if ch.isdigit())
    return cleaned if cleaned else None


def _parse_range_bounds(label: str) -> tuple[str, str] | None:
    text = label.replace(",", "").strip()
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    low = "".join(ch for ch in left if ch.isdigit())
    high = "".join(ch for ch in right if ch.isdigit())
    if not low or not high:
        return None
    return low, high


def _label_number(label: str) -> float | None:
    try:
        return float(label)
    except (TypeError, ValueError):
        return None


def _book_depth_untrusted(book: OrderBook) -> bool:
    return bool(book.raw_json.get("best_bid_ask_depth_untrusted"))


def _pair_leg_token_ids(pair: PairConfig) -> tuple[str | None, str | None]:
    leg_1 = _trade_template_leg(pair, "leg_1", default_market="parent", default_outcome="YES")
    leg_2 = _trade_template_leg(pair, "leg_2", default_market="child", default_outcome="NO")
    return _token_for_leg(pair, leg_1), _token_for_leg(pair, leg_2)


def _trade_template_leg(pair: PairConfig, key: str, *, default_market: str, default_outcome: str) -> dict[str, str]:
    leg = pair.trade_template.get(key) if isinstance(pair.trade_template, dict) else None
    if not isinstance(leg, dict):
        leg = {}
    return {
        "market": str(leg.get("market") or default_market).lower(),
        "outcome": str(leg.get("outcome") or default_outcome).upper(),
        "side": str(leg.get("side") or "BUY").upper(),
    }


def _token_for_leg(pair: PairConfig, leg: dict[str, str]) -> str | None:
    market = leg.get("market")
    outcome = leg.get("outcome")
    if leg.get("side") != "BUY":
        return None
    if market == "parent" and outcome == "YES":
        return pair.parent_yes_token_id or _kalshi_token_from_pair_ticker(pair.parent_market_ticker, "yes")
    if market == "parent" and outcome == "NO":
        return pair.parent_no_token_id or _kalshi_token_from_pair_ticker(pair.parent_market_ticker, "no")
    if market == "child" and outcome == "YES":
        return pair.child_yes_token_id or _kalshi_token_from_pair_ticker(pair.child_market_ticker, "yes")
    if market == "child" and outcome == "NO":
        return pair.child_no_token_id or _kalshi_token_from_pair_ticker(pair.child_market_ticker, "no")
    return None


def _kalshi_token_from_pair_ticker(ticker: str, side: str) -> str | None:
    clean = str(ticker or "").strip()
    return kalshi_token_id(clean, side) if clean else None


def _infer_fee_category(pair: PairConfig) -> str:
    text = " ".join(
        [
            pair.name,
            pair.parent_market_slug,
            pair.child_market_slug,
            pair.parent_outcome_label,
            pair.child_outcome_label,
        ]
    ).lower()
    if any(word in text for word in ["btc", "bitcoin", "eth", "ethereum", "solana", "xrp", "crypto"]):
        return "crypto"
    return "other"


def _passes_relation_safety(pair: PairConfig, mode: str) -> bool:
    normalized = (mode or "all").lower()
    if normalized == "all":
        return True
    safety = (pair.relation_safety or "unknown").lower()
    if normalized == "clean":
        return safety in {"clean", "unknown"} and not pair.boundary_ambiguity
    if normalized == "boundary_ambiguous":
        return safety == "boundary_ambiguous" or pair.boundary_ambiguity
    return True


def _passes_display_price(pair: PairConfig, min_display_price: float | None) -> bool:
    if min_display_price is None:
        return True
    prices = [price for price in [pair.parent_display_price, pair.child_display_price] if price is not None]
    if not prices:
        return True
    return any(price >= min_display_price for price in prices)


def _passes_threshold_distance(
    pair: PairConfig,
    spot_prices: dict[str, float],
    max_threshold_distance_pct: float | None,
) -> bool:
    if max_threshold_distance_pct is None or not spot_prices:
        return True
    asset = _infer_asset(pair)
    spot = spot_prices.get(asset)
    if not spot or spot <= 0:
        return True
    thresholds = _extract_numbers(pair.parent_outcome_label) + _extract_numbers(pair.child_outcome_label)
    if not thresholds:
        return True
    closest_distance = min(abs(threshold - spot) / spot for threshold in thresholds)
    return closest_distance <= (max_threshold_distance_pct / 100.0)


def _infer_asset(pair: PairConfig) -> str:
    text = " ".join([pair.name, pair.parent_market_slug, pair.child_market_slug]).lower()
    if "ethereum" in text or "eth" in text:
        return "ETH"
    if "solana" in text or "sol" in text:
        return "SOL"
    if "xrp" in text:
        return "XRP"
    if "bitcoin" in text or "btc" in text:
        return "BTC"
    return "UNKNOWN"


def _extract_numbers(text: str) -> list[float]:
    import re

    numbers: list[float] = []
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            numbers.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return numbers


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class _RunLock:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._owned = False

    def __enter__(self) -> "_RunLock":
        if self.path is None:
            return self
        _ensure_parent(self.path)
        pid = os.getpid()
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing_pid = _read_lock_pid(self.path)
                if existing_pid is not None and not _pid_running(existing_pid):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                owner = f" by pid {existing_pid}" if existing_pid else ""
                raise SystemExit(f"Another bot process appears to be running{owner}; lock file: {self.path}")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{pid}\n")
            self._owned = True
            return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.path is not None and self._owned:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._owned = False


def _read_lock_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(raw)
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _write_csv_header(path: Path, cls: type[Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cls.__dataclass_fields__.keys()))
        writer.writeheader()


def _append_csv_row(path: Path, row: Any) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.__dataclass_fields__.keys()))
        writer.writerow(asdict(row))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
