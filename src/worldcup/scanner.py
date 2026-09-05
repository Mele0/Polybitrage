"""World Cup arbitrage scanner — orchestration layer.

Pipeline per scan:
    discover events -> infer relations -> fetch CLOB order books ->
    compute depth-aware executable edge per relation -> state-validate worst case ->
    rank (deterministic first, statistical second) -> execute (paper/dry-run/live) ->
    log (events / relations / opportunities / skips) + print a concise table.

Reuses the existing :class:`~src.polymarket.models.OrderBook`,
:class:`~src.polymarket.clob_client.ClobClient`, and the WebSocket provider for
book data, and the guarded :class:`~src.polymarket.live_trader.LiveTrader` for
live execution — no market-data or signing logic is duplicated here.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.polymarket.clob_client import ClobClient
from src.polymarket.models import OrderBook
from src.worldcup.discovery import (
    DiscoveryResult,
    collect_token_ids,
    discover_world_cup,
    load_discovery_cache,
    save_discovery_cache,
)
from src.worldcup.executor import (
    ExposureConfig,
    ExposureLimiter,
    Opportunity,
    OpportunityLeg,
    WorldCupExecutor,
)
from src.worldcup.relations import (
    COMPLEMENT,
    QUANTITY,
    SUBSET,
    Relation,
    RelationGraph,
    infer_relations,
    relation_to_pair_dict,
)
from src.worldcup.state_validator import validate_relation


@dataclass
class WorldCupScannerConfig:
    # discovery
    tag_id: str = "102232"
    series_id: str | None = "11433"
    include_closed: bool = False
    include_futures: bool = True
    discovery_cache_path: Path = Path("reports/worldcup/discovery_cache.json")
    discovery_cache_max_age_s: float = 120.0
    overrides_path: Path | None = Path("config/worldcup_overrides.yaml")
    ladder_all_pairs: bool = False
    enable_exact_score: bool = True
    # edge requirements
    min_edge_bps: float = 25.0          # MIN_EDGE_BPS: min net edge per share, in bps of payout
    max_leg_slippage: float = 0.0025    # MAX_LEG_SLIPPAGE: per-share slippage buffer
    fee_rate: float = 0.0               # taker fee rate (per-share fee = rate*p*(1-p))
    fee_buffer_per_share: float = 0.0
    latency_buffer_per_share: float = 0.0
    max_spread_per_leg: float = 0.10
    min_liquidity: float = 1.0          # MIN_LIQUIDITY: min shares of depth on every leg
    max_stale_ms: float = 5000.0        # MAX_STALE_MS
    # sizing / risk
    max_trade_size: float = 20.0
    min_trade_size: float = 1.0
    near_arb_threshold: float = 1.03    # for the statistical layer
    # toggles
    world_cup_mode: bool = True
    paper_trade: bool = True
    execution_mode: str = "paper"       # paper | dry_run | live
    enable_deterministic_arbs: bool = True
    enable_statistical_arbs: bool = False
    # exposure caps
    max_event_exposure: float = 100.0
    max_team_exposure: float = 75.0
    max_relation_exposure: float = 150.0
    max_total_exposure: float = 500.0
    max_daily_loss: float = 100.0
    kill_switch_path: Path | None = Path("data/worldcup_kill_switch")
    # IO
    out_dir: Path = Path("reports/worldcup")
    max_concurrent_requests: int = 10
    show_top: int = 12

    @classmethod
    def from_yaml(cls, path: Path) -> "WorldCupScannerConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls()
        for key, value in data.items():
            if not hasattr(cfg, key):
                continue
            current = getattr(cfg, key)
            if isinstance(current, Path) and value is not None:
                setattr(cfg, key, Path(value))
            else:
                setattr(cfg, key, value)
        return cfg

    def exposure_config(self) -> ExposureConfig:
        return ExposureConfig(
            max_event_exposure=self.max_event_exposure,
            max_team_exposure=self.max_team_exposure,
            max_relation_exposure=self.max_relation_exposure,
            max_total_exposure=self.max_total_exposure,
            max_daily_loss=self.max_daily_loss,
            kill_switch_path=self.kill_switch_path,
        )


@dataclass
class SkippedRelation:
    name: str
    relation_subtype: str
    reason: str
    net_unit_cost: float | None = None


@dataclass
class ScanReport:
    timestamp: str
    discovery: dict[str, Any]
    relations: dict[str, int]
    opportunities: list[Opportunity] = field(default_factory=list)
    statistical: list[Opportunity] = field(default_factory=list)
    skipped: list[SkippedRelation] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    exposure: dict[str, Any] = field(default_factory=dict)


# ── Book source adapters ────────────────────────────────────────────────────────


class _RestBookSource:
    """Minimal book source backed by the public CLOB REST `/books` batch endpoint."""

    def __init__(self, clob: ClobClient, *, max_concurrent_requests: int = 10) -> None:
        self.clob = clob
        self.max_concurrent_requests = max_concurrent_requests
        self._fetched_at: dict[str, float] = {}

    async def start(self) -> None:
        await self.clob.warmup()

    async def stop(self) -> None:
        await self.clob.close()

    async def get_books(self, token_ids: list[str]) -> dict[str, OrderBook | None]:
        batch = await self.clob.get_order_books(token_ids, max_concurrent_requests=self.max_concurrent_requests)
        now = time.monotonic()
        for tid in token_ids:
            self._fetched_at[tid] = now
        return {tid: batch.books.get(tid) for tid in token_ids}

    def book_age_ms(self, token_id: str) -> float | None:
        t = self._fetched_at.get(token_id)
        return None if t is None else max(0.0, (time.monotonic() - t) * 1000.0)


# ── Edge math ─────────────────────────────────────────────────────────────────


def _fee_per_share(price: float, fee_rate: float) -> float:
    if fee_rate <= 0.0:
        return 0.0
    return fee_rate * price * (1.0 - price)


def _cumulative_sizes(book: OrderBook) -> list[float]:
    total = 0.0
    out: list[float] = []
    for level in book.asks:
        total += float(level.size)
        out.append(total)
    return out


def best_basket_size(
    leg_books: list[OrderBook],
    payout: float,
    cfg: WorldCupScannerConfig,
    *,
    max_capital: float,
) -> dict[str, float] | None:
    """Depth-aware size that maximizes guaranteed (worst-case) profit subject to
    min-edge, min-liquidity, spread, and capital constraints."""
    if not leg_books or any(b is None for b in leg_books):
        return None
    if any(not b.asks or b.best_ask is None for b in leg_books):
        return None
    min_depth = min(b.ask_depth for b in leg_books)
    if min_depth < cfg.min_liquidity:
        return None
    # spread guard
    for b in leg_books:
        if cfg.max_spread_per_leg is not None and b.best_bid is not None and b.best_ask is not None:
            if (b.best_ask - b.best_bid) > cfg.max_spread_per_leg:
                return None
    min_edge_per_share = (cfg.min_edge_bps / 10_000.0) * payout
    top_unit = sum(float(b.best_ask or 0.0) for b in leg_books)
    if top_unit <= 0:
        return None
    cap_size = max_capital / max(top_unit, 0.01)
    upper = min(min_depth, max(cap_size * 1.25, cfg.min_trade_size / max(top_unit, 0.01)))
    candidates: set[float] = set()
    for b in leg_books:
        candidates.update(s for s in _cumulative_sizes(b) if 0 < s <= upper)
    for i in range(1, 51):
        candidates.add(upper * i / 50.0)
    from decimal import Decimal
    best: dict[str, float] | None = None
    for size in sorted(candidates):
        if size <= 0 or size > min_depth:
            continue
        prices = [b.avg_ask_price_for_size(Decimal(str(size))) for b in leg_books]
        if any(p is None for p in prices):
            continue
        unit_cost = sum(float(p) for p in prices)  # type: ignore[arg-type]
        fee = sum(_fee_per_share(float(p), cfg.fee_rate) for p in prices)  # type: ignore[arg-type]
        net_unit = unit_cost + fee + cfg.max_leg_slippage + cfg.fee_buffer_per_share + cfg.latency_buffer_per_share
        edge = payout - net_unit
        if edge < min_edge_per_share:
            continue
        capital = size * unit_cost
        if capital > max_capital + 1e-9 or capital < cfg.min_trade_size:
            continue
        profit = size * edge
        roi = profit / max(capital, 1e-9)
        if best is None or profit > best["profit"] + 1e-12:
            best = {
                "size": size, "unit_cost": unit_cost, "net_unit": net_unit,
                "capital": capital, "profit": profit, "roi": roi, "edge": edge,
            }
    return best


def _opportunity_from(
    rel: Relation, leg_books: list[OrderBook], sized: dict[str, float], *, kind: str = "deterministic"
) -> Opportunity:
    legs = [
        OpportunityLeg(
            token_id=leg.token_id, side=leg.side, label=leg.label,
            price=float(b.avg_ask_price_for_size(__import__("decimal").Decimal(str(sized["size"]))) or (b.best_ask or 0.0)),
            size=sized["size"],
            tick_size=None, neg_risk=None,
        )
        for leg, b in zip(rel.legs, leg_books, strict=False)
    ]
    return Opportunity(
        name=rel.name,
        relation_type=rel.relation_type,
        relation_subtype=rel.relation_subtype,
        legs=legs,
        guaranteed_payout=rel.guaranteed_payout,
        size=sized["size"],
        unit_cost=sized["unit_cost"],
        capital=sized["capital"],
        worst_case_profit=sized["profit"],
        roi=sized["roi"],
        edge_per_share=sized["edge"],
        game_id=rel.game_id,
        teams=rel.teams,
        event_slugs=rel.event_slugs,
        confidence=rel.confidence,
        safety=rel.safety,
        reason=rel.reason,
        kind=kind,
    )


# ── Scanner ─────────────────────────────────────────────────────────────────────


class WorldCupScanner:
    def __init__(self, config: WorldCupScannerConfig, *, book_source: Any | None = None,
                 live_trader: Any | None = None) -> None:
        self.cfg = config
        self._clob = ClobClient()
        self.book_source = book_source or _RestBookSource(
            self._clob, max_concurrent_requests=config.max_concurrent_requests
        )
        self.limiter = ExposureLimiter(config.exposure_config())
        self.executor = WorldCupExecutor(
            self.limiter, mode=config.execution_mode, live_trader=live_trader
        )

    async def discover(self, *, use_cache: bool = True) -> DiscoveryResult:
        if use_cache:
            cached = load_discovery_cache(self.cfg.discovery_cache_path, max_age_s=self.cfg.discovery_cache_max_age_s)
            if cached is not None:
                return cached
        result = await discover_world_cup(
            tag_id=self.cfg.tag_id,
            series_id=self.cfg.series_id,
            include_closed=self.cfg.include_closed,
            include_futures=self.cfg.include_futures,
        )
        save_discovery_cache(result, self.cfg.discovery_cache_path)
        return result

    def infer(self, result: DiscoveryResult) -> RelationGraph:
        overrides = self.cfg.overrides_path if (self.cfg.overrides_path and self.cfg.overrides_path.exists()) else None
        return infer_relations(
            result,
            overrides_path=overrides,
            ladder_all_pairs=self.cfg.ladder_all_pairs,
            enable_exact_score=self.cfg.enable_exact_score,
        )

    async def scan_once(self, *, use_cache: bool = True, execute: bool = True) -> ScanReport:
        await self.book_source.start()
        try:
            result = await self.discover(use_cache=use_cache)
            graph = self.infer(result)
            tokens = sorted({tid for rel in graph.relations for tid in rel.token_ids if tid})
            books = await self.book_source.get_books(tokens) if tokens else {}
            report = self._evaluate(result, graph, books)
            if execute:
                await self._execute_all(report)
            self._write_logs(result, graph, report)
            return report
        finally:
            if isinstance(self.book_source, _RestBookSource):
                # leave the client open across repeated scans only if reused; one-shot closes
                pass

    def _stale(self, token_id: str) -> bool:
        age = self.book_source.book_age_ms(token_id) if hasattr(self.book_source, "book_age_ms") else None
        return age is not None and age > self.cfg.max_stale_ms

    def _evaluate(self, result: DiscoveryResult, graph: RelationGraph, books: dict[str, OrderBook | None]) -> ScanReport:
        report = ScanReport(
            timestamp=datetime.now(UTC).isoformat(),
            discovery=result.summary(),
            relations=graph.summary(),
        )
        max_capital = min(self.cfg.max_trade_size, self.cfg.max_total_exposure)
        for rel in graph.relations:
            leg_books = [books.get(tid) for tid in rel.token_ids]
            if any(b is None for b in leg_books):
                report.skipped.append(SkippedRelation(rel.name, rel.relation_subtype, "missing_book"))
                continue
            if any(self._stale(tid) for tid in rel.token_ids):
                report.skipped.append(SkippedRelation(rel.name, rel.relation_subtype, "stale_book"))
                continue
            sized = best_basket_size(leg_books, rel.guaranteed_payout, self.cfg, max_capital=max_capital)
            if sized is None:
                # statistical fallback: positive but below-threshold / boundary-ambiguous
                stat = self._statistical_candidate(rel, leg_books)
                if stat is not None:
                    report.statistical.append(stat)
                else:
                    report.skipped.append(SkippedRelation(rel.name, rel.relation_subtype, "no_profitable_size"))
                continue
            # State-space validation at the executable cost — the hard guard.
            validation = validate_relation(rel, cost=sized["net_unit"], buffer=0.0)
            if not validation.ok:
                report.skipped.append(
                    SkippedRelation(rel.name, rel.relation_subtype,
                                    f"state_validation_failed:{validation.reason}", sized["net_unit"])
                )
                continue
            is_clean = rel.safety == "clean"
            opp = _opportunity_from(rel, leg_books, sized,
                                    kind="deterministic" if is_clean else "statistical")
            if is_clean and self.cfg.enable_deterministic_arbs:
                report.opportunities.append(opp)
            else:
                report.statistical.append(opp)
        report.opportunities.sort(key=lambda o: o.worst_case_profit, reverse=True)
        report.statistical.sort(key=lambda o: o.roi, reverse=True)
        report.exposure = self.limiter.snapshot()
        return report

    def _statistical_candidate(self, rel: Relation, leg_books: list[OrderBook]) -> Opportunity | None:
        """Lightweight statistical layer: relations whose top-of-book net cost is
        within near_arb_threshold of the guaranteed payout but don't clear the
        deterministic min-edge. Ranked by gap; never auto-executed unless
        enable_statistical_arbs is set."""
        if not self.cfg.enable_statistical_arbs:
            return None
        if any(b is None or b.best_ask is None for b in leg_books):
            return None
        unit = sum(float(b.best_ask or 0.0) for b in leg_books)
        if unit <= 0 or unit > rel.guaranteed_payout * self.cfg.near_arb_threshold:
            return None
        min_depth = min(b.ask_depth for b in leg_books)
        if min_depth < self.cfg.min_liquidity:
            return None
        size = min(min_depth, self.cfg.max_trade_size / max(unit, 0.01))
        edge = rel.guaranteed_payout - unit
        legs = [
            OpportunityLeg(leg.token_id, leg.side, leg.label, float(b.best_ask or 0.0), size)
            for leg, b in zip(rel.legs, leg_books, strict=False)
        ]
        return Opportunity(
            name=rel.name, relation_type=rel.relation_type, relation_subtype=rel.relation_subtype,
            legs=legs, guaranteed_payout=rel.guaranteed_payout, size=size, unit_cost=unit,
            capital=size * unit, worst_case_profit=size * edge, roi=(edge / max(unit, 1e-9)),
            edge_per_share=edge, game_id=rel.game_id, teams=rel.teams, event_slugs=rel.event_slugs,
            confidence=rel.confidence, safety=rel.safety, reason=rel.reason, kind="statistical",
        )

    async def _execute_all(self, report: ScanReport) -> None:
        queue = list(report.opportunities)
        if self.cfg.enable_statistical_arbs and self.cfg.execution_mode != "live":
            queue += report.statistical
        for opp in queue:
            res = await self.executor.execute(opp)
            report.actions.append(f"{opp.relation_subtype}: {res.action}")
        report.exposure = self.limiter.snapshot()

    # ── logging ──────────────────────────────────────────────────────────────

    def _write_logs(self, result: DiscoveryResult, graph: RelationGraph, report: ScanReport) -> None:
        out = self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "events.json").write_text(json.dumps([
            {"game_id": g.game_id, "label": g.label, "teams": list(g.teams),
             "events": [e.slug for e in g.events],
             "markets": [{"id": m.market_id, "kind": m.kind.value, "line": m.line,
                          "title": m.group_item_title or m.question} for m in g.markets]}
            for g in result.all_games
        ], indent=2), encoding="utf-8")
        (out / "relations.json").write_text(json.dumps([
            {"name": r.name, "type": r.relation_type, "subtype": r.relation_subtype,
             "payout": r.guaranteed_payout, "safety": r.safety, "confidence": r.confidence,
             "teams": list(r.teams), "reason": r.reason,
             "legs": [{"token_id": l.token_id, "side": l.side, "label": l.label} for l in r.legs]}
            for r in graph.relations
        ], indent=2), encoding="utf-8")
        # Subset relations as a watchlist YAML consumable by the existing paper-arb-sim.
        pairs = [relation_to_pair_dict(r, enabled=False) for r in graph.subset_relations]
        (out / "generated_worldcup_pairs.yaml").write_text(
            yaml.safe_dump({"notes": "Auto-discovered World Cup subset arbs. Paper-only; verify rules.",
                            "pairs": pairs}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        _append_opps_csv(out / "opportunities.csv", report.opportunities + report.statistical)
        _write_skips_csv(out / "skipped.csv", report.skipped)

    def render_table(self, report: ScanReport) -> str:
        rows = report.opportunities[: self.cfg.show_top]
        lines = [
            "time                 | event                | relation                     | legs | cost   | min_pay | edge    | size    | action",
            "-" * 130,
        ]
        ts = report.timestamp[11:19]
        for o in rows:
            event = (o.teams[0] if o.teams else (o.event_slugs[0] if o.event_slugs else o.game_id) or "?")
            lines.append(
                f"{ts:<20} | {str(event)[:20]:<20} | {o.relation_subtype[:28]:<28} | {len(o.legs):>4} | "
                f"{o.unit_cost:>6.4f} | {o.guaranteed_payout:>7.2f} | {o.worst_case_profit:>7.4f} | "
                f"{o.size:>7.2f} | {o.kind}"
            )
        if not rows:
            lines.append("(no deterministic opportunities this scan)")
        lines.append("")
        lines.append(
            f"discovered games={report.discovery.get('match_games')} futures={report.discovery.get('futures_groups')} "
            f"markets={report.discovery.get('total_markets')} | relations={report.relations.get('total')} | "
            f"deterministic={len(report.opportunities)} statistical={len(report.statistical)} "
            f"skipped={len(report.skipped)}"
        )
        return "\n".join(lines)


def _append_opps_csv(path: Path, opps: list[Opportunity]) -> None:
    fields = ["timestamp", "name", "kind", "relation_type", "relation_subtype", "guaranteed_payout",
              "unit_cost", "edge_per_share", "size", "capital", "worst_case_profit", "roi",
              "safety", "confidence", "game_id", "teams", "n_legs", "reason"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        ts = datetime.now(UTC).isoformat()
        for o in opps:
            w.writerow({
                "timestamp": ts, "name": o.name, "kind": o.kind, "relation_type": o.relation_type,
                "relation_subtype": o.relation_subtype, "guaranteed_payout": o.guaranteed_payout,
                "unit_cost": round(o.unit_cost, 6), "edge_per_share": round(o.edge_per_share, 6),
                "size": round(o.size, 4), "capital": round(o.capital, 4),
                "worst_case_profit": round(o.worst_case_profit, 6), "roi": round(o.roi, 6),
                "safety": o.safety, "confidence": o.confidence, "game_id": o.game_id,
                "teams": "|".join(o.teams), "n_legs": len(o.legs), "reason": o.reason,
            })


def _write_skips_csv(path: Path, skips: list[SkippedRelation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "relation_subtype", "reason", "net_unit_cost"])
        w.writeheader()
        for s in skips:
            w.writerow(asdict(s))
