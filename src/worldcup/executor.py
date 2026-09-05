"""Execution safety layer for World Cup arbitrage.

Provides:
* :class:`ExposureLimiter` — kill switch + per-event / per-team / per-relation /
  daily-loss caps, enforced before any entry.
* :class:`WorldCupExecutor` — paper / dry-run / live execution of a validated
  :class:`Opportunity`.  Live mode reuses the guarded :class:`~src.polymarket.live_trader.LiveTrader`
  (FOK bundle posting); paper/dry-run never touch the network or a wallet.

Conservative by design: an opportunity must already be validated by the state
validator and pass every exposure cap, and live mode requires the same explicit
confirmation phrases / env vars as the rest of the bot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OpportunityLeg:
    token_id: str
    side: str                 # YES | NO (token bought)
    label: str
    price: float              # executable VWAP price for the chosen size
    size: float               # shares
    tick_size: float | None = None
    neg_risk: bool | None = None


@dataclass
class Opportunity:
    name: str
    relation_type: str
    relation_subtype: str
    legs: list[OpportunityLeg]
    guaranteed_payout: float
    size: float               # shares per leg (uniform basket size)
    unit_cost: float          # sum of leg prices
    capital: float            # size * unit_cost
    worst_case_profit: float  # size * (payout - unit_cost)  (state-validated)
    roi: float
    edge_per_share: float     # payout - unit_cost
    game_id: str | None
    teams: tuple[str, ...]
    event_slugs: tuple[str, ...]
    confidence: str
    safety: str
    reason: str
    kind: str = "deterministic"  # deterministic | statistical
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_team_keys(self) -> tuple[str, ...]:
        return self.teams or (self.event_slugs[0],) if self.event_slugs else ()


@dataclass
class ExposureConfig:
    max_event_exposure: float = 100.0
    max_team_exposure: float = 75.0
    max_relation_exposure: float = 150.0
    max_total_exposure: float = 500.0
    max_daily_loss: float = 100.0
    kill_switch_path: Path | None = None     # presence of this file halts trading


class ExposureLimiter:
    def __init__(self, config: ExposureConfig) -> None:
        self.config = config
        self.by_event: dict[str, float] = {}
        self.by_team: dict[str, float] = {}
        self.by_relation: dict[str, float] = {}
        self.total: float = 0.0
        self.realized_pnl_today: float = 0.0
        self._halted_reason: str | None = None

    def halt(self, reason: str) -> None:
        self._halted_reason = reason

    @property
    def halted(self) -> bool:
        return self._halted_reason is not None

    def _kill_switch_active(self) -> bool:
        path = self.config.kill_switch_path
        return bool(path and path.exists())

    def can_enter(self, opp: Opportunity) -> tuple[bool, str]:
        if self._kill_switch_active():
            return False, f"kill switch present ({self.config.kill_switch_path})"
        if self.halted:
            return False, f"trading halted: {self._halted_reason}"
        if -self.realized_pnl_today >= self.config.max_daily_loss:
            return False, f"daily loss cap reached ({self.realized_pnl_today:.2f})"
        cap = opp.capital
        event_key = opp.event_slugs[0] if opp.event_slugs else (opp.game_id or "unknown")
        if self.by_event.get(event_key, 0.0) + cap > self.config.max_event_exposure + 1e-9:
            return False, f"max event exposure exceeded for {event_key}"
        for team in opp.primary_team_keys:
            if self.by_team.get(team, 0.0) + cap > self.config.max_team_exposure + 1e-9:
                return False, f"max team exposure exceeded for {team}"
        if self.by_relation.get(opp.relation_subtype, 0.0) + cap > self.config.max_relation_exposure + 1e-9:
            return False, f"max relation exposure exceeded for {opp.relation_subtype}"
        if self.total + cap > self.config.max_total_exposure + 1e-9:
            return False, "max total exposure exceeded"
        return True, "ok"

    def commit(self, opp: Opportunity) -> None:
        cap = opp.capital
        event_key = opp.event_slugs[0] if opp.event_slugs else (opp.game_id or "unknown")
        self.by_event[event_key] = self.by_event.get(event_key, 0.0) + cap
        for team in opp.primary_team_keys:
            self.by_team[team] = self.by_team.get(team, 0.0) + cap
        self.by_relation[opp.relation_subtype] = self.by_relation.get(opp.relation_subtype, 0.0) + cap
        self.total += cap

    def record_pnl(self, pnl: float) -> None:
        self.realized_pnl_today += pnl

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_exposure": round(self.total, 2),
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "events": {k: round(v, 2) for k, v in self.by_event.items()},
            "relations": {k: round(v, 2) for k, v in self.by_relation.items()},
            "halted": self.halted,
        }


@dataclass
class ExecutionResult:
    accepted: bool
    mode: str
    action: str
    error: str | None = None
    requested_notional: float = 0.0


class WorldCupExecutor:
    """Routes a validated opportunity to paper, dry-run, or live execution."""

    def __init__(
        self,
        limiter: ExposureLimiter,
        *,
        mode: str = "paper",       # paper | dry_run | live
        live_trader: Any | None = None,
    ) -> None:
        self.limiter = limiter
        self.mode = mode
        self.live_trader = live_trader

    async def execute(self, opp: Opportunity) -> ExecutionResult:
        ok, reason = self.limiter.can_enter(opp)
        if not ok:
            return ExecutionResult(False, self.mode, f"skipped: {reason}", error=reason)

        plan = (
            f"{opp.relation_subtype} size={opp.size:.2f} cost/u={opp.unit_cost:.4f} "
            f"payout={opp.guaranteed_payout:g} worst_profit={opp.worst_case_profit:.4f} "
            f"capital={opp.capital:.2f} legs=[{', '.join(f'{l.side}:{l.label}@{l.price:.4f}' for l in opp.legs)}]"
        )

        if self.mode == "dry_run":
            return ExecutionResult(True, self.mode, f"DRY-RUN would buy {plan}", requested_notional=opp.capital)

        if self.mode == "paper":
            self.limiter.commit(opp)
            return ExecutionResult(True, self.mode, f"PAPER entered {plan}", requested_notional=opp.capital)

        if self.mode == "live":
            if self.live_trader is None:
                return ExecutionResult(False, self.mode, "live skipped: no live trader configured", error="no_live_trader")
            from src.polymarket.live_trader import LiveOrderLeg  # lazy: live extra optional
            legs = [
                LiveOrderLeg(
                    label=l.label,
                    token_id=l.token_id,
                    size=l.size,
                    limit_price=l.price,
                    tick_size=(str(l.tick_size) if l.tick_size else None),
                    neg_risk=l.neg_risk,
                )
                for l in opp.legs
            ]
            result = await self.live_trader.buy_bundle_fok(legs)
            if not result.success:
                # FOK means each leg is all-or-nothing; Polymarket is not cross-leg
                # atomic, so a partial bundle is possible. Policy: do NOT chase —
                # surface the error, leave any filled legs flat, and let the operator
                # decide. The limiter is not committed on failure.
                return ExecutionResult(False, self.mode, f"live rejected: {result.error}",
                                       error=result.error, requested_notional=result.requested_notional)
            self.limiter.commit(opp)
            return ExecutionResult(True, self.mode, f"LIVE filled {plan}", requested_notional=result.requested_notional)

        return ExecutionResult(False, self.mode, f"skipped: unknown mode {self.mode}", error="unknown_mode")


def default_kill_switch_path(base: Path | None = None) -> Path:
    return (base or Path("data")) / "worldcup_kill_switch"
