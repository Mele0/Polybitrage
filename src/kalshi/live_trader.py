"""Kalshi live trading via FIX Order Entry (port 8228).

Provides the same ``buy_bundle_fok`` interface as
``src.polymarket.live_trader.LiveTrader`` so it integrates transparently with
``PaperArbSimulator._execute_live_buy_bundle``.

Usage in simulator
------------------
    # Set --exchange kalshi --execution-mode live --book-source fix
    # KALSHI_FIX_OE_API_KEY, KALSHI_FIX_OE_API_SECRET env vars required
    # KALSHI_LIVE_TRADING_ENABLED=true also required

    # The simulator will call _initialize_live_trader() which instantiates
    # this class instead of polymarket.LiveTrader when exchange == "kalshi".
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from src.kalshi.client import parse_kalshi_token_id
from src.kalshi.fix_order_entry import FixOrderResult, KalshiFixOrderEntryClient
from src.polymarket.live_trader import LiveOrderLeg, LiveOrderResult

KALSHI_LIVE_ENV_VALUE = "true"
KALSHI_LIVE_CONFIRMATION = "I_UNDERSTAND_THIS_USES_REAL_MONEY_ON_KALSHI"
KALSHI_LIVE_COMPLIANCE = "I_AM_ALLOWED_TO_TRADE_KALSHI"


class KalshiLiveTradingError(RuntimeError):
    pass


@dataclass
class KalshiLiveTradingConfig:
    api_key: str
    api_secret: str
    host: str = KalshiFixOrderEntryClient.HOST
    port: int = KalshiFixOrderEntryClient.PORT_FAST
    max_session_spend: float = 0.0
    max_legs_per_bundle: int = 5
    order_timeout_seconds: float = 5.0

    @classmethod
    def from_env(
        cls,
        *,
        confirmation: str,
        compliance_ack: str,
        max_session_spend: float,
        max_legs_per_bundle: int,
        host: str = KalshiFixOrderEntryClient.HOST,
        port: int = KalshiFixOrderEntryClient.PORT_FAST,
    ) -> "KalshiLiveTradingConfig":
        if confirmation != KALSHI_LIVE_CONFIRMATION:
            raise KalshiLiveTradingError(
                f"Kalshi live trading requires --live-confirmation {KALSHI_LIVE_CONFIRMATION!r}."
            )
        if compliance_ack != KALSHI_LIVE_COMPLIANCE:
            raise KalshiLiveTradingError(
                f"Kalshi live trading requires --live-compliance-ack {KALSHI_LIVE_COMPLIANCE!r}."
            )
        if os.getenv("KALSHI_LIVE_TRADING_ENABLED", "").strip().lower() != KALSHI_LIVE_ENV_VALUE:
            raise KalshiLiveTradingError("Set KALSHI_LIVE_TRADING_ENABLED=true to enable Kalshi live trading.")
        if max_session_spend <= 0:
            raise KalshiLiveTradingError("Kalshi live trading requires a positive --live-max-session-spend.")
        api_key = os.environ.get("KALSHI_FIX_OE_API_KEY", "").strip()
        api_secret = os.environ.get("KALSHI_FIX_OE_API_SECRET", "").strip()
        if not api_key or not api_secret:
            raise KalshiLiveTradingError(
                "Missing KALSHI_FIX_OE_API_KEY or KALSHI_FIX_OE_API_SECRET environment variables."
            )
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            host=host,
            port=port,
            max_session_spend=max_session_spend,
            max_legs_per_bundle=max(1, max_legs_per_bundle),
        )


class KalshiLiveTrader:
    """Live order execution for Kalshi via FIX Order Entry (port 8228).

    Maintains a persistent FIX session. Each ``buy_bundle_fok`` call
    submits all legs concurrently and awaits their ExecutionReports.
    """

    def __init__(self, config: KalshiLiveTradingConfig) -> None:
        self.config = config
        self.session_requested_spend = 0.0
        self._fix_client = KalshiFixOrderEntryClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            host=config.host,
            port=config.port,
        )
        self._connected = False

    async def start(self) -> None:
        await self._fix_client.connect()
        self._connected = True

    async def stop(self) -> None:
        await self._fix_client.close()
        self._connected = False

    async def buy_bundle_fok(self, legs: list[LiveOrderLeg]) -> LiveOrderResult:
        """Submit all legs as FOK orders concurrently via FIX.

        Compatible with the polymarket LiveTrader interface so it slots into
        PaperArbSimulator._execute_live_buy_bundle without modification.
        """
        if not legs:
            return LiveOrderResult(success=False, requested_notional=0.0, error="empty bundle")
        if len(legs) > self.config.max_legs_per_bundle:
            return LiveOrderResult(
                success=False,
                requested_notional=sum(leg.notional_cap for leg in legs),
                error=f"bundle has {len(legs)} legs; max is {self.config.max_legs_per_bundle}",
            )
        requested_notional = sum(leg.notional_cap for leg in legs)
        if self.session_requested_spend + requested_notional > self.config.max_session_spend + 1e-9:
            return LiveOrderResult(
                success=False,
                requested_notional=requested_notional,
                error="Kalshi live max session spend would be exceeded",
            )
        if not self._fix_client.connected:
            return LiveOrderResult(success=False, requested_notional=requested_notional, error="FIX OE not connected")

        signing_started_ns = time.time_ns()
        signing_started_perf = time.perf_counter_ns()

        # Submit all legs concurrently
        tasks = [
            self._fix_client.submit_fok_order(
                _ticker_from_leg(leg),
                "buy",
                leg.limit_price,
                leg.size,
                timeout=self.config.order_timeout_seconds,
            )
            for leg in legs
        ]
        submission_started_ns = time.time_ns()
        submission_started_perf = time.perf_counter_ns()
        fix_results: list[FixOrderResult] = await asyncio.gather(*tasks)
        ack_ns = time.time_ns()
        ack_perf = time.perf_counter_ns()

        success = all(r.success for r in fix_results)
        if success:
            self.session_requested_spend += requested_notional

        return LiveOrderResult(
            success=success,
            requested_notional=requested_notional,
            responses=[
                {
                    "success": r.success,
                    "order_id": r.order_id,
                    "filled_qty": r.filled_qty,
                    "avg_price": r.avg_price,
                    "error": r.error,
                    "round_trip_ms": r.round_trip_ms,
                }
                for r in fix_results
            ],
            error="; ".join(r.error for r in fix_results if r.error) or None,
            signing_started_ns=signing_started_ns,
            submission_started_ns=submission_started_ns,
            ack_received_ns=ack_ns,
            signing_ms=(submission_started_perf - signing_started_perf) / 1_000_000,
            submission_ms=(ack_perf - submission_started_perf) / 1_000_000,
        )


def _ticker_from_leg(leg: LiveOrderLeg) -> str:
    """Extract the Kalshi market ticker from a LiveOrderLeg token_id."""
    parsed = parse_kalshi_token_id(leg.token_id)
    if parsed is None:
        raise ValueError(f"Cannot parse Kalshi ticker from token_id: {leg.token_id!r}")
    return parsed.ticker
