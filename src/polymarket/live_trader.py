from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Pre-signing price offsets in ticks (±1 around current best ask)
_PRESIGN_TICK_OFFSETS = (-1, 0, 1)
# Minimum seconds before a cached signed order is considered stale
_PRESIGN_MAX_AGE_S = 8.0


LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_THIS_USES_REAL_MONEY"
LIVE_COMPLIANCE_PHRASE = "I_AM_ALLOWED_TO_TRADE_POLYMARKET"
LIVE_ENV_ENABLE_VALUE = "true"


class LiveTradingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveTradingConfig:
    host: str
    chain_id: int
    private_key: str
    signature_type: int
    funder: str | None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    max_session_spend: float = 0.0
    max_legs_per_bundle: int = 5

    @classmethod
    def from_env(
        cls,
        *,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        signature_type: int | None = None,
        max_session_spend: float,
        max_legs_per_bundle: int,
        confirmation: str,
        compliance_ack: str,
    ) -> "LiveTradingConfig":
        if confirmation != LIVE_CONFIRMATION_PHRASE:
            raise LiveTradingError(f"Live trading requires --live-confirmation {LIVE_CONFIRMATION_PHRASE!r}.")
        if compliance_ack != LIVE_COMPLIANCE_PHRASE:
            raise LiveTradingError(f"Live trading requires --live-compliance-ack {LIVE_COMPLIANCE_PHRASE!r}.")
        if os.getenv("POLYMARKET_LIVE_TRADING_ENABLED", "").strip().lower() != LIVE_ENV_ENABLE_VALUE:
            raise LiveTradingError("Set POLYMARKET_LIVE_TRADING_ENABLED=true to enable live trading.")
        if max_session_spend <= 0:
            raise LiveTradingError("Live trading requires a positive --live-max-session-spend.")

        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
        if not private_key:
            raise LiveTradingError("Missing POLYMARKET_PRIVATE_KEY.")
        return cls(
            host=os.getenv("POLYMARKET_CLOB_BASE_URL", host).strip() or host,
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", str(chain_id))),
            private_key=private_key,
            signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", str(signature_type if signature_type is not None else 2))),
            funder=os.getenv("POLYMARKET_FUNDER_ADDRESS", "").strip() or None,
            api_key=os.getenv("POLYMARKET_API_KEY", "").strip() or None,
            api_secret=os.getenv("POLYMARKET_API_SECRET", "").strip() or None,
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", "").strip() or None,
            max_session_spend=max_session_spend,
            max_legs_per_bundle=max(1, max_legs_per_bundle),
        )


@dataclass(frozen=True)
class LiveOrderLeg:
    label: str
    token_id: str
    size: float
    limit_price: float
    tick_size: str | None
    neg_risk: bool | None

    @property
    def notional_cap(self) -> float:
        return self.size * self.limit_price


@dataclass
class LiveOrderResult:
    success: bool
    requested_notional: float
    responses: list[Any] = field(default_factory=list)
    error: str | None = None
    signing_started_ns: int | None = None
    submission_started_ns: int | None = None
    ack_received_ns: int | None = None
    signing_ms: float | None = None
    submission_ms: float | None = None


class PresignedOrderCache:
    """Caches ECDSA-signed Polymarket order payloads computed during idle scan time.

    When a live entry fires, the critical path checks this cache first.  A cache
    hit skips the 1–5 ms secp256k1 signing step and goes directly to REST POST.

    The cache is keyed by (token_id, price_tick_int, size_int) tuples.
    price_tick_int = round(limit_price * 100);  size_int = round(size).
    Entries older than _PRESIGN_MAX_AGE_S are considered stale and re-signed.
    """

    def __init__(self, client: Any, OrderArgs: Any, PartialCreateOrderOptions: Any, BUY: Any) -> None:
        self._client = client
        self._OrderArgs = OrderArgs
        self._PartialCreateOrderOptions = PartialCreateOrderOptions
        self._BUY = BUY
        self._cache: dict[tuple[str, int, int], tuple[Any, float]] = {}

    def lookup(self, token_id: str, limit_price: float, size: float) -> Any | None:
        """Return a valid pre-signed order or None on cache miss / staleness."""
        key = (token_id, round(limit_price * 100), round(size))
        entry = self._cache.get(key)
        if entry is None:
            return None
        signed_order, signed_at = entry
        if time.monotonic() - signed_at >= _PRESIGN_MAX_AGE_S:
            del self._cache[key]
            return None
        return signed_order

    def prefetch_sync(
        self,
        token_id: str,
        limit_price: float,
        size: float,
        tick_size: str | None,
        neg_risk: bool | None,
    ) -> None:
        """Pre-sign orders at ±1 tick from limit_price for the given size.

        Designed to run in a background thread via asyncio.to_thread so it
        does not block the scan loop.
        """
        base_tick = round(limit_price * 100)
        size_int = round(size)
        if size_int <= 0:
            return
        for offset in _PRESIGN_TICK_OFFSETS:
            tick = base_tick + offset
            if tick <= 0 or tick >= 100:
                continue
            price = tick / 100.0
            key = (token_id, tick, size_int)
            entry = self._cache.get(key)
            if entry is not None and time.monotonic() - entry[1] < _PRESIGN_MAX_AGE_S:
                continue  # Still fresh
            try:
                order = self._OrderArgs(token_id=token_id, price=price, size=float(size_int), side=self._BUY)
                options = self._PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
                signed = self._client.create_order(order, options)
                self._cache[key] = (signed, time.monotonic())
            except Exception:  # noqa: BLE001 — do not crash the scan loop for pre-signing failures
                pass

    def evict_stale(self) -> None:
        now = time.monotonic()
        stale = [k for k, (_, ts) in self._cache.items() if now - ts >= _PRESIGN_MAX_AGE_S]
        for k in stale:
            del self._cache[k]


class LiveTrader:
    def __init__(self, config: LiveTradingConfig) -> None:
        self.config = config
        self.session_requested_spend = 0.0
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient, OrderArgs, OrderType, PartialCreateOrderOptions, PostOrdersV2Args
            from py_clob_client_v2.order_builder.constants import BUY
        except Exception as exc:  # pragma: no cover - exercised only when live extra is absent.
            raise LiveTradingError("Install live dependencies with: python -m pip install -e '.[live]'") from exc

        self._ApiCreds = ApiCreds
        self._ClobClient = ClobClient
        self._OrderArgs = OrderArgs
        self._OrderType = OrderType
        self._PartialCreateOrderOptions = PartialCreateOrderOptions
        self._PostOrdersV2Args = PostOrdersV2Args
        self._BUY = BUY
        self._client = self._build_client()
        self._presigned = PresignedOrderCache(
            self._client, self._OrderArgs, self._PartialCreateOrderOptions, self._BUY
        )

    def _build_client(self) -> Any:
        creds = None
        if self.config.api_key and self.config.api_secret and self.config.api_passphrase:
            creds = self._ApiCreds(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                api_passphrase=self.config.api_passphrase,
            )
        client = self._ClobClient(
            host=self.config.host,
            chain_id=self.config.chain_id,
            key=self.config.private_key,
            creds=creds,
            signature_type=self.config.signature_type,
            funder=self.config.funder,
        )
        if creds is None:
            client.set_api_creds(client.create_or_derive_api_key())
        return client

    async def prefetch_for_candidates(
        self,
        candidates: list[tuple[str, float, float, str | None, bool | None]],
    ) -> None:
        """Background pre-sign orders for likely next entries.

        Call from idle scan time (inside wait_for_updates or poll sleep).
        Each tuple is (token_id, limit_price, size, tick_size, neg_risk).
        Runs in a thread so ECDSA signing never blocks the event loop.
        """
        if not candidates:
            return
        self._presigned.evict_stale()
        await asyncio.to_thread(self._prefetch_sync, candidates)

    def _prefetch_sync(
        self,
        candidates: list[tuple[str, float, float, str | None, bool | None]],
    ) -> None:
        for token_id, limit_price, size, tick_size, neg_risk in candidates:
            self._presigned.prefetch_sync(token_id, limit_price, size, tick_size, neg_risk)

    async def buy_bundle_fok(self, legs: list[LiveOrderLeg]) -> LiveOrderResult:
        return await asyncio.to_thread(self._buy_bundle_fok_sync, legs)

    def _buy_bundle_fok_sync(self, legs: list[LiveOrderLeg]) -> LiveOrderResult:
        if not legs:
            return LiveOrderResult(success=False, requested_notional=0.0, error="empty live order bundle")
        if len(legs) > self.config.max_legs_per_bundle:
            return LiveOrderResult(
                success=False,
                requested_notional=sum(leg.notional_cap for leg in legs),
                error=f"bundle has {len(legs)} legs; max live legs is {self.config.max_legs_per_bundle}",
            )
        requested_notional = sum(leg.notional_cap for leg in legs)
        if self.session_requested_spend + requested_notional > self.config.max_session_spend + 1e-9:
            return LiveOrderResult(
                success=False,
                requested_notional=requested_notional,
                error="live max session spend would be exceeded",
            )
        signing_started_ns: int | None = None
        submission_started_ns: int | None = None
        ack_received_ns: int | None = None
        signing_started_perf_ns: int | None = None
        submission_started_perf_ns: int | None = None
        ack_received_perf_ns: int | None = None
        try:
            signed_orders = []
            signing_started_ns = time.time_ns()
            signing_started_perf_ns = time.perf_counter_ns()
            for leg in legs:
                cached = self._presigned.lookup(leg.token_id, leg.limit_price, leg.size)
                if cached is not None:
                    signed_orders.append(cached)
                    continue
                order = self._OrderArgs(
                    token_id=leg.token_id,
                    price=leg.limit_price,
                    size=leg.size,
                    side=self._BUY,
                )
                options = self._PartialCreateOrderOptions(tick_size=leg.tick_size, neg_risk=leg.neg_risk)
                signed_orders.append(self._client.create_order(order, options))
            submission_started_ns = time.time_ns()
            submission_started_perf_ns = time.perf_counter_ns()
            post_args = [
                self._PostOrdersV2Args(order=signed_order, orderType=self._OrderType.FOK)
                for signed_order in signed_orders
            ]
            responses = self._client.post_orders(post_args)
            ack_received_ns = time.time_ns()
            ack_received_perf_ns = time.perf_counter_ns()
            response_list = responses if isinstance(responses, list) else [responses]
            success = all(_response_success(response) for response in response_list)
            if success:
                self.session_requested_spend += requested_notional
            return LiveOrderResult(
                success=success,
                requested_notional=requested_notional,
                responses=response_list,
                signing_started_ns=signing_started_ns,
                submission_started_ns=submission_started_ns,
                ack_received_ns=ack_received_ns,
                signing_ms=(
                    (submission_started_perf_ns - signing_started_perf_ns) / 1_000_000
                    if signing_started_perf_ns is not None and submission_started_perf_ns is not None
                    else None
                ),
                submission_ms=(
                    (ack_received_perf_ns - submission_started_perf_ns) / 1_000_000
                    if submission_started_perf_ns is not None and ack_received_perf_ns is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - live order failures need to return to the scanner cleanly.
            ack_received_ns = time.time_ns()
            ack_received_perf_ns = time.perf_counter_ns()
            return LiveOrderResult(
                success=False,
                requested_notional=requested_notional,
                error=str(exc),
                signing_started_ns=signing_started_ns,
                submission_started_ns=submission_started_ns,
                ack_received_ns=ack_received_ns,
                signing_ms=(
                    (submission_started_perf_ns - signing_started_perf_ns) / 1_000_000
                    if signing_started_perf_ns is not None and submission_started_perf_ns is not None
                    else None
                ),
                submission_ms=(
                    (ack_received_perf_ns - submission_started_perf_ns) / 1_000_000
                    if submission_started_perf_ns is not None and ack_received_perf_ns is not None
                    else None
                ),
            )


def _response_success(response: Any) -> bool:
    if isinstance(response, dict):
        if "success" in response:
            return bool(response.get("success"))
        if "errorMsg" in response and response.get("errorMsg"):
            return False
        if "error" in response and response.get("error"):
            return False
        return bool(response.get("orderID") or response.get("id") or response.get("status"))
    success = getattr(response, "success", None)
    if success is not None:
        return bool(success)
    error = getattr(response, "errorMsg", None) or getattr(response, "error", None)
    if error:
        return False
    return bool(getattr(response, "orderID", None) or getattr(response, "id", None) or getattr(response, "status", None))
