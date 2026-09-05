"""Per-token/pair EMA fill-rate tracker.

Records whether entry attempts succeed (filled) or miss (price moved / depth
gone / REST recheck failed).  Provides a confidence score in [floor, 1.0] that
scales the minimum edge requirement:

    effective_min_edge = base_min_edge / confidence

A token that historically fills 50 % of attempts needs 2× the raw edge to
achieve the same expected return as a token that always fills.

Thread-safety: all operations are single dict reads/writes protected by the
CPython GIL.  No asyncio is required.
"""
from __future__ import annotations


class FillRateTracker:
    """Exponentially weighted moving-average fill-rate per key.

    Keys are typically pair names (for 2-leg entries) or N-leg spec names.
    """

    __slots__ = ("_ema_alpha", "_min_obs", "_floor", "_rates", "_counts")

    def __init__(
        self,
        ema_alpha: float = 0.15,
        min_observations: int = 3,
        confidence_floor: float = 0.2,
    ) -> None:
        """
        Parameters
        ----------
        ema_alpha:
            EMA decay weight for each new observation.  Higher = faster
            reaction, more volatile.  0.15 → half-life ≈ 4 observations.
        min_observations:
            Number of outcomes required before confidence deviates from 1.0.
            Until then, confidence is always 1.0 (no penalty for early bad luck).
        confidence_floor:
            Lower bound on confidence to prevent zero-division and
            excessive suppression.  Default 0.2 → never require more than 5×
            base edge.
        """
        self._ema_alpha = ema_alpha
        self._min_obs = min_observations
        self._floor = confidence_floor
        self._rates: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_outcome(self, key: str, *, filled: bool) -> None:
        """Record one entry outcome for ``key``.

        Call with ``filled=True`` when the trade is entered, ``filled=False``
        when the opportunity was identified but the fill failed for a
        *market-driven* reason (REST recheck no longer valid, depth gone,
        price moved).  Do NOT call for portfolio-constraint skips (cooldown,
        capital limits) — those say nothing about market liquidity.
        """
        current = self._rates.get(key, 1.0)
        a = self._ema_alpha
        self._rates[key] = (1.0 - a) * current + a * (1.0 if filled else 0.0)
        self._counts[key] = self._counts.get(key, 0) + 1

    # ── Reading ───────────────────────────────────────────────────────────────

    def confidence(self, key: str) -> float:
        """Return fill-rate confidence ∈ [floor, 1.0].

        Returns 1.0 until ``min_observations`` outcomes have been recorded for
        ``key`` so that early bad luck doesn't permanently suppress a trade.
        """
        if self._counts.get(key, 0) < self._min_obs:
            return 1.0
        return max(self._floor, self._rates.get(key, 1.0))

    def get_effective_min_edge(self, key: str, base_min_edge: float) -> float:
        """Return ``base_min_edge`` scaled by inverse fill confidence.

        Higher confidence → lower required edge (as expected).
        Lower confidence → higher required edge (compensate for misses).
        """
        return base_min_edge / self.confidence(key)

    def fill_rate(self, key: str) -> float | None:
        """Raw EMA fill rate, or None if ``key`` has never been observed."""
        return self._rates.get(key)

    def observation_count(self, key: str) -> int:
        """Number of outcomes recorded for ``key``."""
        return self._counts.get(key, 0)

    def __repr__(self) -> str:
        return (
            f"FillRateTracker(tracked={len(self._rates)}, "
            f"alpha={self._ema_alpha}, floor={self._floor})"
        )
