"""Time-to-resolution sizing model for prediction-market arbitrage.

Positions in markets that resolve sooner are capital-efficient: the same edge
is returned and recycled in days rather than months.  The sizing multiplier
scales position size up for fast-resolving markets and down for long-dated ones.

Formula:
    multiplier = clip(sqrt(target_days / max(days_to_resolution, 1)), min_m, max_m)

Examples with target_days=30:
    1 day   → sqrt(30/1)   ≈ 5.48 → clipped to max_multiplier (e.g. 2.0)
   10 days  → sqrt(30/10)  ≈ 1.73
   30 days  → sqrt(30/30)  = 1.00 (baseline)
   90 days  → sqrt(30/90)  ≈ 0.58
  365 days  → sqrt(30/365) ≈ 0.29 → clipped to min_multiplier (e.g. 0.3)

The square-root (rather than linear) scaling dampens extremes: we don't 10× a
1-day position and don't cut a 365-day position to near-zero.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any


def resolution_size_multiplier(
    days_to_resolution: float | None,
    *,
    target_days: float = 30.0,
    max_multiplier: float = 2.0,
    min_multiplier: float = 0.3,
) -> float:
    """Sizing multiplier in [min_multiplier, max_multiplier].

    Returns 1.0 for unknown / non-positive days (no adjustment when the
    resolution date is not configured).
    """
    if days_to_resolution is None or days_to_resolution <= 0:
        return 1.0
    raw = math.sqrt(target_days / max(days_to_resolution, 1.0))
    return max(min_multiplier, min(max_multiplier, raw))


def pair_days_to_resolution(pair_raw: dict[str, Any]) -> float | None:
    """Extract days-to-resolution from a PairConfig.raw payload.

    Tries common YAML/API key names in order.  Returns None if no parseable
    future date is found.

    Recognised keys (checked in order):
        end_date, end_date_iso, end_time, close_time, resolution_time,
        close_date, expiry_date
    """
    _KEYS = (
        "end_date", "end_date_iso", "end_time",
        "close_time", "resolution_time", "close_date", "expiry_date",
    )
    for key in _KEYS:
        val = pair_raw.get(key)
        if not val:
            continue
        try:
            s = str(val).strip().replace("Z", "+00:00")
            end_dt = datetime.fromisoformat(s)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=UTC)
            delta = (end_dt - datetime.now(UTC)).total_seconds()
            return max(0.0, delta / 86400.0)
        except (ValueError, TypeError, AttributeError):
            continue
    return None
