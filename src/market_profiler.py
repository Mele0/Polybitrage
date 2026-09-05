"""Order-book microstructure aggregator (zero extra fetches, off the hot path).

Stale, crossed, or one-sided books are where an arb bot bleeds — you act on a
quote that isn't really there (adverse selection).  This module turns the
``OrderBook`` objects the scanner *already* reconstructed each scan into market-
health aggregates: spread, depth, top-of-book imbalance, and crossed / locked /
one-sided counts across the recently-updated token set.

Flow (all O(1) per book — every field below is a cached scalar on ``OrderBook``):
  * ``observe_books(books)`` — called per scan; appends cheap samples to bounded
    accumulators.  No fetching, no reconstruction.
  * ``maybe_flush(now)`` — at most once per ``interval`` recomputes percentile
    aggregates and resets the accumulators.
  * ``snapshot()`` — the cached aggregates, read by the /metrics scrape.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_CAP = 20_000  # max samples retained between flushes (guards a full-scan burst)

_spreads: list[float] = []
_depths: list[float] = []
_imbalances: list[float] = []
_mids: list[float] = []
_counts = {"observed": 0, "two_sided": 0, "crossed": 0, "locked": 0, "one_sided": 0, "empty": 0}
_last_flush: float = 0.0
_LATEST: dict[str, float] = {}


def observe_books(books: dict[str, Any]) -> None:
    """Accumulate cheap microstructure samples from books already in hand."""
    for b in books.values():
        if b is None:
            _counts["empty"] += 1
            continue
        _counts["observed"] += 1
        bb = b.best_bid
        ba = b.best_ask
        bd = b.bid_depth
        ad = b.ask_depth
        if bb is not None and ba is not None:
            _counts["two_sided"] += 1
            sp = ba - bb
            if len(_spreads) < _CAP:
                _spreads.append(sp)
                _mids.append((ba + bb) * 0.5)
            if sp < 0:
                _counts["crossed"] += 1      # bid > ask: data error / arbitrageable
            elif sp == 0:
                _counts["locked"] += 1       # bid == ask
        elif bb is None and ba is None:
            _counts["empty"] += 1
        else:
            _counts["one_sided"] += 1        # only one side present
        tot = bd + ad
        if tot > 0 and len(_depths) < _CAP:
            _depths.append(tot)
            _imbalances.append((bd - ad) / tot)  # +1 all-bid … -1 all-ask


def maybe_flush(now: float, interval: float = 1.0) -> None:
    """Recompute aggregates at most once per ``interval`` and reset accumulators."""
    global _last_flush, _LATEST
    if now - _last_flush < interval:
        return
    _last_flush = now
    out: dict[str, float] = {float_k: float(v) for float_k, v in _counts.items()}
    if _spreads:
        sp = np.asarray(_spreads)
        p = np.percentile(sp, [50, 90, 99])
        out.update(spread_median=round(float(p[0]), 5), spread_p90=round(float(p[1]), 5),
                   spread_p99=round(float(p[2]), 5),
                   spread_min=round(float(sp.min()), 5), spread_max=round(float(sp.max()), 5))
    if _depths:
        dp = np.asarray(_depths)
        out.update(depth_median=round(float(np.percentile(dp, 50)), 2),
                   depth_total=round(float(dp.sum()), 2))
    if _imbalances:
        out["imbalance_median"] = round(float(np.median(_imbalances)), 4)
    obs = _counts["observed"] or 1
    out["two_sided_frac"] = round(_counts["two_sided"] / obs, 4)
    _LATEST = out
    # reset for the next window
    _spreads.clear(); _depths.clear(); _imbalances.clear(); _mids.clear()
    for k in _counts:
        _counts[k] = 0


def snapshot() -> dict[str, float]:
    return _LATEST
