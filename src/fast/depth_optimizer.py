"""High-performance depth optimizer for the arbitrage scan hot path.

Provides ``find_best_size_fast`` — a drop-in replacement for the Python float loop
in ``simulator._find_best_size``.

Dispatch chain (fastest available at import time):
  1. Numba @njit  — C-speed compiled loop (~0.5 µs for 10-level book, 60 candidates)
  2. NumPy vectorised — searchsorted-based batch VWAP (~5 µs)
  3. Pure Python float — fallback from Phase 1 (always available, ~30 µs)

The first call to the Numba path triggers JIT compilation (~200 ms, cached to disk
afterwards). NumPy is the practical fast path when Numba is not installed.

Usage
-----
    from src.fast.depth_optimizer import find_best_size_fast, BACKEND

    result = find_best_size_fast(
        parent_float_asks,   # list[tuple[float, float]] — (price, size)
        child_float_asks,
        fee_rate_1, fee_rate_2,
        slippage, fee_buffer,
        entry_threshold,
        min_edge,
        capital_limit,
        min_trade_size,
        candidates,          # set[float]
    )
    # result: dict with keys size, capital, profit, net, parent_avg, child_avg
    #         or {} if no valid size found
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ── Numba path ────────────────────────────────────────────────────────────────

try:
    import numba

    @numba.njit(cache=True, nogil=True)
    def _vwap_jit(prices: np.ndarray, sizes: np.ndarray, target: float) -> float:
        """JIT-compiled VWAP.  Returns -1.0 if depth is insufficient."""
        remaining = target
        total = 0.0
        filled = 0.0
        for i in range(len(prices)):
            take = sizes[i] if sizes[i] < remaining else remaining
            if take <= 0.0:
                continue
            total += take * prices[i]
            filled += take
            remaining -= take
            if remaining <= 1e-12:
                break
        return total / target if filled >= target - 1e-9 else -1.0

    @numba.njit(cache=True, nogil=True)
    def _find_best_size_jit(
        pp: np.ndarray, ps: np.ndarray,
        cp: np.ndarray, cs: np.ndarray,
        fee1: float, fee2: float,
        slippage: float, fee_buffer: float,
        entry_threshold: float, min_edge: float,
        capital_limit: float, min_trade_size: float,
        candidates: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Returns (best_size, best_profit, best_net, best_parent_avg, best_child_avg).
        All -1.0 if no valid candidate found.
        """
        best_size = -1.0
        best_profit = -1e18
        best_net = -1.0
        best_pp = -1.0
        best_cp = -1.0

        for i in range(len(candidates)):
            size = candidates[i]
            if size <= 0.0:
                continue
            parent_avg = _vwap_jit(pp, ps, size)
            child_avg = _vwap_jit(cp, cs, size)
            if parent_avg < 0.0 or child_avg < 0.0:
                continue
            fee = fee1 * parent_avg * (1.0 - parent_avg) + fee2 * child_avg * (1.0 - child_avg)
            net = parent_avg + child_avg + fee + slippage + fee_buffer
            capital = size * net
            profit = size * (1.0 - net)
            if size < min_trade_size / (net if net > 0.01 else 0.01):
                continue
            if capital > capital_limit + 1e-9:
                continue
            if net >= entry_threshold:
                continue
            if (1.0 - net) < min_edge:
                continue
            if profit <= 0.0:
                continue
            if profit > best_profit:
                best_profit = profit
                best_size = size
                best_net = net
                best_pp = parent_avg
                best_cp = child_avg

        return best_size, best_profit, best_net, best_pp, best_cp

    def _find_best_size_numba(
        parent_float_asks: list[tuple[float, float]],
        child_float_asks: list[tuple[float, float]],
        fee_rate_1: float,
        fee_rate_2: float,
        slippage: float,
        fee_buffer: float,
        entry_threshold: float,
        min_edge: float,
        capital_limit: float,
        min_trade_size: float,
        candidates: set[float],
    ) -> dict[str, float]:
        if not parent_float_asks or not child_float_asks or not candidates:
            return {}
        pp = np.array([x[0] for x in parent_float_asks], dtype=np.float64)
        ps = np.array([x[1] for x in parent_float_asks], dtype=np.float64)
        cp = np.array([x[0] for x in child_float_asks], dtype=np.float64)
        cs = np.array([x[1] for x in child_float_asks], dtype=np.float64)
        cands = np.array(sorted(candidates), dtype=np.float64)
        size, profit, net, parent_avg, child_avg = _find_best_size_jit(
            pp, ps, cp, cs,
            fee_rate_1, fee_rate_2,
            slippage, fee_buffer,
            entry_threshold, min_edge,
            capital_limit, min_trade_size,
            cands,
        )
        if size < 0:
            return {}
        return {
            "size": size,
            "capital": size * net,
            "profit": profit,
            "net": net,
            "parent_avg": parent_avg,
            "child_avg": child_avg,
        }

    BACKEND = "numba"
    find_best_size_fast = _find_best_size_numba

except ImportError:
    BACKEND = None  # set below

# ── NumPy vectorised path (always available since numpy is a dep) ─────────────


def _batch_vwap(cum_sizes: np.ndarray, cum_costs: np.ndarray, prices: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fully vectorised VWAP for ALL target sizes in one call.

    Uses np.searchsorted (C-speed binary search) to find the fill level for
    each target simultaneously.  Returns nan for targets with insufficient depth.
    """
    idxs = np.searchsorted(cum_sizes, targets, side="left")
    vwaps = np.full(len(targets), np.nan, dtype=np.float64)

    valid = idxs < len(cum_sizes)

    # Targets filled entirely from the first level
    first = valid & (idxs == 0)
    if np.any(first):
        vwaps[first] = prices[0]

    # Targets requiring multiple levels
    multi = valid & (idxs > 0)
    if np.any(multi):
        im = idxs[multi]
        partial = targets[multi] - cum_sizes[im - 1]
        total = cum_costs[im - 1] + partial * prices[im]
        vwaps[multi] = total / targets[multi]

    return vwaps


def _find_best_size_numpy(
    parent_float_asks: list[tuple[float, float]],
    child_float_asks: list[tuple[float, float]],
    fee_rate_1: float,
    fee_rate_2: float,
    slippage: float,
    fee_buffer: float,
    entry_threshold: float,
    min_edge: float,
    capital_limit: float,
    min_trade_size: float,
    candidates: set[float],
) -> dict[str, float]:
    """Truly vectorised optimizer: all 60 candidate sizes processed in one batch.

    Key improvement over a per-candidate loop: np.searchsorted, cumsum, and all
    filter masks execute at C-speed with no Python-level iteration over candidates.
    """
    if not parent_float_asks or not child_float_asks or not candidates:
        return {}

    pp = np.fromiter((x[0] for x in parent_float_asks), dtype=np.float64, count=len(parent_float_asks))
    ps = np.fromiter((x[1] for x in parent_float_asks), dtype=np.float64, count=len(parent_float_asks))
    cp = np.fromiter((x[0] for x in child_float_asks), dtype=np.float64, count=len(child_float_asks))
    cs = np.fromiter((x[1] for x in child_float_asks), dtype=np.float64, count=len(child_float_asks))

    # Cumulative arrays built once (O(N_levels))
    p_cum = np.cumsum(ps)
    p_cost_cum = np.cumsum(pp * ps)
    c_cum = np.cumsum(cs)
    c_cost_cum = np.cumsum(cp * cs)

    # All candidate sizes as a sorted numpy array
    sizes = np.fromiter(sorted(candidates), dtype=np.float64, count=len(candidates))
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return {}

    # Batch VWAP for parent and child — single searchsorted call each
    p_vwaps = _batch_vwap(p_cum, p_cost_cum, pp, sizes)
    c_vwaps = _batch_vwap(c_cum, c_cost_cum, cp, sizes)

    # Vectorised cost/profit/filter — no Python loop
    valid = ~(np.isnan(p_vwaps) | np.isnan(c_vwaps))
    if not np.any(valid):
        return {}

    fees = fee_rate_1 * p_vwaps * (1.0 - p_vwaps) + fee_rate_2 * c_vwaps * (1.0 - c_vwaps)
    nets = p_vwaps + c_vwaps + fees + slippage + fee_buffer
    capitals = sizes * nets
    profits = sizes * (1.0 - nets)

    mask = (
        valid
        & (sizes >= min_trade_size / np.maximum(nets, 0.01))
        & (capitals <= capital_limit + 1e-9)
        & (nets < entry_threshold)
        & ((1.0 - nets) >= min_edge)
        & (profits > 0.0)
    )
    if not np.any(mask):
        return {}

    # Best: highest profit among valid candidates
    masked_profits = np.where(mask, profits, -np.inf)
    best_idx = int(np.argmax(masked_profits))

    return {
        "size": float(sizes[best_idx]),
        "capital": float(capitals[best_idx]),
        "profit": float(profits[best_idx]),
        "net": float(nets[best_idx]),
        "parent_avg": float(p_vwaps[best_idx]),
        "child_avg": float(c_vwaps[best_idx]),
    }


# ── Pure Python fallback ──────────────────────────────────────────────────────


def _float_avg(levels: list[tuple[float, float]], size: float) -> float:
    remaining, total, filled = size, 0.0, 0.0
    for price, lsize in levels:
        take = min(lsize, remaining)
        if take <= 1e-12:
            continue
        total += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    return total / size if filled >= size - 1e-9 else -1.0


def _find_best_size_python(
    parent_float_asks: list[tuple[float, float]],
    child_float_asks: list[tuple[float, float]],
    fee_rate_1: float,
    fee_rate_2: float,
    slippage: float,
    fee_buffer: float,
    entry_threshold: float,
    min_edge: float,
    capital_limit: float,
    min_trade_size: float,
    candidates: set[float],
) -> dict[str, float]:
    best: dict[str, float] = {}
    best_profit = -1e18
    for size in sorted(candidates):
        if size <= 0:
            continue
        parent_avg = _float_avg(parent_float_asks, size)
        child_avg = _float_avg(child_float_asks, size)
        if parent_avg < 0 or child_avg < 0:
            continue
        fee = fee_rate_1 * parent_avg * (1.0 - parent_avg) + fee_rate_2 * child_avg * (1.0 - child_avg)
        net = parent_avg + child_avg + fee + slippage + fee_buffer
        capital = size * net
        profit = size * (1.0 - net)
        if size < min_trade_size / max(net, 0.01):
            continue
        if capital > capital_limit + 1e-9:
            continue
        if net >= entry_threshold:
            continue
        if (1.0 - net) < min_edge:
            continue
        if profit <= 0:
            continue
        if profit > best_profit:
            best_profit = profit
            best = {
                "size": size,
                "capital": capital,
                "profit": profit,
                "net": net,
                "parent_avg": parent_avg,
                "child_avg": child_avg,
            }
    return best


# ── Dispatcher ────────────────────────────────────────────────────────────────
# numpy array construction costs ~15-30µs for small (≤10 element) books, which
# exceeds the savings from vectorised computation.  The numpy path only beats
# pure Python when there are 20+ price levels in the book.  We therefore use
# an adaptive dispatcher: numpy for deep books, Python for shallow ones.
# With Numba installed the compiled path always wins (C-speed, negligible
# array-creation overhead after the first JIT compilation).

_NUMPY_LEVEL_THRESHOLD = 20


def _find_best_size_adaptive(
    parent_float_asks: list[tuple[float, float]],
    child_float_asks: list[tuple[float, float]],
    fee_rate_1: float,
    fee_rate_2: float,
    slippage: float,
    fee_buffer: float,
    entry_threshold: float,
    min_edge: float,
    capital_limit: float,
    min_trade_size: float,
    candidates: set[float],
) -> dict[str, float]:
    if max(len(parent_float_asks), len(child_float_asks)) >= _NUMPY_LEVEL_THRESHOLD:
        return _find_best_size_numpy(
            parent_float_asks, child_float_asks,
            fee_rate_1, fee_rate_2, slippage, fee_buffer,
            entry_threshold, min_edge, capital_limit, min_trade_size, candidates,
        )
    return _find_best_size_python(
        parent_float_asks, child_float_asks,
        fee_rate_1, fee_rate_2, slippage, fee_buffer,
        entry_threshold, min_edge, capital_limit, min_trade_size, candidates,
    )


if BACKEND is None:
    BACKEND = "numpy+adaptive"
    find_best_size_fast = _find_best_size_adaptive
