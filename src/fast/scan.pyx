# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
"""Cython hot-path scan classifier — compiled to C with -O3 -march=native.

Build:
    python setup_cython.py build_ext --inplace
"""
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# C-level integer constants (avoids GIL requirement inside loops)
cdef int _REJECTED   = 0
cdef int _NEAR_ARB   = 1
cdef int _EXECUTABLE = 2

# Exported Python names so callers can compare against them
REJECTED   = 0
NEAR_ARB   = 1
EXECUTABLE = 2


# ── Pair observation batch classifier ────────────────────────────────────────

cpdef void scan_classify_batch(
    double[::1] parent_asks,
    double[::1] child_asks,
    double[::1] fee_rates_1,
    double[::1] fee_rates_2,
    double[::1] slippages,
    double[::1] fee_buffers,
    double[::1] min_edges,
    double near_threshold,
    double entry_threshold,
    int n,
    double[::1] gross_out,
    double[::1] net_out,
    int[::1] class_out,
):
    """Classify all N pairs in a tight C loop.

    All inputs are C-contiguous typed memoryviews.  Cython compiles the inner
    loop to pure C arithmetic with no Python overhead.
    """
    cdef int i
    cdef double pa, ca, gross, fee, net

    for i in range(n):
        pa    = parent_asks[i]
        ca    = child_asks[i]
        gross = pa + ca
        gross_out[i] = gross

        fee = (fee_rates_1[i] * pa * (1.0 - pa)
               + fee_rates_2[i] * ca * (1.0 - ca))
        net = gross + fee + slippages[i] + fee_buffers[i]
        net_out[i] = net

        if net < entry_threshold and (1.0 - net) >= min_edges[i]:
            class_out[i] = _EXECUTABLE
        elif net <= near_threshold:
            class_out[i] = _NEAR_ARB
        else:
            class_out[i] = _REJECTED


# ── N-leg VWAP inner loop ─────────────────────────────────────────────────────

cdef double _vwap_c(
    double[::1] prices,
    double[::1] sizes,
    int n_levels,
    double target,
):
    """VWAP for target shares from pre-sorted price/size arrays."""
    cdef double remaining = target
    cdef double total = 0.0
    cdef double filled = 0.0
    cdef double take
    cdef int j

    for j in range(n_levels):
        take = sizes[j] if sizes[j] < remaining else remaining
        if take <= 0.0:
            continue
        total  += take * prices[j]
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break

    return total / target if filled >= target - 1e-9 else -1.0


cpdef void n_leg_vwap_score(
    double[:, ::1] leg_prices,
    double[:, ::1] leg_sizes,
    int[::1] n_levels,
    double[::1] candidates,
    int n_candidates,
    int n_legs,
    double guaranteed_payout,
    double max_capital,
    double min_trade_size,
    double[::1] result_out,
):
    """Find best N-leg trade size. Writes (size, capital, profit, roi) into result_out[4]."""
    cdef int i, k
    cdef double size, unit_cost, vwap, cost, profit, roi
    cdef double best_roi = -1e18

    result_out[0] = -1.0
    result_out[1] = 0.0
    result_out[2] = 0.0
    result_out[3] = 0.0

    for i in range(n_candidates):
        size = candidates[i]
        if size <= 0.0:
            continue

        unit_cost = 0.0
        for k in range(n_legs):
            vwap = _vwap_c(leg_prices[k], leg_sizes[k], n_levels[k], size)
            if vwap < 0.0:
                unit_cost = -1.0
                break
            unit_cost += vwap

        if unit_cost < 0.0:
            continue

        cost   = size * unit_cost
        profit = size * (guaranteed_payout - unit_cost)
        if cost > max_capital + 1e-9 or cost < min_trade_size or profit <= 0.0:
            continue

        roi = profit / (cost if cost > 1e-9 else 1e-9)
        if roi > best_roi:
            best_roi = roi
            result_out[0] = size
            result_out[1] = cost
            result_out[2] = profit
            result_out[3] = roi
