"""numpy/numba fallback for the Cython scan classifier.

Provides the same interface as the compiled ``src.fast._scan_ext`` module so
the simulator can use it transparently when the Cython extension has not been
compiled yet.

Backend dispatch:
  1. numba @njit  — C-speed compiled, ~1–2 µs for 75 pairs (``pip install numba``)
  2. numpy         — vectorised, ~5–10 µs for 75 pairs  (always available)
"""
from __future__ import annotations

import numpy as np

REJECTED   = 0
NEAR_ARB   = 1
EXECUTABLE = 2

# ── numba path ────────────────────────────────────────────────────────────────

try:
    import numba

    @numba.njit(cache=True, nogil=True)
    def _classify_jit(
        parent_asks, child_asks,
        fee_rates_1, fee_rates_2,
        slippages, fee_buffers, min_edges,
        near_threshold, entry_threshold,
        n,
        gross_out, net_out, class_out,
    ):
        for i in range(n):
            pa = parent_asks[i]
            ca = child_asks[i]
            gross = pa + ca
            gross_out[i] = gross
            fee = (fee_rates_1[i] * pa * (1.0 - pa)
                   + fee_rates_2[i] * ca * (1.0 - ca))
            net = gross + fee + slippages[i] + fee_buffers[i]
            net_out[i] = net
            if net < entry_threshold and (1.0 - net) >= min_edges[i]:
                class_out[i] = EXECUTABLE
            elif net <= near_threshold:
                class_out[i] = NEAR_ARB
            else:
                class_out[i] = REJECTED

    @numba.njit(cache=True, nogil=True)
    def _vwap_jit(prices, sizes, n_levels, target):
        remaining = target
        total = 0.0
        filled = 0.0
        for j in range(n_levels):
            take = sizes[j] if sizes[j] < remaining else remaining
            if take <= 0.0:
                continue
            total += take * prices[j]
            filled += take
            remaining -= take
            if remaining <= 1e-12:
                break
        return total / target if filled >= target - 1e-9 else -1.0

    @numba.njit(cache=True, nogil=True)
    def _n_leg_score_jit(
        leg_prices, leg_sizes, n_levels, candidates, n_candidates, n_legs,
        guaranteed_payout, max_capital, min_trade_size, result_out,
    ):
        best_roi = -1e18
        result_out[0] = -1.0
        for i in range(n_candidates):
            size = candidates[i]
            if size <= 0.0:
                continue
            unit_cost = 0.0
            ok = True
            for k in range(n_legs):
                vwap = _vwap_jit(leg_prices[k], leg_sizes[k], n_levels[k], size)
                if vwap < 0.0:
                    ok = False
                    break
                unit_cost += vwap
            if not ok:
                continue
            cost = size * unit_cost
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

    def scan_classify_batch(
        parent_asks, child_asks,
        fee_rates_1, fee_rates_2,
        slippages, fee_buffers, min_edges,
        near_threshold, entry_threshold,
        n,
        gross_out, net_out, class_out,
    ):
        _classify_jit(
            parent_asks, child_asks,
            fee_rates_1, fee_rates_2,
            slippages, fee_buffers, min_edges,
            near_threshold, entry_threshold,
            n, gross_out, net_out, class_out,
        )

    def n_leg_vwap_score(
        leg_prices, leg_sizes, n_levels, candidates, n_candidates, n_legs,
        guaranteed_payout, max_capital, min_trade_size, result_out,
    ):
        _n_leg_score_jit(
            leg_prices, leg_sizes, n_levels, candidates, n_candidates, n_legs,
            guaranteed_payout, max_capital, min_trade_size, result_out,
        )

    SCAN_BACKEND = "numba"

except ImportError:

    # ── numpy vectorised fallback ─────────────────────────────────────────────

    def scan_classify_batch(
        parent_asks, child_asks,
        fee_rates_1, fee_rates_2,
        slippages, fee_buffers, min_edges,
        near_threshold, entry_threshold,
        n,
        gross_out, net_out, class_out,
    ):
        gross = parent_asks[:n] + child_asks[:n]
        fee = (fee_rates_1[:n] * parent_asks[:n] * (1.0 - parent_asks[:n])
               + fee_rates_2[:n] * child_asks[:n] * (1.0 - child_asks[:n]))
        net = gross + fee + slippages[:n] + fee_buffers[:n]
        gross_out[:n] = gross
        net_out[:n] = net
        class_out[:n] = np.where(
            (net < entry_threshold) & ((1.0 - net) >= min_edges[:n]),
            EXECUTABLE,
            np.where(net <= near_threshold, NEAR_ARB, REJECTED),
        ).astype(np.int32)

    def n_leg_vwap_score(
        leg_prices, leg_sizes, n_levels, candidates, n_candidates, n_legs,
        guaranteed_payout, max_capital, min_trade_size, result_out,
    ):
        result_out[0] = -1.0
        result_out[1] = 0.0
        result_out[2] = 0.0
        result_out[3] = 0.0
        best_roi = -1e18
        for i in range(n_candidates):
            size = float(candidates[i])
            if size <= 0:
                continue
            unit_cost = 0.0
            ok = True
            for k in range(n_legs):
                nl = int(n_levels[k])
                remaining = size
                total = 0.0
                filled = 0.0
                for j in range(nl):
                    take = min(float(leg_sizes[k, j]), remaining)
                    if take <= 0:
                        continue
                    total += take * float(leg_prices[k, j])
                    filled += take
                    remaining -= take
                    if remaining <= 1e-12:
                        break
                if filled < size - 1e-9:
                    ok = False
                    break
                unit_cost += total / size
            if not ok:
                continue
            cost = size * unit_cost
            profit = size * (guaranteed_payout - unit_cost)
            if cost > max_capital + 1e-9 or cost < min_trade_size or profit <= 0:
                continue
            roi = profit / max(cost, 1e-9)
            if roi > best_roi:
                best_roi = roi
                result_out[0] = size
                result_out[1] = cost
                result_out[2] = profit
                result_out[3] = roi

    SCAN_BACKEND = "numpy"
