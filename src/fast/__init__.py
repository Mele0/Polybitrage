"""Fast C/Cython/numba/numpy scan and optimizer kernels.

Dispatch chain (fastest available at import time):
  Cython (_scan_ext) → numba → numpy

Build the Cython extension for maximum performance:
    python setup_cython.py build_ext --inplace
"""
from __future__ import annotations

# ── Scan classifier ───────────────────────────────────────────────────────────
try:
    from src.fast._scan_ext import (  # type: ignore[import]
        scan_classify_batch,
        n_leg_vwap_score,
        REJECTED,
        NEAR_ARB,
        EXECUTABLE,
    )
    SCAN_BACKEND = "cython"
except ImportError:
    from src.fast.scan_numpy import (  # type: ignore[no-redef]
        scan_classify_batch,
        n_leg_vwap_score,
        SCAN_BACKEND,
        REJECTED,
        NEAR_ARB,
        EXECUTABLE,
    )

# ── Depth optimizer ───────────────────────────────────────────────────────────
from src.fast.depth_optimizer import (
    find_best_size_fast,
    BACKEND as OPTIMIZER_BACKEND,
)

__all__ = [
    "scan_classify_batch",
    "n_leg_vwap_score",
    "find_best_size_fast",
    "SCAN_BACKEND",
    "OPTIMIZER_BACKEND",
    "REJECTED",
    "NEAR_ARB",
    "EXECUTABLE",
]
