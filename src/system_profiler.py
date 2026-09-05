"""OS / process resource sampler for the arb bot (background, low overhead).

A trading loop's tail latency is often *not* in the code — it's the OS preempting
the process (involuntary context switches), GC, page faults, or CPU contention.
This module samples those once per interval on a background task and caches the
result so the Prometheus ``/metrics`` scrape can read it with zero work.

Sources (in preference order):
  * ``resource.getrusage`` (stdlib, always available): voluntary / **involuntary**
    context switches, minor/major page faults, user/sys CPU time, peak RSS.  The
    involuntary count is the key jitter signal — it counts how often the kernel
    yanked the CPU away mid-work.
  * ``gc`` (stdlib): per-generation collection counts + live thresholds.
  * ``psutil`` (optional): current RSS, thread/fd counts, instantaneous process &
    system CPU%, system memory%, load average.  Degrades gracefully if absent.

All rate fields (``*_per_s``, ``cpu_percent``) are derived by diffing consecutive
samples, so they need two ticks before they read non-zero.
"""
from __future__ import annotations

import gc
import sys
import threading
import time
from typing import Any

try:  # `resource` is Unix-only; degrade gracefully on Windows.
    import resource  # type: ignore
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None  # type: ignore

try:  # optional richer metrics
    import psutil  # type: ignore

    _PROC: Any = psutil.Process()
    _PROC.cpu_percent(None)   # prime the per-call delta
    psutil.cpu_percent(None)  # prime system-wide delta
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore
    _PROC = None

_PAGE = resource.getpagesize() if resource is not None else 4096
_IS_DARWIN = sys.platform == "darwin"

_LATEST: dict[str, float] = {}
_STATE: dict[str, float] = {}


def _read(prev: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    now = time.monotonic()
    ru = resource.getrusage(resource.RUSAGE_SELF) if resource is not None else None
    cpu_total = (ru.ru_utime + ru.ru_stime) if ru is not None else 0.0
    flt_total = float((ru.ru_minflt + ru.ru_majflt)) if ru is not None else 0.0

    out: dict[str, float] = {"threads_python": float(threading.active_count())}
    if ru is not None:
        out.update({
            "cpu_user_seconds": round(ru.ru_utime, 4),
            "cpu_sys_seconds": round(ru.ru_stime, 4),
            "ctx_switch_voluntary_total": float(ru.ru_nvcsw),
            "ctx_switch_involuntary_total": float(ru.ru_nivcsw),
            "page_faults_minor_total": float(ru.ru_minflt),
            "page_faults_major_total": float(ru.ru_majflt),
            # ru_maxrss is bytes on macOS, KiB on Linux — normalize to bytes.
            "maxrss_bytes": float(ru.ru_maxrss if _IS_DARWIN else ru.ru_maxrss * 1024),
        })

    # ── rates (need a previous sample) ──
    if prev and ru is not None:
        dt = max(1e-6, now - prev.get("_t", now))
        out["cpu_percent"] = round(100.0 * (cpu_total - prev.get("_cpu", cpu_total)) / dt, 2)
        out["ctx_switch_voluntary_per_s"] = round((ru.ru_nvcsw - prev.get("_nvcsw", ru.ru_nvcsw)) / dt, 1)
        out["ctx_switch_involuntary_per_s"] = round((ru.ru_nivcsw - prev.get("_nivcsw", ru.ru_nivcsw)) / dt, 1)
        out["page_faults_per_s"] = round((flt_total - prev.get("_flt", flt_total)) / dt, 1)

    # ── GC ──
    counts = gc.get_count()
    for i, c in enumerate(counts):
        out[f"gc_gen{i}_pending"] = float(c)
    for i, st in enumerate(gc.get_stats()):
        out[f"gc_gen{i}_collections_total"] = float(st.get("collections", 0))
        out[f"gc_gen{i}_collected_total"] = float(st.get("collected", 0))

    # ── psutil extras ──
    if _PROC is not None:
        try:
            with _PROC.oneshot():
                out["rss_bytes"] = float(_PROC.memory_info().rss)
                out["threads_os"] = float(_PROC.num_threads())
                out["proc_cpu_percent"] = round(float(_PROC.cpu_percent(None)), 2)
                try:
                    out["open_fds"] = float(_PROC.num_fds())
                except Exception:  # noqa: BLE001 — not on all platforms
                    pass
            out["sys_cpu_percent"] = round(float(psutil.cpu_percent(None)), 2)  # type: ignore[union-attr]
            out["sys_mem_percent"] = round(float(psutil.virtual_memory().percent), 2)  # type: ignore[union-attr]
            try:
                l1, l5, l15 = psutil.getloadavg()  # type: ignore[union-attr]
                out["load_avg_1m"], out["load_avg_5m"], out["load_avg_15m"] = (
                    round(l1, 2), round(l5, 2), round(l15, 2))
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 — never let sampling crash the loop
            pass

    state = {"_t": now, "_cpu": cpu_total, "_flt": flt_total}
    if ru is not None:
        state["_nvcsw"] = float(ru.ru_nvcsw)
        state["_nivcsw"] = float(ru.ru_nivcsw)
    return out, state


async def sampler(interval_s: float = 1.0) -> None:
    """Background task: refresh the cached OS snapshot every ``interval_s``.

    Started by the simulator alongside the other profiling tasks; cancelled on
    shutdown.  getrusage + a psutil oneshot is a handful of fast syscalls — the
    cost is negligible at 1 Hz and runs off the event-loop-critical scan path.
    """
    import asyncio
    global _LATEST, _STATE
    _LATEST, _STATE = _read(_STATE)  # prime (no rates yet)
    while True:
        await asyncio.sleep(max(0.05, interval_s))
        try:
            _LATEST, _STATE = _read(_STATE)
        except Exception:  # noqa: BLE001
            pass


def snapshot() -> dict[str, float]:
    """Latest OS/process metrics (cheap dict read for the /metrics scrape)."""
    return _LATEST


def has_psutil() -> bool:
    return _PROC is not None
