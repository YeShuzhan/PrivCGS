"""Exact GFL-MPE E-step with reusable released row-kernel sufficient state."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gfl_mpe_exact_parallel_estep_v1 as base


SOURCE = Path(__file__).with_suffix(".cpp")
CANDIDATE_NAME = "exact-parallel-row-kernel-sufficient-state-v1"
THREAD_COUNT = 8
_LOADED = None
_PLAN = None


def _load_library():
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    from route_s_clean._native_build import build_shared_library

    target = build_shared_library(
        "gcfr-gfl-mpe-rkss1-v1",
        SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++17", "-shared",
            "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(target))
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    precompute = library.gcfr_gfl_mpe_rkss1_precompute_rows_v1
    precompute.argtypes = [ctypes.c_int64, f64, i64, ctypes.c_double, f64, f64, f64]
    precompute.restype = ctypes.c_int
    estep = library.gcfr_gfl_mpe_exact_parallel_estep_rkss1_v1
    estep.argtypes = [
        ctypes.c_int64, f64, i64, i64, ctypes.c_int64, i64,
        ctypes.c_int, ctypes.c_int, f64, ctypes.c_double, ctypes.c_int,
        f64, f64, f64, f64, f64,
    ]
    estep.restype = ctypes.c_int
    _LOADED = library
    return library


@dataclass(frozen=True)
class _RKSSPlan:
    grouped: object
    sigma_hex: str
    row_log_x: np.ndarray
    row_x: np.ndarray
    row_half_square: np.ndarray


def _pointer(value: np.ndarray) -> int:
    return int(value.__array_interface__["data"][0])


def _plan_for(observation: np.ndarray, center: np.ndarray, sigma: float) -> _RKSSPlan:
    global _PLAN
    sigma_hex = float(sigma).hex()
    if (
        _PLAN is None
        or _PLAN.grouped.row_count != int(observation.size)
        or _PLAN.grouped.source_observation_pointer != _pointer(observation)
        or _PLAN.grouped.source_center_pointer != _pointer(center)
        or _PLAN.sigma_hex != sigma_hex
    ):
        grouped = base._make_plan(observation, center)
        row_log_x = np.empty(grouped.row_count, dtype=np.float64)
        row_x = np.empty(grouped.row_count, dtype=np.float64)
        row_half_square = np.empty(grouped.row_count, dtype=np.float64)
        status = int(
            _load_library().gcfr_gfl_mpe_rkss1_precompute_rows_v1(
                grouped.row_count,
                grouped.observation_sorted,
                grouped.center_sorted,
                float(sigma),
                row_log_x,
                row_x,
                row_half_square,
            )
        )
        if status:
            raise RuntimeError(f"RKSS1 row-kernel compilation returned {status}")
        _PLAN = _RKSSPlan(
            grouped=grouped,
            sigma_hex=sigma_hex,
            row_log_x=row_log_x,
            row_x=row_x,
            row_half_square=row_half_square,
        )
    return _PLAN


def reset_plan() -> None:
    global _PLAN
    _PLAN = None


def plan_diagnostics() -> dict[str, int | str]:
    if _PLAN is None:
        return {"backend": "grouped_full_log_multipoint_em_v1", "group_count": 0, "maximum_group_size": 0, "row_count": 0}
    grouped = _PLAN.grouped
    return {
        "backend": "grouped_full_log_multipoint_em_v1",
        "group_count": int(grouped.group_count),
        "maximum_group_size": int(grouped.maximum_group_size),
        "row_count": int(grouped.row_count),
    }


def expected_log_counts(observation, center, half_bandwidth, log_probability, sigma):
    observed = np.ascontiguousarray(observation, dtype=np.float64).reshape(-1)
    centers = np.ascontiguousarray(center, dtype=np.int64).reshape(-1)
    log_state = np.ascontiguousarray(log_probability, dtype=np.float64).reshape(-1)
    if observed.shape != centers.shape:
        raise ValueError("observation/center dimension mismatch")
    plan = _plan_for(observed, centers, float(sigma))
    grouped = plan.grouped
    log_expected = np.empty(log_state.size, dtype=np.float64)
    row_log_density = np.empty(observed.size, dtype=np.float64)
    status = int(
        _load_library().gcfr_gfl_mpe_exact_parallel_estep_rkss1_v1(
            int(observed.size),
            grouped.observation_sorted,
            grouped.center_sorted,
            grouped.original_index_sorted,
            int(grouped.group_count),
            grouped.group_starts,
            int(half_bandwidth),
            int(log_state.size),
            log_state,
            float(sigma),
            int(THREAD_COUNT),
            plan.row_log_x,
            plan.row_x,
            plan.row_half_square,
            log_expected,
            row_log_density,
        )
    )
    if status:
        raise RuntimeError(f"RKSS1 exact parallel E-step returned {status}")
    return log_expected, row_log_density


__all__ = ("expected_log_counts", "plan_diagnostics", "reset_plan")
