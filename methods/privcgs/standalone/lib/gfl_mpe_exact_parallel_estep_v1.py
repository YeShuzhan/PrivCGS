"""Exact parallel centre-fibre E-step backend for aligned GFL-MPE."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np


SOURCE = Path(__file__).with_suffix(".cpp")
_LOADED = None
_PLAN = None
THREAD_COUNT = 8


def _load_library():
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    from route_s_clean._native_build import build_shared_library

    target = build_shared_library(
        "gcfr-gfl-mpe-exact-parallel-estep-v1",
        SOURCE,
        compiler_flags=(
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-shared",
            "-static-libgcc",
            "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(target))
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    function = library.gcfr_gfl_mpe_exact_parallel_estep_v1
    function.argtypes = [
        ctypes.c_int64,
        f64,
        i64,
        i64,
        ctypes.c_int64,
        i64,
        ctypes.c_int,
        ctypes.c_int,
        f64,
        ctypes.c_double,
        ctypes.c_int,
        f64,
        f64,
    ]
    function.restype = ctypes.c_int
    _LOADED = library
    return library


@dataclass(frozen=True)
class _GroupedPlan:
    source_observation_pointer: int
    source_center_pointer: int
    row_count: int
    observation_sorted: np.ndarray
    center_sorted: np.ndarray
    original_index_sorted: np.ndarray
    group_starts: np.ndarray
    group_count: int
    maximum_group_size: int


def _pointer(value: np.ndarray) -> int:
    return int(value.__array_interface__["data"][0])


def _make_plan(observation: np.ndarray, center: np.ndarray) -> _GroupedPlan:
    order = np.argsort(center, kind="stable")
    observed_sorted = np.ascontiguousarray(observation[order], dtype=np.float64)
    center_sorted = np.ascontiguousarray(center[order], dtype=np.int64)
    original = np.ascontiguousarray(order, dtype=np.int64)
    if center_sorted.size:
        starts = np.r_[0, 1 + np.flatnonzero(center_sorted[1:] != center_sorted[:-1])]
        ends = np.r_[starts[1:], center_sorted.size]
        group_starts = np.ascontiguousarray(
            np.r_[starts, center_sorted.size], dtype=np.int64
        )
        maximum_group_size = int(np.max(ends - starts))
        group_count = int(starts.size)
    else:
        group_starts = np.zeros(1, dtype=np.int64)
        maximum_group_size = 0
        group_count = 0
    return _GroupedPlan(
        source_observation_pointer=_pointer(observation),
        source_center_pointer=_pointer(center),
        row_count=int(observation.size),
        observation_sorted=observed_sorted,
        center_sorted=center_sorted,
        original_index_sorted=original,
        group_starts=group_starts,
        group_count=group_count,
        maximum_group_size=maximum_group_size,
    )


def _plan_for(observation: np.ndarray, center: np.ndarray) -> _GroupedPlan:
    global _PLAN
    if (
        _PLAN is None
        or _PLAN.row_count != int(observation.size)
        or _PLAN.source_observation_pointer != _pointer(observation)
        or _PLAN.source_center_pointer != _pointer(center)
    ):
        _PLAN = _make_plan(observation, center)
    return _PLAN


def reset_plan() -> None:
    global _PLAN
    _PLAN = None


def plan_diagnostics() -> dict[str, int | str]:
    if _PLAN is None:
        return {
            "backend": "grouped_full_log_multipoint_em_v1",
            "group_count": 0,
            "maximum_group_size": 0,
            "row_count": 0,
        }
    return {
        "backend": "grouped_full_log_multipoint_em_v1",
        "group_count": int(_PLAN.group_count),
        "maximum_group_size": int(_PLAN.maximum_group_size),
        "row_count": int(_PLAN.row_count),
    }


def expected_log_counts(observation, center, half_bandwidth, log_probability, sigma):
    observed = np.ascontiguousarray(observation, dtype=np.float64).reshape(-1)
    centers = np.ascontiguousarray(center, dtype=np.int64).reshape(-1)
    log_state = np.ascontiguousarray(log_probability, dtype=np.float64).reshape(-1)
    if observed.shape != centers.shape:
        raise ValueError("observation/center dimension mismatch")
    plan = _plan_for(observed, centers)
    log_expected = np.empty(log_state.size, dtype=np.float64)
    row_log_density = np.empty(observed.size, dtype=np.float64)
    status = int(
        _load_library().gcfr_gfl_mpe_exact_parallel_estep_v1(
            int(observed.size),
            plan.observation_sorted,
            plan.center_sorted,
            plan.original_index_sorted,
            int(plan.group_count),
            plan.group_starts,
            int(half_bandwidth),
            int(log_state.size),
            log_state,
            float(sigma),
            int(THREAD_COUNT),
            log_expected,
            row_log_density,
        )
    )
    if status:
        raise RuntimeError(f"exact parallel GFL-MPE E-step returned {status}")
    return log_expected, row_log_density


__all__ = ("expected_log_counts", "plan_diagnostics", "reset_plan")
