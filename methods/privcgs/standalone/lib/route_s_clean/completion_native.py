"""Compact native admission for the globally disjoint completion chart."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source


SOURCE = native_source("degree_chain", "completion.cpp")
_FUNCTION = None


def _load_function():
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    path = build_shared_library(
        "completion-admission",
        SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++20", "-shared",
            "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(path))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    i8 = np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_completion_admit
    function.argtypes = [
        ctypes.c_int64, ctypes.c_int64,
        i64, ctypes.c_int64, i64, i64, ctypes.c_int64, i64,
        i64, i64, i64, i64, i64, i64,
        i8, i64, i8, i8, i64, i64, i64,
    ]
    function.restype = ctypes.c_int
    _FUNCTION = function
    return function


def admit_completion_units(
    n_nodes: int,
    core,
    descent,
    positive_direction,
    required_count,
):
    direction = np.ascontiguousarray(positive_direction, dtype=np.int8)
    required = np.ascontiguousarray(required_count, dtype=np.int64)
    negative_mask = np.zeros(descent.negative_energy.size, dtype=np.int8)
    positive_mask = np.zeros(descent.positive_energy.size, dtype=np.int8)
    selected = np.zeros(1, dtype=np.int64)
    reserved = np.zeros(1, dtype=np.int64)
    capacity = np.zeros(1, dtype=np.int64)

    def edge_values(array):
        return np.ascontiguousarray(array, dtype=np.int64).reshape(-1)

    status = int(_load_function()(
        int(n_nodes), int(direction.size),
        edge_values(core.negative_old_edge), int(core.negative_energy.size),
        edge_values(core.negative_new_edge),
        edge_values(core.positive_old_edge), int(core.positive_energy.size),
        edge_values(core.positive_new_edge),
        np.ascontiguousarray(descent.negative_offsets, dtype=np.int64),
        edge_values(descent.negative_old_edge),
        edge_values(descent.negative_new_edge),
        np.ascontiguousarray(descent.positive_offsets, dtype=np.int64),
        edge_values(descent.positive_old_edge),
        edge_values(descent.positive_new_edge),
        direction, required, negative_mask, positive_mask,
        selected, reserved, capacity,
    ))
    if status:
        raise RuntimeError(f"clean completion-admission kernel returned {status}")
    return (
        negative_mask.astype(bool, copy=False),
        positive_mask.astype(bool, copy=False),
        {
            "selected_unit_count": int(selected[0]),
            "reserved_dyad_count": int(reserved[0]),
            "compact_table_capacity": int(capacity[0]),
        },
    )


__all__ = ("admit_completion_units",)
