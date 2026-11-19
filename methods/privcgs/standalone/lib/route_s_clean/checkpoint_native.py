"""Bounded-memory native validation for private undirected CSR input."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source


SOURCE = native_source("checkpoint", "csr_validate.cpp")
_FUNCTION = None


def _load_function():
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    path = build_shared_library(
        "csr-validator",
        SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++20", "-shared",
            "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(path))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_validate_undirected_csr
    function.argtypes = [ctypes.c_int64, ctypes.c_int64, i64, i64]
    function.restype = ctypes.c_int
    _FUNCTION = function
    return function


def validate_simple_undirected_csr(offsets, neighbors):
    row_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    column_indices = np.ascontiguousarray(neighbors, dtype=np.int64)
    if row_offsets.ndim != 1 or column_indices.ndim != 1 or row_offsets.size < 2:
        raise ValueError("invalid undirected CSR graph")
    n_nodes = int(row_offsets.size - 1)
    status = int(
        _load_function()(
            n_nodes,
            int(column_indices.size),
            row_offsets,
            column_indices,
        )
    )
    if status:
        reasons = {
            1: "node/edge count",
            2: "terminal offset",
            3: "offset monotonicity",
            4: "neighbor range",
            5: "self loop",
            6: "row ordering or duplicate",
            7: "undirected symmetry",
        }
        raise ValueError(
            "invalid undirected CSR graph: " + reasons.get(status, f"status {status}")
        )
    return row_offsets, column_indices


__all__ = ("validate_simple_undirected_csr",)
