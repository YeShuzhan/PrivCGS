"""Loader for native kernels owned and built inside GCFR-Scale."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source

SOURCE = native_source("ragged_prefix.cpp")
_FUNCTION = None

def _load_function():
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    library_path = build_shared_library(
        "ragged-prefix",
        SOURCE,
        compiler_flags=(
            "-O3", "-std=c++20", "-shared", "-static-libgcc",
            "-static-libstdc++", "-ffp-contract=off", "-fno-tree-vectorize",
            "-fno-tree-slp-vectorize", "-frounding-math",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_ragged_prefix
    function.argtypes = [ctypes.c_int64, i64, i64, i64, f64, f64, f64]
    function.restype = ctypes.c_int32
    _FUNCTION = function
    return function


def ragged_prefix(
    state_offsets: np.ndarray,
    negative_offsets: np.ndarray,
    positive_offsets: np.ndarray,
    negative: np.ndarray,
    positive: np.ndarray,
) -> np.ndarray:
    state_offsets = np.ascontiguousarray(state_offsets, dtype=np.int64)
    negative_offsets = np.ascontiguousarray(negative_offsets, dtype=np.int64)
    positive_offsets = np.ascontiguousarray(positive_offsets, dtype=np.int64)
    negative = np.ascontiguousarray(negative, dtype=np.float64)
    positive = np.ascontiguousarray(positive, dtype=np.float64)
    if not (
        state_offsets.size == negative_offsets.size == positive_offsets.size
        and negative.size == int(negative_offsets[-1])
        and positive.size == int(positive_offsets[-1])
    ):
        raise ValueError("ragged-prefix arrays are not aligned")
    output = np.empty(int(state_offsets[-1]), dtype=np.float64)
    status = int(
        _load_function()(
            int(state_offsets.size - 1),
            state_offsets,
            negative_offsets,
            positive_offsets,
            negative,
            positive,
            output,
        )
    )
    if status:
        raise ArithmeticError(f"clean ragged-prefix kernel returned {status}")
    return output
