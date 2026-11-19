"""Native sparse candidate enumeration and global-disjoint admission."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source

SOURCE = native_source("degree_chain", "candidate_chart.cpp")
SCARCITY_SOURCE = native_source("degree_chain", "scarcity_stratified_descent.cpp")
_LIBRARY = None
_SCARCITY_LIBRARY = None


def load_candidate_chart_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    library_path = build_shared_library(
        "candidate-chart",
        SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++20", "-shared",
            "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    i8 = np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS")
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    library.route_s_count_relocation_candidates.argtypes = [
        ctypes.c_int64, i64, i64, i64, ctypes.c_int64, u64, i64
    ]
    library.route_s_count_relocation_candidates.restype = ctypes.c_int
    library.route_s_fill_relocation_candidates.argtypes = [
        ctypes.c_int64, i64, i64, i64, ctypes.c_int64, u64, ctypes.c_int64,
        i64, i8, i64, i64, i64, i64,
    ]
    library.route_s_fill_relocation_candidates.restype = ctypes.c_int
    library.route_s_sort_and_admit_relocation_ids.argtypes = [
        ctypes.c_int64, f64, ctypes.c_int64, i64, i64, i64, i64
    ]
    library.route_s_sort_and_admit_relocation_ids.restype = ctypes.c_int
    library.route_s_count_descent_candidates.argtypes = [
        ctypes.c_int64, i64, i8, i64, i64, i64, ctypes.c_int64, u64, i64, i64
    ]
    library.route_s_count_descent_candidates.restype = ctypes.c_int
    library.route_s_fill_descent_candidates.argtypes = [
        ctypes.c_int64, i64, i8, i64, i64, i64, ctypes.c_int64, u64,
        ctypes.c_int64, i64, i64, i64, i64, i64,
    ]
    library.route_s_fill_descent_candidates.restype = ctypes.c_int
    library.route_s_admit_descent_candidates.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, i64, i64, i64,
        i64, i64, i64, i64, i64,
    ]
    library.route_s_admit_descent_candidates.restype = ctypes.c_int
    _LIBRARY = library
    return library


def load_scarcity_stratified_descent_library():
    global _SCARCITY_LIBRARY
    if _SCARCITY_LIBRARY is not None:
        return _SCARCITY_LIBRARY
    library_path = build_shared_library(
        "scarcity-stratified-descent",
        SCARCITY_SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++17", "-shared",
            "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    i8 = np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function = library.route_s_scarcity_stratified_descent
    function.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        f64, f64, f64, f64, ctypes.c_double, i32, i32, f64, f64, f64,
        i64, i8, i64, i64, i64, ctypes.c_int64, u64, ctypes.c_int64,
        i64, i64, i64, i64, i64, f64, i64, i64, i64,
    ]
    function.restype = ctypes.c_int32
    _SCARCITY_LIBRARY = library
    return library


__all__ = (
    "load_candidate_chart_library",
    "load_scarcity_stratified_descent_library",
)
