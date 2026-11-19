"""Native source-order kernel owned and built by the clean project."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source

SOURCE = native_source("source_order", "source_order.cpp")
SOURCE_DIRECTORY = SOURCE.parent
DEPENDENCIES = (
    SOURCE,
    SOURCE_DIRECTORY / "operator_cell_bound.cpp",
    *sorted(SOURCE_DIRECTORY.glob("*.hpp")), *sorted(SOURCE_DIRECTORY.glob("*.inc")),
)
_LOADED = None


def _build():
    return build_shared_library(
        "source-order",
        SOURCE,
        dependencies=DEPENDENCIES[1:],
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++17", "-shared",
            "-static-libgcc", "-static-libstdc++",
        ),
    )


def load_source_order_library():
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    library = ctypes.CDLL(str(_build()))
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")

    generate = library.route_s_generate_edges
    generate.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_double,
        f64,
        f64,
        f64,
        f64,
        f64,
        ctypes.c_double,
        i32,
        i32,
        f64,
        f64,
        f64,
        ctypes.c_int,
        i32,
        i32,
        i32,
        i32,
        f64,
        f64,
        f64,
        f64,
        u64,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        i64,
        f64,
        i64,
        f64,
    ]
    generate.restype = ctypes.c_int

    eta = library.route_s_edge_eta_batch
    eta.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        f64,
        f64,
        f64,
        f64,
        ctypes.c_double,
        i32,
        i32,
        f64,
        f64,
        f64,
        ctypes.c_int64,
        i64,
        i64,
        f64,
    ]
    eta.restype = ctypes.c_int

    delta = library.route_s_flag_face_delta_batch
    delta.argtypes = [
        ctypes.c_int,
        i64,
        i64,
        ctypes.c_int64,
        i64,
        i64,
        i64,
        i64,
    ]
    delta.restype = ctypes.c_int

    integer_diagnostics = (
        "route_s_last_source_candidate_prefix_limit",
        "route_s_last_source_maximum_materialized_segment_size",
        "route_s_last_source_maximum_dfs_stack_size",
        "route_s_last_source_topk_round_count",
        "route_s_last_source_used_frontier_free_topk",
        "route_s_last_source_used_monolithic_fallback",
        "route_s_last_source_binary64_serial_tie_comparison_count",
        "operator_bound_last_joint_bound_call_count",
        "operator_bound_last_tightened_cell_count",
        "operator_bound_last_component_restricted_count",
        "operator_bound_last_joint_only_prune_count",
        "operator_bound_last_operator_only_prune_count",
        "operator_bound_last_operator_parent_prune_count",
    )
    for name in integer_diagnostics:
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_longlong
    for name in (
        "operator_bound_last_cumulative_upper_reduction",
        "operator_bound_last_maximum_upper_reduction",
    ):
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_double

    _LOADED = library
    return library


__all__ = ("load_source_order_library",)
