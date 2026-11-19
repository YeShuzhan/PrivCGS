"""Native public counter kernel for the quenched Palm-token measure."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source

SOURCE = native_source("degree_chain", "palm_tokens.cpp")
_FUNCTION = None


def _load_function():
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    library_path = build_shared_library(
        "palm-tokens",
        SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++20", "-shared",
            "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    function = library.route_s_counter_palm_tokens
    function.argtypes = [
        ctypes.c_int64, ctypes.c_int64, i64, i64, i64, ctypes.c_uint64,
        u64, i64, i64,
    ]
    function.restype = ctypes.c_int
    _FUNCTION = function
    return function


def counter_palm_tokens(edges, offsets, neighbors, n_nodes: int, public_seed: int):
    edge_array = np.ascontiguousarray(edges, dtype=np.int64).reshape((-1, 2))
    csr_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    csr_neighbors = np.ascontiguousarray(neighbors, dtype=np.int64)
    output = np.empty(edge_array.shape[0], dtype=np.uint64)
    output_count = np.zeros(1, dtype=np.int64)
    skipped_count = np.zeros(1, dtype=np.int64)
    status = int(_load_function()(
        int(n_nodes), int(edge_array.shape[0]), edge_array.reshape(-1),
        csr_offsets, csr_neighbors, np.uint64(int(public_seed)), output,
        output_count, skipped_count,
    ))
    if status:
        raise RuntimeError(f"clean Palm-token kernel returned {status}")
    return np.ascontiguousarray(output[: int(output_count[0])]), int(skipped_count[0])


__all__ = ("counter_palm_tokens",)
