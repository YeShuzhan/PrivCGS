"""Strict-exact cross-chart gain reuse candidate for Candidate02 common compile."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import time
from typing import Mapping

import numpy as np


SOURCE = Path(__file__).with_name("degree_chain_gain_cache_v1.cpp").resolve()
_LIBRARY = None
_ACTIVE_CACHE = None
_FROZEN = None


@dataclass
class ExactGainCache:
    gain: np.ndarray
    positive_count: np.ndarray
    negative_count: np.ndarray


def install(frozen_pipeline) -> None:
    global _FROZEN
    _FROZEN = frozen_pipeline
    frozen_pipeline.build_unique_dyad_chart = build_unique_dyad_chart
    frozen_pipeline.build_descent_support_chart = build_descent_support_chart


def _chart_globals():
    if _FROZEN is None:
        raise RuntimeError("gain-cache candidate has not been installed")
    return _FROZEN.build_unique_dyad_chart.__globals__


def _production_chart_globals():
    if _FROZEN is None:
        raise RuntimeError("gain-cache candidate has not been installed")
    module = importlib.import_module("route_s_clean.candidate_chart")
    return module.__dict__


def load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    native_build = importlib.import_module("route_s_clean._native_build")
    library_path = native_build.build_shared_library(
        "degree-chain-gain-cache-v1",
        SOURCE,
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
    count = library.route_s_bundle_direction_counts
    count.argtypes = [ctypes.c_int64, ctypes.c_int64, i64, i8, i64, i64]
    count.restype = ctypes.c_int32
    descent = library.route_s_cached_scarcity_stratified_descent
    descent.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        f64, f64, f64, f64, ctypes.c_double, i32, i32, f64, f64, f64,
        i64, i8, i64, i64, i64, ctypes.c_int64, u64,
        ctypes.c_int64, i64, i64, f64, ctypes.c_int32, ctypes.c_int64,
        i64, i64, i64, i64, i64, f64, i64, i64, i64,
    ]
    descent.restype = ctypes.c_int32
    _LIBRARY = library
    return library


def build_unique_dyad_chart(law, source, path):
    """Execute the frozen unique chart and retain only its exact gain certificate."""

    global _ACTIVE_CACHE
    if _ACTIVE_CACHE is not None:
        raise RuntimeError("an unconsumed exact gain cache is already active")
    production = _production_chart_globals()
    started = time.perf_counter()
    order = np.ascontiguousarray(path, dtype=np.int64)
    arrays = production["_candidate_arrays"](source, order)
    path_index, direction, old_left, old_right, new_left, new_right = arrays
    count = int(path_index.size)
    # Same scalar edge_eta and subtraction, in bounded batches. No global
    # concatenated key array, unique-sort temporaries, or inverse map.
    gain = np.empty(count, dtype=np.float64)
    for start in range(0, count, 262144):
        stop = min(start + 262144, count)
        old_eta = production["edge_eta_batch"](law, old_left[start:stop], old_right[start:stop])
        new_eta = production["edge_eta_batch"](law, new_left[start:stop], new_right[start:stop])
        gain[start:stop] = new_eta - old_eta
    if count: del old_eta, new_eta

    bundle_count = max(int(order.size) - 1, 0)
    positive_count = np.empty(bundle_count, dtype=np.int64)
    negative_count = np.empty(bundle_count, dtype=np.int64)
    status = int(
        load_library().route_s_bundle_direction_counts(
            count,
            bundle_count,
            np.ascontiguousarray(path_index, dtype=np.int64),
            np.ascontiguousarray(direction, dtype=np.int8),
            positive_count,
            negative_count,
        )
    )
    if status:
        raise RuntimeError(f"bundle direction count returned {status}")
    if int(np.sum(positive_count, dtype=np.int64)) + int(
        np.sum(negative_count, dtype=np.int64)
    ) != count:
        raise RuntimeError("bundle direction counts do not cover the unique-chart candidates")

    from route_s_clean.compact_admission import select
    selected, reserved_bytes = select(source.n_nodes, gain, old_left, old_right,
                                      new_left, new_right, source.edge_keys.size)
    admitted_count = int(selected.size)
    selected_path = path_index[selected]
    selected_direction = direction[selected]
    selected_old = np.column_stack((old_left[selected], old_right[selected]))
    selected_new = np.column_stack((new_left[selected], new_right[selected]))
    selected_gain = gain[selected]
    del arrays, path_index, direction, old_left, old_right, new_left, new_right, selected
    tape = production["_pack_selected"](order, selected_path, selected_direction,
                                        selected_old, selected_new, selected_gain)
    _ACTIVE_CACHE = ExactGainCache(
        gain=gain,
        positive_count=positive_count,
        negative_count=negative_count,
    )
    return tape, {
        "raw_candidate_count": count,
        "unique_eta_dyad_count": None,
        "compact_exact_admission": True,
        "reserved_hash_bytes": reserved_bytes,
        "physical_eta_calls": 2 * count,
        "admitted_unit_count": admitted_count,
        "seconds": float(time.perf_counter() - started),
        "global_dyad_disjoint": True,
        "complete_dyad_enumeration": False,
        "complete_wedge_enumeration": False,
        "exact_gain_cache_candidate_count": count,
        "exact_gain_cache_bytes": int(
            gain.nbytes + positive_count.nbytes + negative_count.nbytes
        ),
    }


def build_descent_support_chart(law, source, target_endpoint_mass, path):
    """Consume the exact unique-chart gain certificate under frozen scarcity order."""

    global _ACTIVE_CACHE
    cache = _ACTIVE_CACHE
    if cache is None:
        raise RuntimeError("descent support requires the immediately preceding exact gain cache")
    production = _production_chart_globals()
    started = time.perf_counter()
    try:
        order = np.ascontiguousarray(path, dtype=np.int64)
        target = np.ascontiguousarray(target_endpoint_mass, dtype=np.float64)
        degree = source.degree_array.astype(np.float64)
        residual = target - degree
        desired_flow = -np.cumsum(residual[order])[:-1]
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            float(np.max(np.abs(desired_flow), initial=0.0)), 1.0
        )
        requested = np.ceil(np.abs(desired_flow)).astype(np.int64)
        requested[np.abs(desired_flow) <= tolerance] = 0
        direction = np.sign(desired_flow).astype(np.int8)
        direction[requested == 0] = 0
        requested = np.ascontiguousarray(requested)
        direction = np.ascontiguousarray(direction)
        capacity = min(
            int(np.sum(requested, dtype=np.int64)), int(source.edge_keys.size)
        )
        storage = max(capacity, 1)
        path_index = np.empty(storage, dtype=np.int64)
        old_left = np.empty(storage, dtype=np.int64)
        old_right = np.empty(storage, dtype=np.int64)
        new_left = np.empty(storage, dtype=np.int64)
        new_right = np.empty(storage, dtype=np.int64)
        gain = np.empty(storage, dtype=np.float64)
        admitted = np.zeros(max(source.n_nodes - 1, 0), dtype=np.int64)
        selected_count = np.zeros(1, dtype=np.int64)
        candidate_diagnostics = np.zeros(16, dtype=np.int64)
        audit_direct_gain = int(
            os.environ.get("ROUTE_S_GAIN_CACHE_AUDIT", "0").strip() not in ("", "0")
        )
        status = int(
            load_library().route_s_cached_scarcity_stratified_descent(
                law.n_nodes,
                law.rank,
                int(np.size(law.marked_shift)),
                int(np.size(law.component_shift)),
                np.ascontiguousarray(law.sociability, dtype=np.float64),
                np.ascontiguousarray(law.star, dtype=np.float64),
                np.ascontiguousarray(law.interaction, dtype=np.float64),
                np.ascontiguousarray(law.interaction_diagonal, dtype=np.float64),
                float(law.dust),
                np.ascontiguousarray(law.parent_fiber, dtype=np.int32),
                np.ascontiguousarray(law.component_fiber, dtype=np.int32),
                np.ascontiguousarray(law.marked_shift, dtype=np.float64),
                np.ascontiguousarray(law.component_shift, dtype=np.float64),
                np.ascontiguousarray(law.parent_flux, dtype=np.float64),
                order,
                direction,
                requested,
                source.csr_offsets,
                source.csr_neighbors,
                source.edge_keys.size,
                source.edge_keys,
                int(cache.gain.size),
                cache.positive_count,
                cache.negative_count,
                cache.gain,
                audit_direct_gain,
                capacity,
                path_index,
                old_left,
                old_right,
                new_left,
                new_right,
                gain,
                admitted,
                selected_count,
                candidate_diagnostics,
            )
        )
        if status:
            raise RuntimeError(f"cached scarcity-stratified descent returned {status}")
        selected = int(selected_count[0])
        path_index = np.ascontiguousarray(path_index[:selected])
        old_left = np.ascontiguousarray(old_left[:selected])
        old_right = np.ascontiguousarray(old_right[:selected])
        new_left = np.ascontiguousarray(new_left[:selected])
        new_right = np.ascontiguousarray(new_right[:selected])
        gain = np.ascontiguousarray(gain[:selected])
        tape = production["_pack_selected"](
            order,
            path_index,
            direction[path_index],
            np.column_stack((old_left, old_right)),
            np.column_stack((new_left, new_right)),
            gain,
        )
        real_shortfall = np.maximum(
            np.abs(desired_flow) - admitted.astype(np.float64), 0.0
        )
        return tape, {
            "raw_candidate_count": int(candidate_diagnostics[0]),
            "unique_eta_dyad_count": None,
            "requested_capacity_sum": int(np.sum(requested)),
            "admitted_capacity_sum": int(np.sum(admitted)),
            "real_capacity_shortfall_sum": float(np.sum(real_shortfall)),
            "exact_real_target_flow_inside_admitted_box": bool(
                np.all(real_shortfall <= tolerance)
            ),
            "seconds": float(time.perf_counter() - started),
            "scarcity_stratified_descent": True,
            "active_bundle_count": int(candidate_diagnostics[1]),
            "scarcity_stratum_count": int(candidate_diagnostics[2]),
            "maximum_stratum_candidate_count": int(candidate_diagnostics[3]),
            "admitted_unit_count": int(candidate_diagnostics[4]),
            "reserved_dyad_count": int(candidate_diagnostics[5]),
            "examined_candidate_count": int(candidate_diagnostics[6]),
            "provably_unread_tail_count": int(candidate_diagnostics[7]),
            "maximum_local_candidate_bytes": int(candidate_diagnostics[8]),
            "direct_eta_call_count": int(candidate_diagnostics[9]),
            "complete_dyad_enumeration": False,
            "complete_wedge_enumeration": False,
            "exact_gain_cache_candidate_count": int(candidate_diagnostics[10]),
            "exact_gain_cache_lookup_count": int(candidate_diagnostics[11]),
            "exact_gain_cache_bit_mismatch_count": int(candidate_diagnostics[12]),
            "exact_gain_cache_first_bit_mismatch_index": int(candidate_diagnostics[13]),
            "physical_direct_eta_audit_call_count": int(candidate_diagnostics[14]),
            "physical_direct_eta_calls_in_performance_path": 0,
            "direct_gain_bit_audit_enabled": bool(candidate_diagnostics[15]),
        }
    finally:
        _ACTIVE_CACHE = None


__all__ = (
    "ExactGainCache",
    "build_descent_support_chart",
    "build_unique_dyad_chart",
    "install",
    "load_library",
)
