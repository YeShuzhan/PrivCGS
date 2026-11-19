"""Exact native stable-heap backend for positive Hamiltonian completion."""

from __future__ import annotations

import ctypes
from dataclasses import replace
import importlib
import os
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).with_name("hamiltonian_heap_native_v1.cpp").resolve()
_LIBRARY = None
LAST_NATIVE_DIAGNOSTICS = None
LAST_AUDIT_TRACE = None


def load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    native_build = importlib.import_module("route_s_clean._native_build")
    library_path = native_build.build_shared_library(
        "positive-hamiltonian-heap-v1",
        SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++17", "-shared",
            "-fno-fast-math", "-ffp-contract=off", "-fno-associative-math",
            "-fno-finite-math-only", "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_positive_hamiltonian_heap_v1
    function.argtypes = [
        ctypes.c_int64, ctypes.c_int64, i64, f64, f64,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        i64, i64, i64, f64, i64, f64, i64, f64, i64,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int32
    _LIBRARY = library
    return library


def positive_hamiltonian_completion(
    law,
    path,
    flagged_core,
    flagged_completion,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    feature_cache=None,
):
    """Preserve the frozen completion while moving only its heap state machine native."""

    global LAST_NATIVE_DIAGNOSTICS, LAST_AUDIT_TRACE
    production = importlib.import_module("route_s_clean.hamiltonian_completion")
    order = np.ascontiguousarray(path, dtype=np.int64)
    degree = np.ascontiguousarray(source_degree, dtype=np.float64)
    target = np.ascontiguousarray(target_endpoint_mass, dtype=np.float64)
    sigma = float(degree_noise_std)
    core_score = production.efficient_spectral_score_messages(
        law,
        order,
        flagged_core,
        degree,
        target,
        sigma,
        feature_cache=feature_cache,
        _calibration_only=True,
        _retain_spectral_columns=False,
    )
    scale = float(core_score.residual_scale)
    coefficient = float(core_score.quadratic_coefficient)
    full = production.combine_tapes(flagged_core, flagged_completion)
    spectral = production.compile_spectral_columns(
        law, full, feature_cache=feature_cache
    )
    completion_negative_spectral, completion_positive_spectral = (
        production._completion_spectral_columns(
            flagged_core,
            flagged_completion,
            spectral.negative_delta,
            spectral.positive_delta,
        )
    )
    core_negative = np.diff(flagged_core.negative_offsets)
    core_positive = np.diff(flagged_core.positive_offsets)
    completion_negative = np.diff(flagged_completion.negative_offsets)
    completion_positive = np.diff(flagged_completion.positive_offsets)
    zero_fisher_core = bool(
        core_score.diagnostics.get("fisher_quotient_zero_score_branch", False)
    )
    if zero_fisher_core:
        accepted = np.zeros(full.bundle_count, dtype=np.int64)
        filtered = production.directional_prefixes(
            flagged_completion,
            np.zeros_like(completion_negative),
            np.zeros_like(completion_positive),
        )
        spectral_full = replace(
            full,
            negative_energy=np.ascontiguousarray(
                spectral.negative_delta, dtype=np.float64
            ),
            positive_energy=np.ascontiguousarray(
                spectral.positive_delta, dtype=np.float64
            ),
            structural_gain=np.ascontiguousarray(
                spectral.additive_gain, dtype=np.float64
            ),
        )
        final_spectral = production.directional_prefixes(
            spectral_full, core_negative, core_positive
        )
        LAST_NATIVE_DIAGNOSTICS = {"zero_fisher_reference_branch": True}
        LAST_AUDIT_TRACE = None
        return production.HamiltonianCompletion(
            filtered_completion=filtered,
            final_spectral_tape=final_spectral,
            accepted_count=accepted,
            statistics={
                **dict(spectral.statistics),
                "core_residual_scale": scale,
                "core_signed_degree_energy_coefficient": coefficient,
                "full_completion_unit_count": int(
                    flagged_completion.negative_energy.size
                    + flagged_completion.positive_energy.size
                ),
                "score_admitted_completion_unit_count": 0,
                "minimum_accepted_hamiltonian_gain": 0.0,
                "terminal_maximum_hamiltonian_gain": float("-inf"),
                "all_accepted_gains_strictly_positive": True,
                "terminal_frontier_nonpositive": True,
                "completion_is_bundlewise_prefix": True,
                "local_heap_update_radius": 1,
                "fisher_quotient_zero_completion_branch": True,
                "hidden_threshold_count": 0,
                "complete_dyad_enumeration": False,
                "complete_wedge_enumeration": False,
                "common_neighbor_intersection_work": 0,
            },
        )

    sign = np.zeros(full.bundle_count, dtype=np.int64)
    core_capacity = np.zeros(full.bundle_count, dtype=np.int64)
    positive_completion = completion_positive > 0
    negative_completion = completion_negative > 0
    sign[positive_completion] = 1
    core_capacity[positive_completion] = core_positive[positive_completion]
    sign[negative_completion] = -1
    core_capacity[negative_completion] = core_negative[negative_completion]
    empty = ~(positive_completion | negative_completion)
    positive_core = empty & (core_positive > 0)
    negative_core = empty & ~positive_core & (core_negative > 0)
    sign[positive_core] = 1
    core_capacity[positive_core] = core_positive[positive_core]
    sign[negative_core] = -1
    core_capacity[negative_core] = core_negative[negative_core]
    sign = np.ascontiguousarray(sign)
    core_capacity = np.ascontiguousarray(core_capacity)
    accepted = np.zeros(full.bundle_count, dtype=np.int64)
    statistics = np.empty(2, dtype=np.float64)
    diagnostics = np.zeros(8, dtype=np.int64)
    audit_enabled = os.environ.get("ROUTE_S_HAMILTONIAN_AUDIT", "0").strip() not in (
        "", "0"
    )
    audit_capacity = int(
        flagged_completion.negative_energy.size
        + flagged_completion.positive_energy.size
    ) if audit_enabled else 0
    audit_index = (
        np.empty(audit_capacity, dtype=np.int64)
        if audit_enabled else np.empty(0, dtype=np.int64)
    )
    audit_gain_bits = (
        np.empty(audit_capacity, dtype=np.uint64)
        if audit_enabled else np.empty(0, dtype=np.uint64)
    )
    status = int(load_library().route_s_positive_hamiltonian_heap_v1(
        full.bundle_count,
        law.n_nodes,
        order,
        degree,
        target,
        float(2.0 * sigma**2),
        scale,
        coefficient,
        sign,
        core_capacity,
        np.ascontiguousarray(flagged_completion.negative_offsets, dtype=np.int64),
        np.ascontiguousarray(completion_negative_spectral, dtype=np.float64),
        np.ascontiguousarray(flagged_completion.positive_offsets, dtype=np.int64),
        np.ascontiguousarray(completion_positive_spectral, dtype=np.float64),
        accepted,
        statistics,
        diagnostics,
        audit_capacity,
        None if not audit_enabled else ctypes.c_void_p(audit_index.ctypes.data),
        None if not audit_enabled else ctypes.c_void_p(audit_gain_bits.ctypes.data),
    ))
    if status:
        raise RuntimeError(f"native positive Hamiltonian heap returned {status}")
    accepted_total = int(diagnostics[5])
    LAST_NATIVE_DIAGNOSTICS = {
        "heap_push_count": int(diagnostics[0]),
        "heap_pop_count": int(diagnostics[1]),
        "stale_pop_count": int(diagnostics[2]),
        "current_exhausted_pop_count": int(diagnostics[3]),
        "score_call_count": int(diagnostics[4]),
        "accepted_count": accepted_total,
        "maximum_heap_size": int(diagnostics[6]),
        "terminal_heap_size": int(diagnostics[7]),
        "audit_enabled": bool(audit_enabled),
    }
    LAST_AUDIT_TRACE = (
        None if not audit_enabled else {
            "accepted_index": np.ascontiguousarray(audit_index[:accepted_total]),
            "accepted_gain_bits": np.ascontiguousarray(
                audit_gain_bits[:accepted_total]
            ),
        }
    )
    filtered_negative = np.where(sign < 0, accepted, 0)
    filtered_positive = np.where(sign > 0, accepted, 0)
    filtered = production.directional_prefixes(
        flagged_completion, filtered_negative, filtered_positive
    )
    spectral_full = replace(
        full,
        negative_energy=np.ascontiguousarray(
            spectral.negative_delta, dtype=np.float64
        ),
        positive_energy=np.ascontiguousarray(
            spectral.positive_delta, dtype=np.float64
        ),
        structural_gain=np.ascontiguousarray(
            spectral.additive_gain, dtype=np.float64
        ),
    )
    final_spectral = production.directional_prefixes(
        spectral_full,
        core_negative + filtered_negative,
        core_positive + filtered_positive,
    )
    return production.HamiltonianCompletion(
        filtered_completion=filtered,
        final_spectral_tape=final_spectral,
        accepted_count=accepted,
        statistics={
            **dict(spectral.statistics),
            "core_residual_scale": scale,
            "core_signed_degree_energy_coefficient": coefficient,
            "full_completion_unit_count": int(
                flagged_completion.negative_energy.size
                + flagged_completion.positive_energy.size
            ),
            "score_admitted_completion_unit_count": int(np.sum(accepted)),
            "minimum_accepted_hamiltonian_gain": float(statistics[0]),
            "terminal_maximum_hamiltonian_gain": float(statistics[1]),
            "all_accepted_gains_strictly_positive": True,
            "terminal_frontier_nonpositive": bool(statistics[1] <= 0.0),
            "completion_is_bundlewise_prefix": True,
            "local_heap_update_radius": 1,
            "hidden_threshold_count": 0,
            "complete_dyad_enumeration": False,
            "complete_wedge_enumeration": False,
            "common_neighbor_intersection_work": 0,
        },
    )


__all__ = (
    "LAST_AUDIT_TRACE",
    "LAST_NATIVE_DIAGNOSTICS",
    "load_library",
    "positive_hamiltonian_completion",
)
