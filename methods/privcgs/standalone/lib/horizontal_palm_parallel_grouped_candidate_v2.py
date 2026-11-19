"""Exact horizontal Palm backend with certified support, parallel lookup, and shared grouping."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).with_name("palm_count_lookup_parallel_v1.cpp").resolve()
_LIBRARY = None
LAST_DIAGNOSTICS = None


@dataclass(frozen=True)
class _QueryGrouping:
    query_unique: np.ndarray
    first_position: np.ndarray
    query_position: np.ndarray


def load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    native_build = importlib.import_module("route_s_clean._native_build")
    library_path = native_build.build_shared_library(
        "palm-count-lookup-parallel-v1",
        SOURCE,
        compiler_flags=(
            "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-pthread",
            "-static-libgcc", "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_parallel_count_lookup_v1
    function.argtypes = [
        u64, i64, ctypes.c_int64, u64, ctypes.c_int64, i64, ctypes.c_int32
    ]
    function.restype = ctypes.c_int32
    _LIBRARY = library
    return library


def _parallel_count_lookup(unique, count, query):
    unique_array = np.ascontiguousarray(unique, dtype=np.uint64)
    count_array = np.ascontiguousarray(count, dtype=np.int64)
    query_array = np.ascontiguousarray(query, dtype=np.uint64)
    result = np.empty(query_array.size, dtype=np.int64)
    threads = max(1, int(os.cpu_count() or 1))
    status = int(load_library().route_s_parallel_count_lookup_v1(
        unique_array,
        count_array,
        unique_array.size,
        query_array,
        query_array.size,
        result,
        threads,
    ))
    if status:
        raise RuntimeError(f"parallel count lookup returned {status}")
    return result


def _prepare_grouping(production, query_type) -> _QueryGrouping:
    query = np.asarray(query_type)
    query_unique, first_position = np.unique(query, return_index=True)
    if query_unique.size <= np.iinfo(np.uint16).max:
        position_dtype = np.uint16
    elif query_unique.size <= np.iinfo(np.uint32).max:
        position_dtype = np.uint32
    else:
        position_dtype = np.uint64
    flat_query = query.reshape(-1)
    query_position = np.empty(flat_query.size, dtype=position_dtype)
    for start in range(0, flat_query.size, production._COUNT_CHUNK):
        stop = min(start + production._COUNT_CHUNK, flat_query.size)
        query_position[start:stop] = np.searchsorted(
            query_unique, flat_query[start:stop]
        ).astype(position_dtype, copy=False)
    return _QueryGrouping(
        query_unique=query_unique,
        first_position=first_position,
        query_position=query_position,
    )


def _ratio_from_grouping(
    production,
    source_type,
    exposure_type,
    grouping: _QueryGrouping,
    source_denominator: int,
    exposure_denominator: int,
    *,
    source_weight=None,
    exposure_weight=None,
    query_self_weight=None,
):
    source_unique, source_count = production._counts(source_type, source_weight)
    exposure_unique, exposure_count = production._counts(
        exposure_type, exposure_weight
    )
    unique_source = production._count_lookup(
        source_unique, source_count, grouping.query_unique
    )
    unique_exposure = production._count_lookup(
        exposure_unique, exposure_count, grouping.query_unique
    )
    if query_self_weight is None:
        unique_weight = np.ones(grouping.query_unique.size, dtype=np.int64)
    else:
        weights = np.asarray(query_self_weight, dtype=np.int64)
        unique_weight = weights[grouping.first_position]
    unique_ratio = np.zeros(grouping.query_unique.size, dtype=np.float64)
    for index in range(grouping.query_unique.size):
        weight = int(unique_weight[index])
        if weight > 0:
            source_mass = (int(unique_source[index]) + weight) / (
                int(source_denominator) + weight
            )
            exposure_mass = (int(unique_exposure[index]) + weight) / (
                int(exposure_denominator) + weight
            )
            unique_ratio[index] = float(
                math.log(source_mass) - math.log(exposure_mass)
            )
    result = np.empty(grouping.query_position.size, dtype=np.float64)
    for start in range(0, result.size, production._COUNT_CHUNK):
        stop = min(start + production._COUNT_CHUNK, result.size)
        result[start:stop] = unique_ratio[
            grouping.query_position[start:stop].astype(np.int64, copy=False)
        ]
    return result, int(source_unique.size), int(exposure_unique.size)


def compile_horizontal_palm_score_columns(source, tape, public_seed: int):
    """Return frozen Palm columns with only exact implementation redundancy removed."""

    global LAST_DIAGNOSTICS
    production = importlib.import_module("route_s_clean.palm_coordinates")
    wedge_key, skipped = production.counter_palm_tokens(
        source.edge_array,
        source.csr_offsets,
        source.csr_neighbors,
        source.n_nodes,
        int(public_seed),
    )
    closure_key, closure_count = np.unique(wedge_key, return_counts=True)
    closure_count = closure_count.astype(np.int64)

    source_edge = np.asarray(source.edge_array, dtype=np.int64).reshape((-1, 2))
    source_key = (
        source_edge[:, 0].astype(np.uint64) * np.uint64(source.n_nodes)
        + source_edge[:, 1].astype(np.uint64)
    )
    source_closure = _parallel_count_lookup(
        closure_key, closure_count, source_key
    ).astype(np.int64, copy=False)
    degree = source.degree_array
    source_opportunity = (
        degree[source_edge[:, 0]] + degree[source_edge[:, 1]] - 2
    ).astype(np.int64)

    negative_edges = np.concatenate(
        (tape.negative_old_edge, tape.negative_new_edge), axis=0
    )
    positive_edges = np.concatenate(
        (tape.positive_old_edge, tape.positive_new_edge), axis=0
    )
    all_edges = np.concatenate((negative_edges, positive_edges), axis=0)
    all_key = (
        all_edges[:, 0].astype(np.uint64) * np.uint64(source.n_nodes)
        + all_edges[:, 1].astype(np.uint64)
    )
    all_closure = _parallel_count_lookup(
        closure_key, closure_count, all_key
    ).astype(np.int64, copy=False)

    negative_units = int(tape.negative_energy.size)
    positive_units = int(tape.positive_energy.size)
    negative_old = slice(0, negative_units)
    negative_new = slice(negative_units, 2 * negative_units)
    positive_base = 2 * negative_units
    positive_old = slice(positive_base, positive_base + positive_units)
    positive_new = slice(
        positive_base + positive_units, positive_base + 2 * positive_units
    )
    all_opportunity = (
        degree[all_edges[:, 0]] + degree[all_edges[:, 1]]
    ).astype(np.int64)
    all_opportunity[negative_old] -= 2
    all_opportunity[positive_old] -= 2

    type_base = np.uint64(2 * source.n_nodes + 1)
    source_type = (
        source_closure.astype(np.uint64) * type_base
        + source_opportunity.astype(np.uint64)
    )
    exposure_type = (
        all_closure.astype(np.uint64) * type_base
        + all_opportunity.astype(np.uint64)
    )
    grouping = _prepare_grouping(production, exposure_type)
    occupation_ratio, source_type_count, exposure_type_count = (
        _ratio_from_grouping(
            production,
            source_type,
            exposure_type,
            grouping,
            source.edge_count,
            all_edges.shape[0],
        )
    )
    occupation_negative = production._paired_delta(
        occupation_ratio, negative_old, negative_new
    )
    occupation_positive = production._paired_delta(
        occupation_ratio, positive_old, positive_new
    )
    del occupation_ratio

    source_total = int(np.sum(source_opportunity))
    exposure_total = int(np.sum(all_opportunity))
    palm_ratio, palm_source_types, palm_exposure_types = (
        _ratio_from_grouping(
            production,
            source_type,
            exposure_type,
            grouping,
            source_total,
            exposure_total,
            source_weight=source_opportunity,
            exposure_weight=all_opportunity,
            query_self_weight=all_opportunity,
        )
    )
    palm_negative = production._paired_delta(
        palm_ratio, negative_old, negative_new
    )
    palm_positive = production._paired_delta(
        palm_ratio, positive_old, positive_new
    )
    palm_additive = production._increment(tape, palm_negative, palm_positive)
    query_unique_count = int(grouping.query_unique.size)
    query_position_dtype = grouping.query_position.dtype.str
    del palm_ratio, palm_negative, palm_positive, source_type, exposure_type, grouping

    closure_negative = all_closure[negative_new] - all_closure[negative_old]
    closure_positive = all_closure[positive_new] - all_closure[positive_old]
    closure_increment_additive = production._increment(
        tape, closure_negative, closure_positive
    )
    negative_donor, negative_receiver = production._relocation_roles(
        tape.negative_old_edge, tape.negative_new_edge
    )
    positive_donor, positive_receiver = production._relocation_roles(
        tape.positive_old_edge, tape.positive_new_edge
    )
    wedge_negative = degree[negative_receiver] - degree[negative_donor] + 1
    wedge_positive = degree[positive_receiver] - degree[positive_donor] + 1
    wedge_increment_additive = production._increment(
        tape, wedge_negative, wedge_positive
    )
    del negative_donor, negative_receiver, positive_donor, positive_receiver

    stub_mass = int(np.sum(degree))
    wedge_count = int(np.sum(degree * (degree - 1) // 2))
    active_tokens = int(wedge_key.size)
    raw_probability = (
        float(float(2 * wedge_count) ** 2 / float(stub_mass) ** 3)
        if stub_mass > 0 else 0.0
    )
    probability = float(min(max(raw_probability, 0.0), 1.0))
    variance_mass = float(
        probability * (1.0 - probability) * max(active_tokens, 1)
    )
    scale = float(math.sqrt(variance_mass)) if variance_mass > 0.0 else 0.0
    if scale > 0.0:
        configuration_negative = np.fromiter(
            (
                float(float(int(closed) - probability * int(wedge)) / scale)
                for closed, wedge in zip(closure_negative, wedge_negative)
            ), dtype=np.float64, count=negative_units,
        )
        configuration_positive = np.fromiter(
            (
                float(float(int(closed) - probability * int(wedge)) / scale)
                for closed, wedge in zip(closure_positive, wedge_positive)
            ), dtype=np.float64, count=positive_units,
        )
    else:
        configuration_negative = np.zeros(negative_units, dtype=np.float64)
        configuration_positive = np.zeros(positive_units, dtype=np.float64)
    configuration_increment_additive = production._increment(
        tape, configuration_negative, configuration_positive
    )
    source_closed = int(np.sum(source_closure))
    LAST_DIAGNOSTICS = {
        "parallel_lookup_thread_count": max(1, int(os.cpu_count() or 1)),
        "source_lookup_count": int(source_key.size),
        "exposure_lookup_count": int(all_key.size),
        "query_unique_count": query_unique_count,
        "query_position_dtype": query_position_dtype,
    }
    return production.HorizontalPalmScoreColumns(
        occupation_negative_delta=occupation_negative,
        occupation_positive_delta=occupation_positive,
        palm_additive=palm_additive,
        closure_increment_additive=closure_increment_additive,
        wedge_increment_additive=wedge_increment_additive,
        configuration_increment_additive=configuration_increment_additive,
        statistics={
            "source_type_count": source_type_count,
            "exposure_type_count": exposure_type_count,
            "palm_source_type_count": palm_source_types,
            "palm_exposure_type_count": palm_exposure_types,
            "source_closed_token_count": source_closed,
            "source_active_token_count": active_tokens,
            "source_skipped_token_count": int(skipped),
            "token_mass_conserved": bool(active_tokens + int(skipped) == source.edge_count),
            "configuration_probability": probability,
            "configuration_scale": scale,
            "aligned_column_count": 5,
            "common_neighbor_query_count": 0,
            "intersection_work": 0,
            "complete_wedge_enumeration": False,
            "horizontal_minimum_sufficient_state": True,
            "complexity": "O(n+m+U log U) time and O(n+m+U) packed state",
        },
    )


__all__ = (
    "LAST_DIAGNOSTICS",
    "compile_horizontal_palm_score_columns",
    "load_library",
)
