"""Aligned sufficient-statistic columns from one quenched Palm-token measure."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape
from .graph_carrier import CSRGraph
from .native import ragged_prefix
from .palm_native import counter_palm_tokens


_COUNT_CHUNK = 1 << 20


@dataclass(frozen=True)
class PalmScoreColumns:
    occupation_negative_delta: np.ndarray
    occupation_positive_delta: np.ndarray
    occupation_additive: np.ndarray
    palm_negative_delta: np.ndarray
    palm_positive_delta: np.ndarray
    palm_additive: np.ndarray
    closure_negative_delta: np.ndarray
    closure_positive_delta: np.ndarray
    closure_additive: np.ndarray
    closure_increment_additive: np.ndarray
    wedge_negative_delta: np.ndarray
    wedge_positive_delta: np.ndarray
    wedge_additive: np.ndarray
    wedge_increment_additive: np.ndarray
    configuration_negative_delta: np.ndarray
    configuration_positive_delta: np.ndarray
    configuration_additive: np.ndarray
    configuration_increment_additive: np.ndarray
    phase_negative_delta: np.ndarray
    phase_positive_delta: np.ndarray
    phase_additive: np.ndarray
    statistics: Mapping[str, object]


@dataclass(frozen=True)
class HorizontalPalmScoreColumns:
    """Minimum Palm sufficient state consumed by the horizontal endpoint branch."""

    occupation_negative_delta: np.ndarray
    occupation_positive_delta: np.ndarray
    palm_additive: np.ndarray
    closure_increment_additive: np.ndarray
    wedge_increment_additive: np.ndarray
    configuration_increment_additive: np.ndarray
    statistics: Mapping[str, object]


@dataclass(frozen=True)
class VerticalPalmScoreColumns:
    """Minimum Palm sufficient state consumed by the vertical endpoint branch."""

    closure_increment_additive: np.ndarray
    wedge_increment_additive: np.ndarray
    configuration_increment_additive: np.ndarray
    phase_additive: np.ndarray
    statistics: Mapping[str, object]


def _count_lookup(unique, count, query):
    query_array = np.asarray(query, dtype=unique.dtype)
    result = np.zeros(query_array.shape, dtype=np.int64)
    if unique.size == 0:
        return result
    position = np.searchsorted(unique, query_array)
    valid = position < unique.size
    clipped = np.minimum(position, unique.size - 1)
    valid &= unique[clipped] == query_array
    result[valid] = count[clipped[valid]]
    return result


def _weighted_counts(keys, weights):
    key_array = np.asarray(keys)
    weight_array = np.asarray(weights, dtype=np.int64)
    if key_array.size == 0:
        return np.empty(0, dtype=key_array.dtype), np.empty(0, dtype=np.int64)
    key_parts = []
    count_parts = []
    for start in range(0, key_array.size, _COUNT_CHUNK):
        stop = min(start + _COUNT_CHUNK, key_array.size)
        unique, inverse = np.unique(key_array[start:stop], return_inverse=True)
        total = np.zeros(unique.size, dtype=np.int64)
        np.add.at(total, inverse, weight_array[start:stop])
        key_parts.append(unique)
        count_parts.append(total)
    if len(key_parts) == 1:
        return key_parts[0], count_parts[0]
    merged_key = np.concatenate(key_parts)
    merged_count = np.concatenate(count_parts)
    unique, inverse = np.unique(merged_key, return_inverse=True)
    total = np.zeros(unique.size, dtype=np.int64)
    np.add.at(total, inverse, merged_count)
    return unique, total


def _counts(keys, weights=None):
    if weights is None:
        unique, count = np.unique(np.asarray(keys), return_counts=True)
        return unique, count.astype(np.int64, copy=False)
    return _weighted_counts(keys, weights)


def log_ratio_by_type(
    source_type,
    exposure_type,
    query_type,
    source_denominator: int,
    exposure_denominator: int,
    *,
    source_weight=None,
    exposure_weight=None,
    query_self_weight=None,
):
    source_unique, source_count = _counts(source_type, source_weight)
    exposure_unique, exposure_count = _counts(exposure_type, exposure_weight)
    query = np.asarray(query_type)
    result = np.empty(query.shape, dtype=np.float64)
    if query.size == 0:
        return result, int(source_unique.size), int(exposure_unique.size)
    query_unique, first_position = np.unique(query, return_index=True)
    unique_source = _count_lookup(source_unique, source_count, query_unique)
    unique_exposure = _count_lookup(exposure_unique, exposure_count, query_unique)
    if query_self_weight is None:
        unique_weight = np.ones(query_unique.size, dtype=np.int64)
    else:
        weights = np.asarray(query_self_weight, dtype=np.int64)
        unique_weight = weights[first_position]
    unique_ratio = np.zeros(query_unique.size, dtype=np.float64)
    for index in range(query_unique.size):
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
    flat_query = query.reshape(-1)
    flat_result = result.reshape(-1)
    for start in range(0, flat_query.size, _COUNT_CHUNK):
        stop = min(start + _COUNT_CHUNK, flat_query.size)
        flat_result[start:stop] = unique_ratio[
            np.searchsorted(query_unique, flat_query[start:stop])
        ]
    return result, int(source_unique.size), int(exposure_unique.size)


def _prefix(tape, negative, positive):
    return ragged_prefix(
        tape.state_offsets,
        tape.negative_offsets,
        tape.positive_offsets,
        np.asarray(negative, dtype=np.float64),
        np.asarray(positive, dtype=np.float64),
    )


def _increment(tape, negative, positive):
    marked_negative = np.fromiter(
        (
            float(float(base) + float(delta))
            for base, delta in zip(tape.negative_energy, negative)
        ),
        dtype=np.float64,
        count=tape.negative_energy.size,
    )
    marked_positive = np.fromiter(
        (
            float(float(base) + float(delta))
            for base, delta in zip(tape.positive_energy, positive)
        ),
        dtype=np.float64,
        count=tape.positive_energy.size,
    )
    return _prefix(tape, marked_negative, marked_positive) - tape.structural_gain


def _paired_delta(values, old_slice, new_slice):
    old = np.asarray(values[old_slice])
    new = np.asarray(values[new_slice])
    return np.fromiter(
        (float(float(right) - float(left)) for left, right in zip(old, new)),
        dtype=np.float64,
        count=old.size,
    )


def _edge_coordinates(source: CSRGraph, closure_key, closure_count, edges):
    edge = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    key = edge[:, 0].astype(np.uint64) * np.uint64(source.n_nodes) + edge[:, 1].astype(np.uint64)
    closure = _count_lookup(closure_key, closure_count, key)
    position = np.searchsorted(source.edge_keys, key)
    present = np.zeros(key.size, dtype=np.int64)
    if source.edge_keys.size:
        clipped = np.minimum(position, source.edge_keys.size - 1)
        present = (source.edge_keys[clipped] == key).astype(np.int64)
    opportunity = (
        source.degree_array[edge[:, 0]]
        + source.degree_array[edge[:, 1]]
        - 2 * present
    ).astype(np.int64)
    return closure.astype(np.int64), opportunity


def _relocation_roles(old_edge, new_edge):
    old_zero_common = (
        (old_edge[:, 0] == new_edge[:, 0])
        | (old_edge[:, 0] == new_edge[:, 1])
    )
    common = np.where(old_zero_common, old_edge[:, 0], old_edge[:, 1])
    donor = np.where(old_zero_common, old_edge[:, 1], old_edge[:, 0])
    receiver = np.where(new_edge[:, 0] == common, new_edge[:, 1], new_edge[:, 0])
    return donor, receiver


def compile_palm_score_columns(
    source: CSRGraph,
    tape: PackedRelocationTape,
    component_fiber,
    public_seed: int,
) -> PalmScoreColumns:
    wedge_key, skipped = counter_palm_tokens(
        source.edge_array,
        source.csr_offsets,
        source.csr_neighbors,
        source.n_nodes,
        int(public_seed),
    )
    closure_key, closure_count = np.unique(wedge_key, return_counts=True)
    closure_count = closure_count.astype(np.int64)
    source_closure, source_opportunity = _edge_coordinates(
        source, closure_key, closure_count, source.edge_array
    )
    negative_edges = np.concatenate((tape.negative_old_edge, tape.negative_new_edge), axis=0)
    positive_edges = np.concatenate((tape.positive_old_edge, tape.positive_new_edge), axis=0)
    all_edges = np.concatenate((negative_edges, positive_edges), axis=0)
    all_closure, all_opportunity = _edge_coordinates(
        source, closure_key, closure_count, all_edges
    )
    negative_units = int(tape.negative_energy.size)
    positive_units = int(tape.positive_energy.size)
    negative_old = slice(0, negative_units)
    negative_new = slice(negative_units, 2 * negative_units)
    positive_base = 2 * negative_units
    positive_old = slice(positive_base, positive_base + positive_units)
    positive_new = slice(positive_base + positive_units, positive_base + 2 * positive_units)
    type_base = np.uint64(2 * source.n_nodes + 1)
    source_type = source_closure.astype(np.uint64) * type_base + source_opportunity.astype(np.uint64)
    exposure_type = all_closure.astype(np.uint64) * type_base + all_opportunity.astype(np.uint64)
    occupation_ratio, source_type_count, exposure_type_count = log_ratio_by_type(
        source_type, exposure_type, exposure_type,
        source.edge_count, all_edges.shape[0],
    )
    occupation_negative = _paired_delta(occupation_ratio, negative_old, negative_new)
    occupation_positive = _paired_delta(occupation_ratio, positive_old, positive_new)
    source_total = int(np.sum(source_opportunity))
    exposure_total = int(np.sum(all_opportunity))
    palm_ratio, palm_source_types, palm_exposure_types = log_ratio_by_type(
        source_type, exposure_type, exposure_type, source_total, exposure_total,
        source_weight=source_opportunity,
        exposure_weight=all_opportunity,
        query_self_weight=all_opportunity,
    )
    palm_negative = _paired_delta(palm_ratio, negative_old, negative_new)
    palm_positive = _paired_delta(palm_ratio, positive_old, positive_new)
    closure_negative = all_closure[negative_new] - all_closure[negative_old]
    closure_positive = all_closure[positive_new] - all_closure[positive_old]
    negative_donor, negative_receiver = _relocation_roles(
        tape.negative_old_edge, tape.negative_new_edge
    )
    positive_donor, positive_receiver = _relocation_roles(
        tape.positive_old_edge, tape.positive_new_edge
    )
    degree = source.degree_array
    wedge_negative = degree[negative_receiver] - degree[negative_donor] + 1
    wedge_positive = degree[positive_receiver] - degree[positive_donor] + 1
    stub_mass = int(np.sum(degree))
    wedge_count = int(np.sum(degree * (degree - 1) // 2))
    active_tokens = int(wedge_key.size)
    raw_probability = (
        float(float(2 * wedge_count) ** 2 / float(stub_mass) ** 3)
        if stub_mass > 0 else 0.0
    )
    probability = float(min(max(raw_probability, 0.0), 1.0))
    variance_mass = float(probability * (1.0 - probability) * max(active_tokens, 1))
    scale = float(math.sqrt(variance_mass)) if variance_mass > 0.0 else 0.0
    if scale > 0.0:
        configuration_negative = np.fromiter(
            (
                float(float(int(closed) - probability * int(wedge)) / scale)
                for closed, wedge in zip(closure_negative, wedge_negative)
            ),
            dtype=np.float64,
            count=negative_units,
        )
        configuration_positive = np.fromiter(
            (
                float(float(int(closed) - probability * int(wedge)) / scale)
                for closed, wedge in zip(closure_positive, wedge_positive)
            ),
            dtype=np.float64,
            count=positive_units,
        )
    else:
        configuration_negative = np.zeros(negative_units, dtype=np.float64)
        configuration_positive = np.zeros(positive_units, dtype=np.float64)
    component = np.asarray(component_fiber, dtype=np.int64)
    component_count = int(np.max(component, initial=0) + 1)
    source_component_low = np.minimum(
        component[source.edge_array[:, 0]], component[source.edge_array[:, 1]]
    )
    source_component_high = np.maximum(
        component[source.edge_array[:, 0]], component[source.edge_array[:, 1]]
    )
    exposure_component_low = np.minimum(
        component[all_edges[:, 0]], component[all_edges[:, 1]]
    )
    exposure_component_high = np.maximum(
        component[all_edges[:, 0]], component[all_edges[:, 1]]
    )
    coordinate_base = np.uint64(max(active_tokens, source.edge_count) + 1)
    phase_source_type = (
        (
            source_component_low.astype(np.uint64) * np.uint64(component_count)
            + source_component_high.astype(np.uint64)
        ) * coordinate_base + source_closure.astype(np.uint64)
    ) * type_base + source_opportunity.astype(np.uint64)
    phase_exposure_type = (
        (
            exposure_component_low.astype(np.uint64) * np.uint64(component_count)
            + exposure_component_high.astype(np.uint64)
        ) * coordinate_base + all_closure.astype(np.uint64)
    ) * type_base + all_opportunity.astype(np.uint64)
    phase_ratio, phase_source_types, phase_exposure_types = log_ratio_by_type(
        phase_source_type, phase_exposure_type, phase_exposure_type,
        source.edge_count, all_edges.shape[0],
    )
    phase_negative = _paired_delta(phase_ratio, negative_old, negative_new)
    phase_positive = _paired_delta(phase_ratio, positive_old, positive_new)
    source_closed = int(np.sum(source_closure))
    return PalmScoreColumns(
        occupation_negative,
        occupation_positive,
        _increment(tape, occupation_negative, occupation_positive),
        palm_negative,
        palm_positive,
        _increment(tape, palm_negative, palm_positive),
        np.asarray(closure_negative, dtype=np.float64),
        np.asarray(closure_positive, dtype=np.float64),
        _prefix(tape, closure_negative, closure_positive),
        _increment(tape, closure_negative, closure_positive),
        np.asarray(wedge_negative, dtype=np.float64),
        np.asarray(wedge_positive, dtype=np.float64),
        _prefix(tape, wedge_negative, wedge_positive),
        _increment(tape, wedge_negative, wedge_positive),
        configuration_negative,
        configuration_positive,
        _prefix(tape, configuration_negative, configuration_positive),
        _increment(tape, configuration_negative, configuration_positive),
        phase_negative,
        phase_positive,
        _prefix(tape, phase_negative, phase_positive),
        {
            "source_type_count": source_type_count,
            "exposure_type_count": exposure_type_count,
            "palm_source_type_count": palm_source_types,
            "palm_exposure_type_count": palm_exposure_types,
            "phase_source_type_count": phase_source_types,
            "phase_exposure_type_count": phase_exposure_types,
            "source_closed_token_count": source_closed,
            "source_active_token_count": active_tokens,
            "source_skipped_token_count": int(skipped),
            "token_mass_conserved": bool(active_tokens + int(skipped) == source.edge_count),
            "configuration_probability": probability,
            "configuration_scale": scale,
            "aligned_column_count": 6,
            "common_neighbor_query_count": 0,
            "intersection_work": 0,
            "complete_wedge_enumeration": False,
            "complexity": "O(n+m+U log U) time and O(n+m+U) packed state",
        },
    )


def compile_horizontal_palm_score_columns(
    source: CSRGraph,
    tape: PackedRelocationTape,
    public_seed: int,
) -> HorizontalPalmScoreColumns:
    """Compile exactly the Palm field projection read by the horizontal branch."""

    wedge_key, skipped = counter_palm_tokens(
        source.edge_array,
        source.csr_offsets,
        source.csr_neighbors,
        source.n_nodes,
        int(public_seed),
    )
    closure_key, closure_count = np.unique(wedge_key, return_counts=True)
    closure_count = closure_count.astype(np.int64)
    source_closure, source_opportunity = _edge_coordinates(
        source, closure_key, closure_count, source.edge_array
    )
    negative_edges = np.concatenate(
        (tape.negative_old_edge, tape.negative_new_edge), axis=0
    )
    positive_edges = np.concatenate(
        (tape.positive_old_edge, tape.positive_new_edge), axis=0
    )
    all_edges = np.concatenate((negative_edges, positive_edges), axis=0)
    all_closure, all_opportunity = _edge_coordinates(
        source, closure_key, closure_count, all_edges
    )
    negative_units = int(tape.negative_energy.size)
    positive_units = int(tape.positive_energy.size)
    negative_old = slice(0, negative_units)
    negative_new = slice(negative_units, 2 * negative_units)
    positive_base = 2 * negative_units
    positive_old = slice(positive_base, positive_base + positive_units)
    positive_new = slice(positive_base + positive_units, positive_base + 2 * positive_units)
    type_base = np.uint64(2 * source.n_nodes + 1)
    source_type = (
        source_closure.astype(np.uint64) * type_base
        + source_opportunity.astype(np.uint64)
    )
    exposure_type = (
        all_closure.astype(np.uint64) * type_base
        + all_opportunity.astype(np.uint64)
    )
    occupation_ratio, source_type_count, exposure_type_count = log_ratio_by_type(
        source_type,
        exposure_type,
        exposure_type,
        source.edge_count,
        all_edges.shape[0],
    )
    occupation_negative = _paired_delta(
        occupation_ratio, negative_old, negative_new
    )
    occupation_positive = _paired_delta(
        occupation_ratio, positive_old, positive_new
    )
    del occupation_ratio

    source_total = int(np.sum(source_opportunity))
    exposure_total = int(np.sum(all_opportunity))
    palm_ratio, palm_source_types, palm_exposure_types = log_ratio_by_type(
        source_type,
        exposure_type,
        exposure_type,
        source_total,
        exposure_total,
        source_weight=source_opportunity,
        exposure_weight=all_opportunity,
        query_self_weight=all_opportunity,
    )
    palm_negative = _paired_delta(palm_ratio, negative_old, negative_new)
    palm_positive = _paired_delta(palm_ratio, positive_old, positive_new)
    palm_additive = _increment(tape, palm_negative, palm_positive)
    del palm_ratio, palm_negative, palm_positive, source_type, exposure_type

    closure_negative = all_closure[negative_new] - all_closure[negative_old]
    closure_positive = all_closure[positive_new] - all_closure[positive_old]
    closure_increment_additive = _increment(
        tape, closure_negative, closure_positive
    )
    negative_donor, negative_receiver = _relocation_roles(
        tape.negative_old_edge, tape.negative_new_edge
    )
    positive_donor, positive_receiver = _relocation_roles(
        tape.positive_old_edge, tape.positive_new_edge
    )
    degree = source.degree_array
    wedge_negative = degree[negative_receiver] - degree[negative_donor] + 1
    wedge_positive = degree[positive_receiver] - degree[positive_donor] + 1
    wedge_increment_additive = _increment(tape, wedge_negative, wedge_positive)
    del negative_donor, negative_receiver, positive_donor, positive_receiver

    stub_mass = int(np.sum(degree))
    wedge_count = int(np.sum(degree * (degree - 1) // 2))
    active_tokens = int(wedge_key.size)
    raw_probability = (
        float(float(2 * wedge_count) ** 2 / float(stub_mass) ** 3)
        if stub_mass > 0
        else 0.0
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
            ),
            dtype=np.float64,
            count=negative_units,
        )
        configuration_positive = np.fromiter(
            (
                float(float(int(closed) - probability * int(wedge)) / scale)
                for closed, wedge in zip(closure_positive, wedge_positive)
            ),
            dtype=np.float64,
            count=positive_units,
        )
    else:
        configuration_negative = np.zeros(negative_units, dtype=np.float64)
        configuration_positive = np.zeros(positive_units, dtype=np.float64)
    configuration_increment_additive = _increment(
        tape, configuration_negative, configuration_positive
    )
    source_closed = int(np.sum(source_closure))
    return HorizontalPalmScoreColumns(
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
            "token_mass_conserved": bool(
                active_tokens + int(skipped) == source.edge_count
            ),
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


def compile_vertical_palm_score_columns(
    source: CSRGraph,
    tape: PackedRelocationTape,
    component_fiber,
    public_seed: int,
) -> VerticalPalmScoreColumns:
    """Compile exactly the Palm columns read by the vertical branch.

    The vertical anti-closure parent and its two endpoint laws never consume the
    occupation or Palm likelihood-ratio coordinates used by the horizontal
    exposure law. Omitting those two ratio constructions is therefore the
    consumer-minimal quotient of the common Palm tape.
    """

    wedge_key, skipped = counter_palm_tokens(
        source.edge_array,
        source.csr_offsets,
        source.csr_neighbors,
        source.n_nodes,
        int(public_seed),
    )
    closure_key, closure_count = np.unique(wedge_key, return_counts=True)
    closure_count = closure_count.astype(np.int64)
    source_closure, source_opportunity = _edge_coordinates(
        source, closure_key, closure_count, source.edge_array
    )
    negative_edges = np.concatenate(
        (tape.negative_old_edge, tape.negative_new_edge), axis=0
    )
    positive_edges = np.concatenate(
        (tape.positive_old_edge, tape.positive_new_edge), axis=0
    )
    all_edges = np.concatenate((negative_edges, positive_edges), axis=0)
    all_closure, all_opportunity = _edge_coordinates(
        source, closure_key, closure_count, all_edges
    )
    negative_units = int(tape.negative_energy.size)
    positive_units = int(tape.positive_energy.size)
    negative_old = slice(0, negative_units)
    negative_new = slice(negative_units, 2 * negative_units)
    positive_base = 2 * negative_units
    positive_old = slice(positive_base, positive_base + positive_units)
    positive_new = slice(
        positive_base + positive_units, positive_base + 2 * positive_units
    )
    type_base = np.uint64(2 * source.n_nodes + 1)
    source_type = (
        source_closure.astype(np.uint64) * type_base
        + source_opportunity.astype(np.uint64)
    )
    exposure_type = (
        all_closure.astype(np.uint64) * type_base
        + all_opportunity.astype(np.uint64)
    )
    source_type_count = int(np.unique(source_type).size)
    exposure_type_count = int(np.unique(exposure_type).size)

    closure_negative = all_closure[negative_new] - all_closure[negative_old]
    closure_positive = all_closure[positive_new] - all_closure[positive_old]
    closure_increment_additive = _increment(
        tape, closure_negative, closure_positive
    )
    negative_donor, negative_receiver = _relocation_roles(
        tape.negative_old_edge, tape.negative_new_edge
    )
    positive_donor, positive_receiver = _relocation_roles(
        tape.positive_old_edge, tape.positive_new_edge
    )
    degree = source.degree_array
    wedge_negative = degree[negative_receiver] - degree[negative_donor] + 1
    wedge_positive = degree[positive_receiver] - degree[positive_donor] + 1
    wedge_increment_additive = _increment(tape, wedge_negative, wedge_positive)
    stub_mass = int(np.sum(degree))
    wedge_count = int(np.sum(degree * (degree - 1) // 2))
    active_tokens = int(wedge_key.size)
    raw_probability = (
        float(float(2 * wedge_count) ** 2 / float(stub_mass) ** 3)
        if stub_mass > 0
        else 0.0
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
            ),
            dtype=np.float64,
            count=negative_units,
        )
        configuration_positive = np.fromiter(
            (
                float(float(int(closed) - probability * int(wedge)) / scale)
                for closed, wedge in zip(closure_positive, wedge_positive)
            ),
            dtype=np.float64,
            count=positive_units,
        )
    else:
        configuration_negative = np.zeros(negative_units, dtype=np.float64)
        configuration_positive = np.zeros(positive_units, dtype=np.float64)
    configuration_increment_additive = _increment(
        tape, configuration_negative, configuration_positive
    )
    component = np.asarray(component_fiber, dtype=np.int64)
    component_count = int(np.max(component, initial=0) + 1)
    source_component_low = np.minimum(
        component[source.edge_array[:, 0]], component[source.edge_array[:, 1]]
    )
    source_component_high = np.maximum(
        component[source.edge_array[:, 0]], component[source.edge_array[:, 1]]
    )
    exposure_component_low = np.minimum(
        component[all_edges[:, 0]], component[all_edges[:, 1]]
    )
    exposure_component_high = np.maximum(
        component[all_edges[:, 0]], component[all_edges[:, 1]]
    )
    coordinate_base = np.uint64(max(active_tokens, source.edge_count) + 1)
    phase_source_type = (
        (
            source_component_low.astype(np.uint64) * np.uint64(component_count)
            + source_component_high.astype(np.uint64)
        ) * coordinate_base + source_closure.astype(np.uint64)
    ) * type_base + source_opportunity.astype(np.uint64)
    phase_exposure_type = (
        (
            exposure_component_low.astype(np.uint64) * np.uint64(component_count)
            + exposure_component_high.astype(np.uint64)
        ) * coordinate_base + all_closure.astype(np.uint64)
    ) * type_base + all_opportunity.astype(np.uint64)
    phase_ratio, phase_source_types, phase_exposure_types = log_ratio_by_type(
        phase_source_type,
        phase_exposure_type,
        phase_exposure_type,
        source.edge_count,
        all_edges.shape[0],
    )
    phase_negative = _paired_delta(phase_ratio, negative_old, negative_new)
    phase_positive = _paired_delta(phase_ratio, positive_old, positive_new)
    phase_additive = _prefix(tape, phase_negative, phase_positive)
    source_closed = int(np.sum(source_closure))
    return VerticalPalmScoreColumns(
        closure_increment_additive=closure_increment_additive,
        wedge_increment_additive=wedge_increment_additive,
        configuration_increment_additive=configuration_increment_additive,
        phase_additive=phase_additive,
        statistics={
            "source_type_count": source_type_count,
            "exposure_type_count": exposure_type_count,
            "palm_source_type_count": source_type_count,
            "palm_exposure_type_count": exposure_type_count,
            "phase_source_type_count": phase_source_types,
            "phase_exposure_type_count": phase_exposure_types,
            "source_closed_token_count": source_closed,
            "source_active_token_count": active_tokens,
            "source_skipped_token_count": int(skipped),
            "token_mass_conserved": bool(
                active_tokens + int(skipped) == source.edge_count
            ),
            "configuration_probability": probability,
            "configuration_scale": scale,
            "aligned_column_count": 4,
            "common_neighbor_query_count": 0,
            "intersection_work": 0,
            "complete_wedge_enumeration": False,
            "vertical_minimum_sufficient_state": True,
            "complexity": "O(n+m+U log U) time and O(n+m+U) packed state",
        },
    )


def rebind_palm_score_columns(
    columns: PalmScoreColumns,
    tape: PackedRelocationTape,
) -> PalmScoreColumns:
    """Reuse immutable raw Palm coordinates on a binary64-equivalent tape chart."""

    return replace(
        columns,
        occupation_additive=_increment(
            tape,
            columns.occupation_negative_delta,
            columns.occupation_positive_delta,
        ),
        palm_additive=_increment(
            tape,
            columns.palm_negative_delta,
            columns.palm_positive_delta,
        ),
        closure_additive=_prefix(
            tape,
            columns.closure_negative_delta,
            columns.closure_positive_delta,
        ),
        closure_increment_additive=_increment(
            tape,
            columns.closure_negative_delta,
            columns.closure_positive_delta,
        ),
        wedge_additive=_prefix(
            tape,
            columns.wedge_negative_delta,
            columns.wedge_positive_delta,
        ),
        wedge_increment_additive=_increment(
            tape,
            columns.wedge_negative_delta,
            columns.wedge_positive_delta,
        ),
        configuration_additive=_prefix(
            tape,
            columns.configuration_negative_delta,
            columns.configuration_positive_delta,
        ),
        configuration_increment_additive=_increment(
            tape,
            columns.configuration_negative_delta,
            columns.configuration_positive_delta,
        ),
        phase_additive=_prefix(
            tape,
            columns.phase_negative_delta,
            columns.phase_positive_delta,
        ),
    )


__all__ = (
    "HorizontalPalmScoreColumns",
    "PalmScoreColumns",
    "VerticalPalmScoreColumns",
    "compile_horizontal_palm_score_columns",
    "compile_palm_score_columns",
    "compile_vertical_palm_score_columns",
    "log_ratio_by_type",
    "rebind_palm_score_columns",
)
