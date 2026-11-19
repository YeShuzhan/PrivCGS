"""Sparse positive-Hamiltonian completion on packed relocation prefixes."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .chain_moments import efficient_spectral_score_messages
from .contracts import PackedRelocationTape, RecoveredGraphLaw
from .degree_chain import combine_tapes, directional_prefixes
from .structural_columns import compile_spectral_columns


@dataclass(frozen=True)
class HamiltonianCompletion:
    filtered_completion: PackedRelocationTape
    final_spectral_tape: PackedRelocationTape
    accepted_count: np.ndarray
    statistics: Mapping[str, object]


def _completion_spectral_columns(core, completion, negative, positive):
    negative_position = (
        np.arange(completion.negative_energy.size, dtype=np.int64)
        + core.negative_offsets[completion.negative_bundle + 1]
    )
    positive_position = (
        np.arange(completion.positive_energy.size, dtype=np.int64)
        + core.positive_offsets[completion.positive_bundle + 1]
    )
    return (
        np.ascontiguousarray(negative[negative_position], dtype=np.float64),
        np.ascontiguousarray(positive[positive_position], dtype=np.float64),
    )


def positive_hamiltonian_completion(
    law: RecoveredGraphLaw,
    path,
    flagged_core: PackedRelocationTape,
    flagged_completion: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    feature_cache=None,
) -> HamiltonianCompletion:
    """Admit the unique greedy prefix while its exact local gain is positive."""

    order = np.ascontiguousarray(path, dtype=np.int64)
    degree = np.ascontiguousarray(source_degree, dtype=np.float64)
    target = np.ascontiguousarray(target_endpoint_mass, dtype=np.float64)
    sigma = float(degree_noise_std)
    core_score = efficient_spectral_score_messages(
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
    full = combine_tapes(flagged_core, flagged_completion)
    spectral = compile_spectral_columns(law, full, feature_cache=feature_cache)
    completion_negative_spectral, completion_positive_spectral = (
        _completion_spectral_columns(
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
        # A unit-Fisher Hamiltonian direction does not exist on a one-path core.
        # In the quotient tangent space its only parameter-free representative
        # is zero, so no completion unit has a certified positive normalized
        # gain.  The empty bundlewise prefix is the unique minimal extension.
        accepted = np.zeros(full.bundle_count, dtype=np.int64)
        filtered = directional_prefixes(
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
        final_spectral = directional_prefixes(
            spectral_full, core_negative, core_positive
        )
        return HamiltonianCompletion(
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

    flow = sign * core_capacity
    realized = degree.copy()
    for index, value in enumerate(flow):
        realized[int(order[index])] -= int(value)
        realized[int(order[index + 1])] += int(value)
    accepted = np.zeros(full.bundle_count, dtype=np.int64)
    versions = np.zeros(full.bundle_count, dtype=np.int64)
    heap: list[tuple[float, int, int]] = []

    def spectral_value(index, position):
        if sign[index] > 0:
            start = int(flagged_completion.positive_offsets[index])
            stop = int(flagged_completion.positive_offsets[index + 1])
            return (
                None
                if start + position >= stop
                else float(completion_positive_spectral[start + position])
            )
        if sign[index] < 0:
            start = int(flagged_completion.negative_offsets[index])
            stop = int(flagged_completion.negative_offsets[index + 1])
            return (
                None
                if start + position >= stop
                else float(completion_negative_spectral[start + position])
            )
        return None

    def score(index):
        position = int(accepted[index])
        value = spectral_value(index, position)
        if value is None:
            return None
        direction = int(sign[index])
        left = int(order[index])
        right = int(order[index + 1])
        left_residual = float(realized[left] - target[left])
        right_residual = float(realized[right] - target[right])
        delta_degree = (
            (left_residual - direction) ** 2
            - left_residual**2
            + (right_residual + direction) ** 2
            - right_residual**2
        ) / (2.0 * sigma**2)
        return float(value / scale - coefficient * delta_degree)

    def push(index):
        if 0 <= index < full.bundle_count:
            value = score(index)
            if value is not None:
                heapq.heappush(heap, (-float(value), int(index), int(versions[index])))

    for index in range(full.bundle_count):
        push(index)
    accepted_scores = []
    terminal_maximum = float("-inf")
    while heap:
        _negative, index, version = heapq.heappop(heap)
        if version != int(versions[index]):
            continue
        value = score(index)
        if value is None:
            continue
        if value <= 0.0:
            terminal_maximum = float(value)
            break
        accepted_scores.append(float(value))
        direction = int(sign[index])
        accepted[index] += 1
        flow[index] += direction
        left = int(order[index])
        right = int(order[index + 1])
        realized[left] -= direction
        realized[right] += direction
        for affected in (index - 1, index, index + 1):
            if 0 <= affected < full.bundle_count:
                versions[affected] += 1
                push(affected)
    filtered_negative = np.where(sign < 0, accepted, 0)
    filtered_positive = np.where(sign > 0, accepted, 0)
    filtered = directional_prefixes(
        flagged_completion, filtered_negative, filtered_positive
    )
    spectral_full = replace(
        full,
        negative_energy=np.ascontiguousarray(spectral.negative_delta, dtype=np.float64),
        positive_energy=np.ascontiguousarray(spectral.positive_delta, dtype=np.float64),
        structural_gain=np.ascontiguousarray(spectral.additive_gain, dtype=np.float64),
    )
    final_spectral = directional_prefixes(
        spectral_full,
        core_negative + filtered_negative,
        core_positive + filtered_positive,
    )
    return HamiltonianCompletion(
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
            "minimum_accepted_hamiltonian_gain": float(
                min(accepted_scores, default=0.0)
            ),
            "terminal_maximum_hamiltonian_gain": terminal_maximum,
            "all_accepted_gains_strictly_positive": bool(
                all(value > 0.0 for value in accepted_scores)
            ),
            "terminal_frontier_nonpositive": bool(terminal_maximum <= 0.0),
            "completion_is_bundlewise_prefix": True,
            "local_heap_update_radius": 1,
            "hidden_threshold_count": 0,
            "complete_dyad_enumeration": False,
            "complete_wedge_enumeration": False,
            "common_neighbor_intersection_work": 0,
        },
    )


__all__ = ("HamiltonianCompletion", "positive_hamiltonian_completion")
