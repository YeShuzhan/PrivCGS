"""Exact finite-support Gaussian messages and backward sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape
from .degree_chain_native import ragged_forward


@dataclass(frozen=True)
class PackedGaussianMessages:
    state_offsets: np.ndarray
    state_value: np.ndarray
    structural_gain: np.ndarray
    forward_log_message: np.ndarray
    log_normalizer: float
    exact_pair_work: int
    linear_log_squared_budget: int
    statistics: Mapping[str, object]


def build_gaussian_messages(
    path,
    tape: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    quadratic_coefficient: float,
    structural_temperature: float = 1.0,
) -> PackedGaussianMessages:
    order = np.ascontiguousarray(path, dtype=np.int64)
    degree = np.ascontiguousarray(source_degree, dtype=np.float64)
    target = np.ascontiguousarray(target_endpoint_mass, dtype=np.float64)
    sigma = float(degree_noise_std)
    coefficient = float(quadratic_coefficient)
    temperature = float(structural_temperature)
    if tape.bundle_count == 0:
        normalizer = float(
            -coefficient * np.sum((degree - target) ** 2) / (2.0 * sigma * sigma)
        )
        return PackedGaussianMessages(
            tape.state_offsets,
            tape.state_value,
            tape.structural_gain,
            np.zeros(0, dtype=np.float64),
            normalizer,
            0,
            0,
            {"transition_count": 0, "native_ragged_forward": True},
        )
    nodes = order[: tape.bundle_count + 1]
    forward, normalizer, pair_work, budget = ragged_forward(
        tape.state_offsets,
        tape.state_value,
        tape.structural_gain,
        degree[nodes],
        target[nodes],
        coefficient / (2.0 * sigma * sigma),
        temperature,
    )
    return PackedGaussianMessages(
        state_offsets=np.ascontiguousarray(tape.state_offsets, dtype=np.int64),
        state_value=np.ascontiguousarray(tape.state_value, dtype=np.int64),
        structural_gain=np.ascontiguousarray(tape.structural_gain, dtype=np.float64),
        forward_log_message=forward,
        log_normalizer=normalizer,
        exact_pair_work=pair_work,
        linear_log_squared_budget=budget,
        statistics={
            "transition_count": max(tape.bundle_count - 1, 0),
            "exact_pair_work": pair_work,
            "linear_log_squared_budget": budget,
            "pair_work_within_certificate": pair_work <= budget,
            "complete_state_pair_matrix_materialized": False,
            "complete_dyad_enumeration": False,
            "complete_wedge_enumeration": False,
            "dense_n_by_n_state": False,
            "native_ragged_forward": True,
        },
    )


def _column_logsumexp(values) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    maximum = np.max(array, axis=0, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(array - maximum), axis=0, keepdims=True)
    )
    return float(result.reshape(-1)[0])


def _guarded_exact_transition(
    previous_state,
    current_state,
    previous_forward,
    degree_value: float,
    target_value: float,
    quadratic: float,
):
    previous = np.asarray(previous_state, dtype=np.float64)
    current = np.asarray(current_state, dtype=np.float64)
    forward = np.asarray(previous_forward, dtype=np.float64)
    length = int(previous.size + current.size - 1)
    logarithm = max(1, int(math.ceil(math.log2(length + 1))))
    budget = int(length * logarithm * logarithm)
    pair_work = int(previous.size * current.size)
    output = np.empty(current.size, dtype=np.float64)
    start = 0
    while start < current.size:
        remaining = int(current.size - start)
        width = 3 if remaining == 3 else min(2, remaining)
        stop = start + width
        block = current[start:stop]
        realized = float(degree_value) + previous[:, None] - block[None, :]
        transition = (
            forward[:, None]
            - (realized - float(target_value)) ** 2 * float(quadratic)
        )
        maximum = np.max(transition, axis=0, keepdims=True)
        output[start:stop] = np.squeeze(
            maximum
            + np.log(
                np.sum(np.exp(transition - maximum), axis=0, keepdims=True)
            ),
            axis=0,
        )
        start = stop
    return output, pair_work, budget


def build_guarded_gaussian_messages(
    path,
    tape: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    quadratic_coefficient: float,
    structural_temperature: float = 1.0,
) -> PackedGaussianMessages:
    """Exact small-block recurrence with a linear-log-squared work certificate."""

    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    sigma = float(degree_noise_std)
    coefficient = float(quadratic_coefficient)
    temperature = float(structural_temperature)
    if tape.bundle_count == 0:
        return build_gaussian_messages(
            order,
            tape,
            degree,
            target,
            sigma,
            quadratic_coefficient=coefficient,
            structural_temperature=temperature,
        )
    offsets = np.asarray(tape.state_offsets, dtype=np.int64)

    def view(values, index):
        return np.asarray(
            values[int(offsets[index]) : int(offsets[index + 1])]
        )

    quadratic = coefficient / (2.0 * sigma * sigma)
    forward_flat = np.empty(tape.state_value.size, dtype=np.float64)
    states = tuple(view(tape.state_value, index) for index in range(tape.bundle_count))
    gains = tuple(view(tape.structural_gain, index) for index in range(tape.bundle_count))
    widths = np.diff(offsets)
    pair_terms = widths[:-1] * widths[1:]
    length_terms = widths[:-1] + widths[1:] - 1
    logarithms = np.maximum(
        1, np.ceil(np.log2(length_terms + 1)).astype(np.int64)
    )
    pair_work = int(np.sum(pair_terms, dtype=np.int64))
    budget = int(np.sum(length_terms * logarithms * logarithms, dtype=np.int64))
    if pair_work > budget:
        raise ArithmeticError(
            "exact chain is outside its global linear-log-squared certificate"
        )
    first = int(order[0])
    forward = (
        np.asarray(gains[0], dtype=np.float64) / temperature
        - (degree[first] - states[0] - target[first]) ** 2 * quadratic
    )
    forward_flat[int(offsets[0]) : int(offsets[1])] = forward
    for index in range(1, tape.bundle_count):
        node = int(order[index])
        transformed, _, _ = _guarded_exact_transition(
            states[index - 1],
            states[index],
            forward,
            degree[node],
            target[node],
            quadratic,
        )
        forward = np.asarray(gains[index], dtype=np.float64) / temperature + transformed
        forward_flat[int(offsets[index]) : int(offsets[index + 1])] = forward
    last = int(order[-1])
    terminal = (
        forward
        - (degree[last] + np.asarray(states[-1], dtype=np.float64) - target[last]) ** 2
        * quadratic
    )
    log_normalizer = _column_logsumexp(terminal)
    if not math.isfinite(log_normalizer):
        raise ArithmeticError("guarded finite-chain normalizer is nonfinite")
    return PackedGaussianMessages(
        state_offsets=np.ascontiguousarray(offsets, dtype=np.int64),
        state_value=np.ascontiguousarray(tape.state_value, dtype=np.int64),
        structural_gain=np.ascontiguousarray(tape.structural_gain, dtype=np.float64),
        forward_log_message=np.ascontiguousarray(forward_flat, dtype=np.float64),
        log_normalizer=float(log_normalizer),
        exact_pair_work=int(pair_work),
        linear_log_squared_budget=int(budget),
        statistics={
            "transition_count": max(tape.bundle_count - 1, 0),
            "exact_pair_work": int(pair_work),
            "linear_log_squared_budget": int(budget),
            "pair_work_within_certificate": bool(pair_work <= budget),
            "complete_state_pair_matrix_materialized": False,
            "complete_dyad_enumeration": False,
            "complete_wedge_enumeration": False,
            "dense_n_by_n_state": False,
            "guarded_small_block_reduction": True,
        },
    )


def _draw_log_categorical(log_weight, rng) -> int:
    values = np.asarray(log_weight, dtype=np.float64)
    maximum = float(np.max(values))
    probability = np.exp(values - maximum)
    probability /= float(np.sum(probability))
    return int(rng.choice(values.size, p=probability))


def sample_backward(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    quadratic_coefficient: float,
    structural_temperature: float,
    sample_seed: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    count = int(messages.state_offsets.size - 1)
    if count == 0:
        return np.zeros(0, dtype=np.int64), {
            "sampled_signed_flow": 0,
            "sampled_absolute_flow": 0,
            "sampled_nonzero_bundle_count": 0,
            "selected_state_log_probability": 0.0,
        }
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    coefficient = float(quadratic_coefficient)
    temperature = float(structural_temperature)
    inverse_two_variance = coefficient / (2.0 * float(degree_noise_std) ** 2)
    rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence(int(sample_seed)))
    )
    state = np.zeros(count, dtype=np.int64)

    def view(values, index):
        start = int(messages.state_offsets[index])
        stop = int(messages.state_offsets[index + 1])
        return np.asarray(values[start:stop])

    last_state = view(messages.state_value, count - 1)
    last_forward = view(messages.forward_log_message, count - 1)
    last_node = int(order[-1])
    terminal = (
        last_forward
        - (degree[last_node] + last_state - target[last_node]) ** 2
        * inverse_two_variance
    )
    selected = _draw_log_categorical(terminal, rng)
    state[-1] = int(last_state[selected])
    for index in range(count - 1, 0, -1):
        node = int(order[index])
        previous = view(messages.state_value, index - 1)
        previous_forward = view(messages.forward_log_message, index - 1)
        current = int(state[index])
        log_weight = (
            previous_forward
            - (degree[node] + previous - current - target[node]) ** 2
            * inverse_two_variance
        )
        selected = _draw_log_categorical(log_weight, rng)
        state[index - 1] = int(previous[selected])

    structural = 0.0
    for index, value in enumerate(state):
        support = view(messages.state_value, index)
        location = np.flatnonzero(support == int(value))
        structural += float(view(messages.structural_gain, index)[int(location[0])])
    realized = degree.copy()
    for index, value in enumerate(state):
        realized[int(order[index])] -= int(value)
        realized[int(order[index + 1])] += int(value)
    energy = float(np.sum((realized - target) ** 2))
    unnormalized = float(
        structural / temperature
        - coefficient * energy / (2.0 * float(degree_noise_std) ** 2)
    )
    return state, {
        "sampled_signed_flow": int(np.sum(state)),
        "sampled_absolute_flow": int(np.sum(np.abs(state))),
        "sampled_nonzero_bundle_count": int(np.count_nonzero(state)),
        "selected_state_log_probability": float(unnormalized - messages.log_normalizer),
    }


__all__ = (
    "PackedGaussianMessages",
    "build_gaussian_messages",
    "build_guarded_gaussian_messages",
    "sample_backward",
)
