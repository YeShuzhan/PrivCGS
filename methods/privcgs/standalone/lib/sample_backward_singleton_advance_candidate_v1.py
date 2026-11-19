"""Exact backward sampler using PCG jump-ahead over deterministic singleton charts."""

from __future__ import annotations

import numpy as np


LAST_DIAGNOSTICS = None
_PRODUCTION_NAMESPACE = None


def configure_production(reference_function):
    global _PRODUCTION_NAMESPACE
    _PRODUCTION_NAMESPACE = reference_function.__globals__


def _production_namespace():
    if _PRODUCTION_NAMESPACE is not None:
        return _PRODUCTION_NAMESPACE
    raise RuntimeError('Standalone runtime must configure the frozen sampler namespace')


def sample_backward(
    path,
    messages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    quadratic_coefficient: float,
    structural_temperature: float,
    sample_seed: int,
):
    """Return the frozen draw while jumping exactly over one-point categorical laws."""

    global LAST_DIAGNOSTICS
    production = _production_namespace()
    count = int(messages.state_offsets.size - 1)
    if count == 0:
        LAST_DIAGNOSTICS = {
            "singleton_chart_count": 0,
            "non_singleton_chart_count": 0,
            "pcg_advance_count": 0,
            "explicit_categorical_count": 0,
        }
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
    offsets = np.asarray(messages.state_offsets, dtype=np.int64)
    widths = np.diff(offsets)
    singleton = widths == 1
    non_singleton_descending = np.flatnonzero(~singleton)[::-1]
    state = np.zeros(count, dtype=np.int64)
    selected_location = np.zeros(count, dtype=np.int64)
    singleton_positions = np.flatnonzero(singleton)
    if singleton_positions.size:
        singleton_flat = offsets[singleton_positions]
        state[singleton_positions] = messages.state_value[singleton_flat]

    def view(values, index):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        return np.asarray(values[start:stop])

    next_position = count - 1
    pcg_advance_count = 0
    explicit_categorical_count = 0
    for raw_position in non_singleton_descending:
        position = int(raw_position)
        deterministic_count = int(next_position - position)
        if deterministic_count:
            rng.bit_generator.advance(deterministic_count)
            pcg_advance_count += deterministic_count
        support = view(messages.state_value, position)
        if position == count - 1:
            forward = view(messages.forward_log_message, position)
            node = int(order[-1])
            log_weight = (
                forward
                - (degree[node] + support - target[node]) ** 2
                * inverse_two_variance
            )
        else:
            node = int(order[position + 1])
            forward = view(messages.forward_log_message, position)
            current = int(state[position + 1])
            log_weight = (
                forward
                - (degree[node] + support - current - target[node]) ** 2
                * inverse_two_variance
            )
        selected = production["_draw_log_categorical"](log_weight, rng)
        selected_location[position] = int(selected)
        state[position] = int(support[selected])
        explicit_categorical_count += 1
        next_position = position - 1
    if next_position >= 0:
        deterministic_count = int(next_position + 1)
        rng.bit_generator.advance(deterministic_count)
        pcg_advance_count += deterministic_count

    selected_flat = offsets[:-1] + selected_location
    selected_gain = np.asarray(messages.structural_gain)[selected_flat]
    structural = 0.0
    for raw_index in np.flatnonzero(selected_gain != 0.0):
        structural += float(selected_gain[int(raw_index)])
    realized = degree.copy()
    for raw_index in np.flatnonzero(state != 0):
        index = int(raw_index)
        value = int(state[index])
        realized[int(order[index])] -= value
        realized[int(order[index + 1])] += value
    energy = float(np.sum((realized - target) ** 2))
    unnormalized = float(
        structural / temperature
        - coefficient * energy / (2.0 * float(degree_noise_std) ** 2)
    )
    LAST_DIAGNOSTICS = {
        "singleton_chart_count": int(singleton_positions.size),
        "non_singleton_chart_count": int(non_singleton_descending.size),
        "pcg_advance_count": int(pcg_advance_count),
        "explicit_categorical_count": int(explicit_categorical_count),
        "total_rng_transition_count": int(pcg_advance_count + explicit_categorical_count),
        "final_bit_generator_state": rng.bit_generator.state,
    }
    return state, {
        "sampled_signed_flow": int(np.sum(state)),
        "sampled_absolute_flow": int(np.sum(np.abs(state))),
        "sampled_nonzero_bundle_count": int(np.count_nonzero(state)),
        "selected_state_log_probability": float(unnormalized - messages.log_normalizer),
    }


__all__ = ("LAST_DIAGNOSTICS", "configure_production", "sample_backward")
