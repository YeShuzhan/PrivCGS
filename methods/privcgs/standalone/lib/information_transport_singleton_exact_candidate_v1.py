"""Exact transport backend that removes degenerate singleton-chart Python work."""

from __future__ import annotations

from dataclasses import replace

import numpy as np


LAST_DIAGNOSTICS = None
_PRODUCTION_NAMESPACE = None


def configure_production(reference_function):
    """Freeze the original production namespace before a direct replacement."""

    global _PRODUCTION_NAMESPACE
    _PRODUCTION_NAMESPACE = reference_function.__globals__


def _production_namespace():
    if _PRODUCTION_NAMESPACE is not None:
        return _PRODUCTION_NAMESPACE
    raise RuntimeError('Standalone runtime must configure the frozen transport namespace')


def baseline_preserving_information_transport(
    path,
    tape,
    reference_messages,
    reference_coefficient: float,
    endpoint_messages,
    endpoint_coefficient: float,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    degree_reliability: float,
):
    """Return the frozen result with exact singleton chart identities applied in bulk."""

    global LAST_DIAGNOSTICS
    production = _production_namespace()
    reference, transition_state = production[
        "_chain_log_marginals_with_transition_sufficient_state"
    ](
        path,
        reference_messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        reference_coefficient,
    )
    endpoint = production["chain_log_marginals_and_transitions"](
        path,
        endpoint_messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        endpoint_coefficient,
        materialize_transitions=False,
    )
    offsets = np.ascontiguousarray(reference_messages.state_offsets, dtype=np.int64)
    state_value = np.ascontiguousarray(reference_messages.state_value, dtype=np.int64)
    position_count = int(offsets.size - 1)
    widths = np.diff(offsets)
    non_singleton = np.flatnonzero(widths != 1)
    singleton = np.flatnonzero(widths == 1)

    # For a one-state chart both normalized endpoint laws are exactly [0.0].
    # Its variance and Fisher distance are exactly zero and its affinity is one.
    raw_fractions = np.zeros(position_count, dtype=np.float64)
    chart_reliabilities = np.zeros(position_count, dtype=np.float64)
    variances = np.zeros(position_count, dtype=np.float64)
    distances = np.zeros(position_count, dtype=np.float64)
    endpoint_affinities = np.ones(position_count, dtype=np.float64)
    for raw_index in non_singleton:
        index = int(raw_index)
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        states = state_value[start:stop]
        left = reference.marginal_view(index)
        right = endpoint.marginal_view(index)
        probability = np.exp(np.asarray(left, dtype=np.float64))
        probability /= float(np.sum(probability))
        mean = float(probability @ states)
        variance = float(probability @ ((states - mean) ** 2))
        raw_fraction, chart_reliability = production[
            "chartwise_gaussian_posterior_fraction"
        ](variance, degree_noise_std)
        endpoint_affinity, endpoint_distance = production[
            "_fisher_rao_endpoint_geometry"
        ](left, right)
        endpoint_affinities[index] = endpoint_affinity
        raw_fractions[index] = raw_fraction
        chart_reliabilities[index] = chart_reliability
        variances[index] = variance
        distances[index] = endpoint_distance

    global_fraction = float(0.5 * float(degree_reliability))
    fractions, calibration = production["baseline_component_box_projection"](
        raw_fractions, distances, global_fraction
    )
    targets = np.empty(state_value.size, dtype=np.float64)
    singleton_state = offsets[singleton]
    targets[singleton_state] = reference.log_marginal_value[singleton_state]
    for raw_index in non_singleton:
        index = int(raw_index)
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        targets[start:stop] = production["_fisher_rao_geodesic_output"](
            reference.marginal_view(index),
            endpoint.marginal_view(index),
            fraction=float(fractions[index]),
            endpoint_geometry=(endpoint_affinities[index], distances[index]),
        )
    (
        left_scales,
        right_scales,
        total_iterations,
        maximum_error,
        maximum_bound,
        maximum_row_error,
    ) = production["packed_on_demand_entropic_projection"](
        transition_state.state_offsets,
        transition_state.state_value,
        transition_state.structural_gain,
        transition_state.backward_log_message,
        transition_state.row_normalization_correction,
        transition_state.path_degree,
        transition_state.path_target,
        transition_state.quadratic_coefficient,
        targets,
    )
    potentials = np.empty(state_value.size, dtype=np.float64)
    if position_count == 1:
        potentials[:] = targets - reference.log_marginal_value
    else:
        np.add(right_scales, left_scales, out=potentials)
        first_stop = int(offsets[1])
        potentials[:first_stop] = (
            targets[:first_stop]
            - reference.log_marginal_value[:first_stop]
            + left_scales[:first_stop]
        )
        last_start = int(offsets[-2])
        potentials[last_start:] = right_scales[last_start:]
    centered = np.zeros_like(potentials)
    for raw_index in non_singleton:
        index = int(raw_index)
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        values = potentials[start:stop]
        centered[start:stop] = np.asarray(values - np.max(values), dtype=np.float64)
    gain = np.ascontiguousarray(
        reference_messages.structural_gain + centered,
        dtype=np.float64,
    )
    transported_tape = replace(tape, structural_gain=gain)
    messages = production["build_gaussian_messages"](
        path,
        transported_tape,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=float(reference_coefficient),
        structural_temperature=1.0,
    )
    diagnostics = {
        "shared_support": True,
        "support_position_count": position_count,
        "support_state_count": int(reference_messages.state_value.size),
        "global_degree_reliability": float(degree_reliability),
        "global_fisher_fraction": global_fraction,
        "chartwise_variance_minimum": float(np.min(variances) if variances.size else 0.0),
        "chartwise_variance_maximum": float(np.max(variances) if variances.size else 0.0),
        "chartwise_reliability_minimum": float(
            np.min(chart_reliabilities) if chart_reliabilities.size else 0.0
        ),
        "chartwise_reliability_maximum": float(
            np.max(chart_reliabilities) if chart_reliabilities.size else 0.0
        ),
        "calibrated_fraction_minimum": float(np.min(fractions) if fractions.size else 0.0),
        "calibrated_fraction_maximum": float(np.max(fractions) if fractions.size else 0.0),
        "calibrated_fraction_mean": float(np.mean(fractions) if fractions.size else 0.0),
        "sinkhorn_total_iterations": int(total_iterations),
        "sinkhorn_maximum_log_marginal_error": maximum_error,
        "sinkhorn_maximum_backward_error_bound": maximum_bound,
        "maximum_log_transition_row_error": maximum_row_error,
        "native_packed_entropic_projection": True,
        "exact_on_demand_transition_projection": True,
        "transition_sufficient_state_bytes": int(
            transition_state.backward_log_message.nbytes
            + transition_state.row_normalization_correction.nbytes
        ),
        "new_fitted_parameter_count": 0,
        "persistent_python_ragged_view_count": 0,
        "complete_dyad_enumeration": False,
        "complete_wedge_enumeration": False,
        **dict(calibration),
    }
    LAST_DIAGNOSTICS = {
        "singleton_chart_count": int(singleton.size),
        "non_singleton_chart_count": int(non_singleton.size),
    }
    return production["TransportResult"](
        messages=messages,
        quadratic_coefficient=float(reference_coefficient),
        calibrated_fractions=np.ascontiguousarray(fractions, dtype=np.float64),
        target_state_offsets=offsets,
        target_log_marginal_value=np.ascontiguousarray(targets),
        diagnostics=diagnostics,
    )


__all__ = (
    "LAST_DIAGNOSTICS",
    "baseline_preserving_information_transport",
    "configure_production",
)
