"""Checkpoint-measurable carrier selection and finite-chain information transport.

This module is the mathematical core that sits between the packed sufficient
statistics and the final backward draw.  It contains no graph inspection and no
historical implementation dispatch: every input is either public or already in
the released checkpoint / packed finite-chain law.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape
from .conditional_transport_native import (
    packed_chain_marginals,
    packed_chain_marginals_sufficient,
    packed_entropic_projection,
    packed_on_demand_entropic_projection,
)
from .finite_chain import PackedGaussianMessages, build_gaussian_messages


# Frozen domain separation of the public carrier stream.  The bytes are kept in
# hexadecimal form so the production API remains semantic while the published
# random stream stays exactly reproducible.
_CARRIER_STREAM_DOMAIN = bytes.fromhex(
    "763138392d73796d6d65747269632d6c6174656e742d636172726965727c"
)


@dataclass(frozen=True)
class CarrierDecision:
    reliability: float
    horizontal_mass: float
    vertical_mass: float
    public_uniform: float
    selected_component: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class ChainMarginals:
    state_offsets: np.ndarray
    log_marginal_value: np.ndarray
    transition_offsets: np.ndarray
    log_transition_value: np.ndarray
    diagnostics: Mapping[str, object]

    def marginal_view(self, index: int) -> np.ndarray:
        start = int(self.state_offsets[index])
        stop = int(self.state_offsets[index + 1])
        return self.log_marginal_value[start:stop]

    def transition_view(self, index: int) -> np.ndarray:
        start = int(self.transition_offsets[index])
        stop = int(self.transition_offsets[index + 1])
        left_width = int(self.state_offsets[index + 1] - self.state_offsets[index])
        right_width = int(
            self.state_offsets[index + 2] - self.state_offsets[index + 1]
        )
        return self.log_transition_value[start:stop].reshape(
            (left_width, right_width)
        )

    @property
    def log_marginals(self) -> tuple[np.ndarray, ...]:
        return tuple(
            self.marginal_view(index)
            for index in range(int(self.state_offsets.size - 1))
        )

    @property
    def log_transitions(self) -> tuple[np.ndarray, ...]:
        return tuple(
            self.transition_view(index)
            for index in range(max(int(self.state_offsets.size - 2), 0))
        )


@dataclass(frozen=True)
class ChainTransitionSufficientState:
    state_offsets: np.ndarray
    state_value: np.ndarray
    structural_gain: np.ndarray
    backward_log_message: np.ndarray
    row_normalization_correction: np.ndarray
    path_degree: np.ndarray
    path_target: np.ndarray
    quadratic_coefficient: float
    transition_work: int
    maximum_marginal_error: float


@dataclass(frozen=True)
class TransportResult:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    calibrated_fractions: np.ndarray
    target_state_offsets: np.ndarray
    target_log_marginal_value: np.ndarray
    diagnostics: Mapping[str, object]

    @property
    def target_log_marginals(self) -> tuple[np.ndarray, ...]:
        return tuple(
            self.target_log_marginal_value[
                int(self.target_state_offsets[index]) : int(
                    self.target_state_offsets[index + 1]
                )
            ]
            for index in range(int(self.target_state_offsets.size - 1))
        )


def _logsumexp(values, axis=None):
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(array - maximum), axis=axis, keepdims=True)
    )
    return float(result.reshape(-1)[0]) if axis is None else np.squeeze(result, axis)


def _views(flat: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(flat[int(offsets[index]) : int(offsets[index + 1])])
        for index in range(int(offsets.size - 1))
    )


def gaussian_degree_reliability(continuous_noisy_degree, degree_noise_std: float):
    """Positive-part normal random-effects empirical-Bayes reliability."""

    continuous = np.asarray(continuous_noisy_degree, dtype=np.float64)
    sigma = float(degree_noise_std)
    centered = continuous - float(np.mean(continuous))
    observed_variance = float(np.mean(centered * centered))
    noise_variance = float(sigma * sigma)
    latent_variance = float(max(observed_variance - noise_variance, 0.0))
    denominator = float(latent_variance + noise_variance)
    reliability = float(latent_variance / denominator) if denominator > 0.0 else 0.0
    if not 0.0 <= reliability <= 1.0:
        raise ArithmeticError("Gaussian degree reliability left [0,1]")
    return reliability, {
        "degree_channel_node_count": int(continuous.size),
        "degree_channel_observed_mle_variance": observed_variance,
        "degree_channel_noise_variance": noise_variance,
        "degree_channel_positive_part_latent_variance": latent_variance,
        "degree_channel_reliability": reliability,
        "positive_part_boundary_active": bool(latent_variance == 0.0),
        "variance_estimator_denominator": int(continuous.size),
        "reliability_is_normal_random_effects_eb": True,
    }


def carrier_uniform(flow_seed_prefix: int, public_experiment_id: int) -> float:
    payload = (
        _CARRIER_STREAM_DOMAIN
        + str(int(flow_seed_prefix)).encode("ascii")
        + b"|"
        + str(int(public_experiment_id)).encode("ascii")
    )
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return float((integer + 0.5) / float(2**64))


def select_carrier_component(
    continuous_noisy_degree,
    degree_noise_std: float,
    *,
    flow_seed_prefix: int,
    public_experiment_id: int,
) -> CarrierDecision:
    reliability, diagnostics = gaussian_degree_reliability(
        continuous_noisy_degree, degree_noise_std
    )
    horizontal = float(1.0 - reliability)
    vertical = float(reliability)
    uniform = carrier_uniform(flow_seed_prefix, public_experiment_id)
    selected = "vertical" if uniform < vertical else "horizontal"
    return CarrierDecision(
        reliability=reliability,
        horizontal_mass=horizontal,
        vertical_mass=vertical,
        public_uniform=uniform,
        selected_component=selected,
        diagnostics={
            **dict(diagnostics),
            "component_masses": (horizontal, vertical),
            "component_mass_sum": float(horizontal + vertical),
            "public_carrier_uniform": uniform,
            "selected_component": selected,
            "selection_uses_only_checkpoint_and_public_seed": True,
            "selection_uses_graph_or_metric": False,
        },
    )


def chain_log_marginals_and_transitions(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    quadratic_coefficient: float,
    *,
    materialize_transitions: bool = True,
) -> ChainMarginals:
    offsets = np.ascontiguousarray(messages.state_offsets, dtype=np.int64)
    state_value = np.ascontiguousarray(messages.state_value, dtype=np.int64)
    gain = np.ascontiguousarray(messages.structural_gain, dtype=np.float64)
    forward = np.ascontiguousarray(messages.forward_log_message, dtype=np.float64)
    position_count = int(offsets.size - 1)
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    quadratic = float(quadratic_coefficient) / (2.0 * float(degree_noise_std) ** 2)
    (
        marginals,
        transition_offsets,
        transitions,
        transition_work,
        maximum_marginal_error,
        maximum_row_error,
    ) = packed_chain_marginals(
        offsets,
        state_value,
        gain,
        forward,
        degree[order],
        target[order],
        quadratic,
        float(messages.log_normalizer),
        materialize_transitions=bool(materialize_transitions),
    )
    return ChainMarginals(
        state_offsets=offsets,
        log_marginal_value=np.ascontiguousarray(marginals),
        transition_offsets=transition_offsets,
        log_transition_value=np.ascontiguousarray(transitions),
        diagnostics={
            "chain_state_position_count": position_count,
            "chain_transition_work": int(transition_work),
            "maximum_log_marginal_normalization_error": maximum_marginal_error,
            "maximum_log_transition_row_error": maximum_row_error,
            "identical_marginal_normalizers_reused": True,
            "native_packed_chain_marginals": True,
            "transitions_materialized": bool(materialize_transitions),
            "persistent_python_ragged_view_count": 0,
        },
    )


def _chain_log_marginals_with_transition_sufficient_state(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    quadratic_coefficient: float,
):
    offsets = np.ascontiguousarray(messages.state_offsets, dtype=np.int64)
    state_value = np.ascontiguousarray(messages.state_value, dtype=np.int64)
    gain = np.ascontiguousarray(messages.structural_gain, dtype=np.float64)
    forward = np.ascontiguousarray(messages.forward_log_message, dtype=np.float64)
    position_count = int(offsets.size - 1)
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    ordered_degree = np.ascontiguousarray(degree[order], dtype=np.float64)
    ordered_target = np.ascontiguousarray(target[order], dtype=np.float64)
    quadratic = float(quadratic_coefficient) / (2.0 * float(degree_noise_std) ** 2)
    (
        marginals,
        backward,
        row_correction,
        transition_work,
        maximum_marginal_error,
    ) = packed_chain_marginals_sufficient(
        offsets,
        state_value,
        gain,
        forward,
        ordered_degree,
        ordered_target,
        quadratic,
        float(messages.log_normalizer),
    )
    marginal_law = ChainMarginals(
        state_offsets=offsets,
        log_marginal_value=np.ascontiguousarray(marginals),
        transition_offsets=np.zeros(position_count, dtype=np.int64),
        log_transition_value=np.empty(0, dtype=np.float64),
        diagnostics={
            "chain_state_position_count": position_count,
            "chain_transition_work": int(transition_work),
            "maximum_log_marginal_normalization_error": maximum_marginal_error,
            "native_packed_chain_marginals": True,
            "transitions_materialized": False,
            "exact_transition_sufficient_state": True,
            "persistent_python_ragged_view_count": 0,
        },
    )
    sufficient_state = ChainTransitionSufficientState(
        state_offsets=offsets,
        state_value=state_value,
        structural_gain=gain,
        backward_log_message=np.ascontiguousarray(backward),
        row_normalization_correction=np.ascontiguousarray(row_correction),
        path_degree=ordered_degree,
        path_target=ordered_target,
        quadratic_coefficient=quadratic,
        transition_work=int(transition_work),
        maximum_marginal_error=float(maximum_marginal_error),
    )
    return marginal_law, sufficient_state


def fisher_rao_probability_geodesic(
    log_left,
    log_right,
    *,
    fraction: float,
    endpoint_geometry=None,
):
    left = np.asarray(log_left, dtype=np.float64)
    right = np.asarray(log_right, dtype=np.float64)
    tau = float(fraction)
    if endpoint_geometry is None:
        endpoint_affinity = float(
            math.exp(float(_logsumexp(0.5 * (left + right))))
        )
        endpoint_affinity = float(min(max(endpoint_affinity, 0.0), 1.0))
        endpoint_distance = float(2.0 * math.acos(endpoint_affinity))
    else:
        endpoint_affinity = float(endpoint_geometry[0])
        endpoint_distance = float(endpoint_geometry[1])
    if tau == 0.0:
        output = left.copy()
    elif tau == 1.0:
        output = right.copy()
    else:
        affinity = endpoint_affinity
        tolerance = float(64.0 * np.finfo(np.float64).eps)
        if affinity < -tolerance or affinity > 1.0 + tolerance:
            raise ArithmeticError("Hellinger affinity left [0,1]")
        affinity = float(min(max(affinity, 0.0), 1.0))
        angle = float(math.acos(affinity))
        if angle <= tolerance:
            output = left.copy()
        else:
            denominator = float(math.sin(angle))
            left_weight = float(math.sin((1.0 - tau) * angle) / denominator)
            right_weight = float(math.sin(tau * angle) / denominator)
            log_root = np.logaddexp(
                math.log(left_weight) + 0.5 * left,
                math.log(right_weight) + 0.5 * right,
            )
            raw = 2.0 * log_root
            output = np.asarray(raw - float(_logsumexp(raw)), dtype=np.float64)
    affinity = float(min(max(endpoint_affinity, 0.0), 1.0))
    endpoint_distance = float(endpoint_distance)
    if tau == 0.0:
        left_distance, right_distance = 0.0, endpoint_distance
    elif tau == 1.0:
        left_distance, right_distance = endpoint_distance, 0.0
    else:
        left_affinity = float(math.exp(float(_logsumexp(0.5 * (left + output)))))
        right_affinity = float(math.exp(float(_logsumexp(0.5 * (right + output)))))
        left_distance = float(2.0 * math.acos(min(max(left_affinity, -1.0), 1.0)))
        right_distance = float(2.0 * math.acos(min(max(right_affinity, -1.0), 1.0)))
    error = max(
        abs(left_distance - tau * endpoint_distance),
        abs(right_distance - (1.0 - tau) * endpoint_distance),
    )
    return output, {
        "endpoint_hellinger_affinity": affinity,
        "endpoint_fisher_rao_distance": endpoint_distance,
        "left_geodesic_distance": left_distance,
        "right_geodesic_distance": right_distance,
        "prescribed_fraction_distance_error": error,
    }


def _fisher_rao_endpoint_geometry(log_left, log_right):
    """Return the exact endpoint geometry consumed by fraction calibration."""

    left = np.asarray(log_left, dtype=np.float64)
    right = np.asarray(log_right, dtype=np.float64)
    endpoint_affinity = float(math.exp(float(_logsumexp(0.5 * (left + right)))))
    endpoint_affinity = float(min(max(endpoint_affinity, 0.0), 1.0))
    endpoint_distance = float(2.0 * math.acos(endpoint_affinity))
    return endpoint_affinity, endpoint_distance


def _fisher_rao_geodesic_output(
    log_left,
    log_right,
    *,
    fraction: float,
    endpoint_geometry,
):
    """Return only the exact geodesic output consumed by target generation."""

    left = np.asarray(log_left, dtype=np.float64)
    right = np.asarray(log_right, dtype=np.float64)
    tau = float(fraction)
    endpoint_affinity = float(endpoint_geometry[0])
    if tau == 0.0:
        return left.copy()
    if tau == 1.0:
        return right.copy()
    affinity = endpoint_affinity
    tolerance = float(64.0 * np.finfo(np.float64).eps)
    if affinity < -tolerance or affinity > 1.0 + tolerance:
        raise ArithmeticError("Hellinger affinity left [0,1]")
    affinity = float(min(max(affinity, 0.0), 1.0))
    angle = float(math.acos(affinity))
    if angle <= tolerance:
        return left.copy()
    denominator = float(math.sin(angle))
    left_weight = float(math.sin((1.0 - tau) * angle) / denominator)
    right_weight = float(math.sin(tau * angle) / denominator)
    log_root = np.logaddexp(
        math.log(left_weight) + 0.5 * left,
        math.log(right_weight) + 0.5 * right,
    )
    raw = 2.0 * log_root
    return np.asarray(raw - float(_logsumexp(raw)), dtype=np.float64)


def chartwise_gaussian_posterior_fraction(state_variance: float, degree_noise_std: float):
    variance = float(state_variance)
    sigma = float(degree_noise_std)
    denominator = float(sigma * sigma + 2.0 * variance)
    reliability = float(2.0 * variance / denominator) if denominator > 0.0 else 0.0
    reliability = float(min(max(reliability, 0.0), 1.0))
    return float(0.5 * reliability), reliability


def baseline_component_box_projection(
    raw_fractions,
    endpoint_distances,
    global_fraction: float,
):
    """Unique weighted box projection preserving the global Fisher component."""

    raw = np.asarray(raw_fractions, dtype=np.float64)
    distances = np.asarray(endpoint_distances, dtype=np.float64)
    tau0 = float(global_fraction)
    weights = np.asarray(distances * distances, dtype=np.float64)
    positive_weight = weights > 0.0
    total_weight = float(np.sum(weights))
    target_balance = float(tau0 * total_weight)
    fractions = np.full(raw.shape, tau0, dtype=np.float64)
    dual = 0.0
    if total_weight > 0.0:
        lower, upper = -0.5, 0.5
        for _ in range(np.finfo(np.float64).nmant + 2):
            midpoint = float(0.5 * (lower + upper))
            values = np.clip(raw + midpoint, 0.0, 0.5)
            if float(np.sum(weights * values)) < target_balance:
                lower = midpoint
            else:
                upper = midpoint
        dual = float(0.5 * (lower + upper))
        fractions = np.clip(raw + dual, 0.0, 0.5)
        fractions[~positive_weight] = tau0
        active = positive_weight & (fractions > 0.0) & (fractions < 0.5)
        if np.any(active):
            residual = float(target_balance - np.sum(weights * fractions))
            dual += float(residual / np.sum(weights[active]))
            fractions = np.clip(raw + dual, 0.0, 0.5)
            fractions[~positive_weight] = tau0
    actual_balance = float(np.sum(weights * fractions))
    balance_error = float(abs(actual_balance - target_balance))
    scale = float(max(1.0, total_weight, abs(target_balance), abs(actual_balance)))
    backward_bound = float(
        np.finfo(np.float64).nmant**2 * np.finfo(np.float64).eps * scale
    )
    if balance_error > backward_bound:
        raise ArithmeticError("weighted baseline balance failed")
    shifted = raw + dual
    active = positive_weight & (fractions > 0.0) & (fractions < 0.5)
    lower_active = positive_weight & (fractions == 0.0)
    upper_active = positive_weight & (fractions == 0.5)
    kkt_error = float(
        max(
            float(np.max(np.abs(fractions[active] - shifted[active]), initial=0.0)),
            float(np.max(np.maximum(shifted[lower_active], 0.0), initial=0.0)),
            float(np.max(np.maximum(0.5 - shifted[upper_active], 0.0), initial=0.0)),
        )
    )
    baseline_norm = float(tau0 * tau0 * total_weight)
    displacement_norm = float(np.sum(weights * fractions * fractions))
    heterogeneity_norm = float(np.sum(weights * (fractions - tau0) ** 2))
    orthogonality_error = float(abs(tau0 * np.sum(weights * (fractions - tau0))))
    pythagorean_error = float(
        abs(displacement_norm - baseline_norm - heterogeneity_norm)
    )
    geometric_bound = float(
        np.finfo(np.float64).nmant**2
        * np.finfo(np.float64).eps
        * max(1.0, total_weight, baseline_norm, displacement_norm, heterogeneity_norm)
    )
    if (
        kkt_error > backward_bound
        or orthogonality_error > geometric_bound
        or pythagorean_error > geometric_bound
    ):
        raise ArithmeticError("box projection KKT or Fisher certificate failed")
    return fractions, {
        "box_projection_dual": dual,
        "weighted_balance_target": target_balance,
        "weighted_balance_actual": actual_balance,
        "weighted_balance_error": balance_error,
        "weighted_balance_backward_bound": backward_bound,
        "kkt_error": kkt_error,
        "kkt_backward_bound": backward_bound,
        "convex_projection_objective": float(np.sum(weights * (fractions - raw) ** 2)),
        "baseline_norm_squared": baseline_norm,
        "displacement_norm_squared": displacement_norm,
        "orthogonal_heterogeneity_norm_squared": heterogeneity_norm,
        "baseline_orthogonality_error": orthogonality_error,
        "fisher_pythagorean_error": pythagorean_error,
        "fisher_geometric_backward_bound": geometric_bound,
        "canonical_zero_distance_fraction": tau0,
        "zero_distance_coordinate_count": int(np.count_nonzero(~positive_weight)),
    }


def entropic_transition_projection(
    log_source_marginal,
    log_target_marginal,
    log_reference_transition,
):
    source = np.asarray(log_source_marginal, dtype=np.float64)
    target = np.asarray(log_target_marginal, dtype=np.float64)
    transition = np.asarray(log_reference_transition, dtype=np.float64)
    base = source[:, None] + transition
    right = np.zeros(target.size, dtype=np.float64)
    machine = np.finfo(np.float64)
    maximum_iterations = int(machine.nmant * machine.nmant)
    operation_scale = float(
        max(
            1.0,
            source.size,
            target.size,
            float(np.max(np.abs(source), initial=0.0)),
            float(np.max(np.abs(target), initial=0.0)),
        )
    )
    bound = float(machine.nmant * machine.nmant * machine.eps * operation_scale)
    error = math.inf
    left = np.zeros(source.size, dtype=np.float64)
    for iteration in range(1, maximum_iterations + 1):
        left = source - _logsumexp(base + right[None, :], axis=1)
        right = target - _logsumexp(base + left[:, None], axis=0)
        pair = base + left[:, None] + right[None, :]
        row_error = float(
            np.max(np.abs(_logsumexp(pair, axis=1) - source), initial=0.0)
        )
        column_error = float(
            np.max(np.abs(_logsumexp(pair, axis=0) - target), initial=0.0)
        )
        error = max(row_error, column_error)
        if error <= bound:
            break
    else:
        raise ArithmeticError("adjacent entropic projection did not converge")
    return left, right, {
        "iterations": int(iteration),
        "maximum_log_marginal_error": error,
        "backward_error_bound": bound,
    }


def baseline_preserving_information_transport(
    path,
    tape: PackedRelocationTape,
    reference_messages: PackedGaussianMessages,
    reference_coefficient: float,
    endpoint_messages: PackedGaussianMessages,
    endpoint_coefficient: float,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    degree_reliability: float,
) -> TransportResult:
    """Transport chart marginals while preserving the global Fisher component.

    The reference conditional supplies the Markov transition geometry.  The
    second endpoint supplies the structural direction.  Gaussian posterior
    reliability allocates local fractions; the weighted box projection retains
    the global baseline component; and adjacent Sinkhorn projections give the
    unique path-space information projection on the same packed support.
    """

    reference, transition_state = _chain_log_marginals_with_transition_sufficient_state(
        path,
        reference_messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        reference_coefficient,
    )
    endpoint = chain_log_marginals_and_transitions(
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
    raw_fractions = np.empty(position_count, dtype=np.float64)
    chart_reliabilities = np.empty(position_count, dtype=np.float64)
    variances = np.empty(position_count, dtype=np.float64)
    distances = np.empty(position_count, dtype=np.float64)
    endpoint_affinities = np.empty(position_count, dtype=np.float64)
    for index in range(position_count):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        states = state_value[start:stop]
        left = reference.marginal_view(index)
        right = endpoint.marginal_view(index)
        probability = np.exp(np.asarray(left, dtype=np.float64))
        probability /= float(np.sum(probability))
        mean = float(probability @ states)
        variance = float(probability @ ((states - mean) ** 2))
        raw_fraction, chart_reliability = chartwise_gaussian_posterior_fraction(
            variance, degree_noise_std
        )
        endpoint_affinity, endpoint_distance = _fisher_rao_endpoint_geometry(
            left, right
        )
        endpoint_affinities[index] = endpoint_affinity
        raw_fractions[index] = raw_fraction
        chart_reliabilities[index] = chart_reliability
        variances[index] = variance
        distances[index] = endpoint_distance
    global_fraction = float(0.5 * float(degree_reliability))
    fractions, calibration = baseline_component_box_projection(
        raw_fractions, distances, global_fraction
    )
    targets = np.empty(state_value.size, dtype=np.float64)
    for index in range(position_count):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        targets[start:stop] = _fisher_rao_geodesic_output(
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
    ) = packed_on_demand_entropic_projection(
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
        first_stop = int(offsets[1])
        potentials[:first_stop] = (
            targets[:first_stop]
            - reference.log_marginal_value[:first_stop]
            + left_scales[:first_stop]
        )
        for index in range(1, position_count - 1):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            potentials[start:stop] = (
                right_scales[start:stop] + left_scales[start:stop]
            )
        last_start = int(offsets[-2])
        potentials[last_start:] = right_scales[last_start:]
    centered = np.empty_like(potentials)
    for index in range(position_count):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        values = potentials[start:stop]
        centered[start:stop] = np.asarray(
            values - np.max(values), dtype=np.float64
        )
    gain = np.ascontiguousarray(
        reference_messages.structural_gain + centered,
        dtype=np.float64,
    )
    transported_tape = replace(tape, structural_gain=gain)
    messages = build_gaussian_messages(
        path,
        transported_tape,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=float(reference_coefficient),
        structural_temperature=1.0,
    )
    return TransportResult(
        messages=messages,
        quadratic_coefficient=float(reference_coefficient),
        calibrated_fractions=np.ascontiguousarray(fractions, dtype=np.float64),
        target_state_offsets=offsets,
        target_log_marginal_value=np.ascontiguousarray(targets),
        diagnostics={
            "shared_support": True,
            "support_position_count": position_count,
            "support_state_count": int(reference_messages.state_value.size),
            "global_degree_reliability": float(degree_reliability),
            "global_fisher_fraction": global_fraction,
            "chartwise_variance_minimum": float(
                np.min(variances) if variances.size else 0.0
            ),
            "chartwise_variance_maximum": float(
                np.max(variances) if variances.size else 0.0
            ),
            "chartwise_reliability_minimum": float(
                np.min(chart_reliabilities) if chart_reliabilities.size else 0.0
            ),
            "chartwise_reliability_maximum": float(
                np.max(chart_reliabilities) if chart_reliabilities.size else 0.0
            ),
            "calibrated_fraction_minimum": float(
                min((float(value) for value in fractions), default=0.0)
            ),
            "calibrated_fraction_maximum": float(
                max((float(value) for value in fractions), default=0.0)
            ),
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
        },
    )


__all__ = (
    "CarrierDecision",
    "ChainMarginals",
    "TransportResult",
    "baseline_component_box_projection",
    "baseline_preserving_information_transport",
    "carrier_uniform",
    "chain_log_marginals_and_transitions",
    "chartwise_gaussian_posterior_fraction",
    "entropic_transition_projection",
    "fisher_rao_probability_geodesic",
    "gaussian_degree_reliability",
    "select_carrier_component",
)
