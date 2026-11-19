"""Two intrinsic endpoint laws on one normalized sparse finite-chain carrier.

The endpoints share a parent distribution.  One protects the configuration
innovation against structural, phase and Gaussian degree nuisance directions;
the other takes the equal-information Fisher midpoint between configuration
and closed/total-triplet quotient directions.  Both are normalized finite laws,
not post-generation graph repairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .chain_moments import BivariateChainMoments, bivariate_chain_moments
from .contracts import PackedRelocationTape
from .degree_chain_native import scalar_chain_moments
from .finite_chain import PackedGaussianMessages, build_gaussian_messages
from .palm_coordinates import PalmScoreColumns


@dataclass(frozen=True)
class EndpointLaw:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    covariance: np.ndarray
    natural_parameter: np.ndarray
    amplitude: float
    diagnostics: Mapping[str, object]


def fisher_rao_bernoulli_midpoint(left: float, right: float) -> float:
    """Intrinsic midpoint of two Bernoulli reliabilities."""

    left_value = float(left)
    right_value = float(right)
    angle = 0.5 * (
        math.asin(math.sqrt(left_value))
        + math.asin(math.sqrt(right_value))
    )
    return float(math.sin(angle) ** 2)


def configuration_score_reliability(columns: PalmScoreColumns) -> float:
    """Positive-part empirical-Bayes reliability of the Palm configuration score."""

    statistics = columns.statistics
    probability = float(statistics["configuration_probability"])
    scale = float(statistics["configuration_scale"])
    closed = float(statistics["source_closed_token_count"])
    total = float(statistics["source_active_token_count"])
    score = float((closed - probability * total) / scale) if scale > 0.0 else 0.0
    squared = float(score * score)
    latent = float(max(squared - 1.0, 0.0))
    return float(latent / squared) if squared > 0.0 else 0.0


def _views(flat, offsets):
    values = np.asarray(flat, dtype=np.float64)
    boundary = np.asarray(offsets, dtype=np.int64)
    return tuple(
        values[int(boundary[index]) : int(boundary[index + 1])]
        for index in range(int(boundary.size - 1))
    )


def _sum_columns(left, right):
    return np.ascontiguousarray(
        np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64),
        dtype=np.float64,
    )


def fisher_covariance(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    quadratic_coefficient: float,
    channels: Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[BivariateChainMoments, ...]]:
    """Exact scalar-semiring assembly of score/degree Fisher covariance."""

    packed = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in channels)

    expanded = list(packed)
    pairs = []
    for left in range(len(packed)):
        for right in range(left + 1, len(packed)):
            pairs.append((left, right))
            expanded.append(_sum_columns(packed[left], packed[right]))
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    coefficient = float(quadratic_coefficient)
    inverse_two_variance = 1.0 / (2.0 * float(degree_noise_std) ** 2)
    moments = scalar_chain_moments(
        messages.state_offsets,
        messages.state_value,
        messages.forward_log_message,
        messages.structural_gain,
        tuple(expanded),
        degree[order],
        target[order],
        inverse_two_variance,
        coefficient,
        messages.log_normalizer,
    )
    variances = np.asarray(moments["variances"], dtype=np.float64)
    degree_covariances = np.asarray(
        moments["degree_covariances"], dtype=np.float64
    )
    dimension = len(packed) + 1
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    covariance[np.arange(len(packed)), np.arange(len(packed))] = variances[: len(packed)]
    covariance[:-1, -1] = covariance[-1, :-1] = degree_covariances[: len(packed)]
    covariance[-1, -1] = float(moments["variance_degree"])
    for channel, (left, right) in enumerate(pairs, start=len(packed)):
        value = float((variances[channel] - variances[left] - variances[right]) / 2.0)
        covariance[left, right] = covariance[right, left] = value
    covariance = 0.5 * (covariance + covariance.T)
    from .centered_fisher import psd_valid, centered
    if not psd_valid(covariance):
        covariance = centered(path, messages, source_degree, target_endpoint_mass,
                              degree_noise_std, coefficient, packed)
    return np.ascontiguousarray(covariance), tuple()


def _tilt_parent(
    path,
    tape: PackedRelocationTape,
    parent: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    parent_coefficient: float,
    structural_additive,
    degree_parameter: float,
):
    additive = np.asarray(structural_additive, dtype=np.float64)
    gain = np.ascontiguousarray(parent.structural_gain + additive, dtype=np.float64)
    tilted_tape = replace(tape, structural_gain=gain)
    coefficient = float(float(parent_coefficient) + float(degree_parameter))
    messages = build_gaussian_messages(
        path,
        tilted_tape,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=coefficient,
        structural_temperature=1.0,
    )
    return messages, coefficient


def _efficient_unit_direction(covariance, target_index, nuisance_indices):
    matrix = np.asarray(covariance, dtype=np.float64)
    indices = np.asarray(nuisance_indices, dtype=np.int64)
    nuisance = matrix[np.ix_(indices, indices)]
    cross = matrix[indices, int(target_index)]
    singular = np.linalg.svd(nuisance, compute_uv=False)
    maximum = float(np.max(singular, initial=0.0))
    tolerance = float(
        np.finfo(np.float64).eps
        * max(int(nuisance.shape[0]), 1)
        * max(maximum, 1.0)
    )
    nuisance_rank = int(np.sum(singular > tolerance))
    inverse = np.linalg.pinv(
        nuisance,
        rcond=(
            tolerance / maximum
            if maximum > 0.0
            else np.finfo(np.float64).eps
        ),
    )
    beta = inverse @ cross
    residual = np.zeros(matrix.shape[0], dtype=np.float64)
    residual[int(target_index)] = 1.0
    residual[indices] = -beta
    variance = float(residual @ matrix @ residual)
    joint_indices = np.asarray(
        [int(target_index), *[int(value) for value in nuisance_indices]],
        dtype=np.int64,
    )
    joint = matrix[np.ix_(joint_indices, joint_indices)]
    joint_singular = np.linalg.svd(joint, compute_uv=False)
    joint_maximum = float(np.max(joint_singular, initial=0.0))
    joint_tolerance = float(
        np.finfo(np.float64).eps
        * max(int(joint.shape[0]), 1)
        * max(joint_maximum, 1.0)
    )
    joint_rank = int(np.sum(joint_singular > joint_tolerance))
    positive = bool(joint_rank > nuisance_rank and variance > joint_tolerance)
    if positive:
        scale = float(math.sqrt(variance))
        unit = residual / scale
    else:
        scale = 0.0
        unit = np.zeros(matrix.shape[0], dtype=np.float64)
    return {
        "nuisance_rank": nuisance_rank,
        "joint_rank": joint_rank,
        "positive": positive,
        "beta": beta,
        "residual": residual,
        "variance": variance if positive else 0.0,
        "scale": scale,
        "unit": unit,
        "unit_variance": float(unit @ matrix @ unit) if positive else 0.0,
    }


def phase_protected_configuration_endpoint(
    path,
    tape: PackedRelocationTape,
    parent: PackedGaussianMessages,
    parent_coefficient: float,
    columns: PalmScoreColumns,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    degree_reliability: float,
) -> EndpointLaw:
    """Configuration endpoint orthogonal to structure, phase and degree energy."""

    amplitude = fisher_rao_bernoulli_midpoint(
        float(degree_reliability), configuration_score_reliability(columns)
    )
    channels = (
        columns.configuration_increment_additive,
        parent.structural_gain,
        tape.structural_gain,
    )
    covariance, _ = fisher_covariance(
        path,
        parent,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        parent_coefficient,
        channels,
    )
    nuisance_indices = np.asarray([1, 2, 3], dtype=np.int64)
    nuisance = covariance[np.ix_(nuisance_indices, nuisance_indices)]
    cross = covariance[nuisance_indices, 0]
    singular = np.linalg.svd(nuisance, compute_uv=False)
    maximum = float(np.max(singular, initial=0.0))
    tolerance = float(
        np.finfo(np.float64).eps
        * max(int(nuisance.shape[0]), 3)
        * max(maximum, 1.0)
    )
    nuisance_rank = int(np.sum(singular > tolerance))
    inverse = np.linalg.pinv(
        nuisance,
        rcond=tolerance / maximum if maximum > 0.0 else np.finfo(np.float64).eps,
    )
    beta = inverse @ cross
    residual = np.asarray([1.0, -beta[0], -beta[1], -beta[2]], dtype=np.float64)
    residual_variance = float(residual @ covariance @ residual)
    joint_singular = np.linalg.svd(covariance, compute_uv=False)
    joint_maximum = float(np.max(joint_singular, initial=0.0))
    joint_tolerance = float(
        np.finfo(np.float64).eps
        * max(int(covariance.shape[0]), 4)
        * max(joint_maximum, 1.0)
    )
    joint_rank = int(np.sum(joint_singular > joint_tolerance))
    positive = bool(
        float(columns.statistics["configuration_scale"]) > 0.0
        and joint_rank > nuisance_rank
        and residual_variance > joint_tolerance
    )
    if positive and amplitude > 0.0:
        natural = -float(amplitude) * residual
        additive = (
            float(natural[0]) * channels[0]
            + float(natural[1]) * channels[1]
            + float(natural[2]) * channels[2]
        )
        messages, coefficient = _tilt_parent(
            path,
            tape,
            parent,
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
            parent_coefficient,
            additive,
            float(-natural[3]),
        )
    else:
        natural = np.zeros(4, dtype=np.float64)
        messages = parent
        coefficient = float(parent_coefficient)
    return EndpointLaw(
        messages=messages,
        quadratic_coefficient=coefficient,
        covariance=covariance,
        natural_parameter=np.ascontiguousarray(natural),
        amplitude=float(amplitude),
        diagnostics={
            "nuisance_rank": nuisance_rank,
            "joint_rank": joint_rank,
            "efficient_rank_positive": positive,
            "projected_variance": residual_variance if positive else 0.0,
            "projection_coefficients": beta.tolist(),
            "configuration_reliability": configuration_score_reliability(columns),
            "finite_chain_normalization": True,
            "new_fitted_parameter_count": 0,
        },
    )


def spherical_equal_information_endpoint(
    path,
    tape: PackedRelocationTape,
    parent: PackedGaussianMessages,
    parent_coefficient: float,
    columns: PalmScoreColumns,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    degree_reliability: float,
) -> EndpointLaw:
    """Fisher-spherical midpoint of configuration and triplet quotient scores."""

    amplitude = fisher_rao_bernoulli_midpoint(
        float(degree_reliability), configuration_score_reliability(columns)
    )
    channels = (
        columns.configuration_increment_additive,
        columns.closure_increment_additive,
        columns.wedge_increment_additive,
        parent.structural_gain,
    )
    covariance, _ = fisher_covariance(
        path,
        parent,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        parent_coefficient,
        channels,
    )
    configuration = _efficient_unit_direction(covariance, 0, (3, 4))
    quotient = _efficient_unit_direction(covariance, 1, (2, 3, 4))
    if configuration["positive"] and quotient["positive"]:
        raw_inner = float(configuration["unit"] @ covariance @ quotient["unit"])
        bound = float(math.sqrt(np.finfo(np.float64).eps))
        if raw_inner < -1.0 - bound or raw_inner > 1.0 + bound:
            raise ArithmeticError("endpoint Fisher inner product left [-1,1]")
        raw_inner = float(min(max(raw_inner, -1.0), 1.0))
        orientation = 1.0 if raw_inner >= 0.0 else -1.0
        aligned = orientation * quotient["unit"]
        denominator = float(math.sqrt(2.0 + 2.0 * abs(raw_inner)))
        midpoint = (configuration["unit"] + aligned) / denominator
    elif configuration["positive"]:
        raw_inner, orientation, denominator = 0.0, 1.0, 1.0
        midpoint = np.asarray(configuration["unit"], dtype=np.float64)
    elif quotient["positive"]:
        raw_inner, orientation, denominator = 0.0, 1.0, 1.0
        midpoint = np.asarray(quotient["unit"], dtype=np.float64)
    else:
        raw_inner, orientation, denominator = 0.0, 1.0, 0.0
        midpoint = np.zeros(5, dtype=np.float64)
    midpoint_variance = float(midpoint @ covariance @ midpoint)
    if midpoint_variance > 0.0 and amplitude > 0.0:
        natural = -float(amplitude) * midpoint
        additive = (
            float(natural[0]) * channels[0]
            + float(natural[1]) * channels[1]
            + float(natural[2]) * channels[2]
            + float(natural[3]) * channels[3]
        )
        messages, coefficient = _tilt_parent(
            path,
            tape,
            parent,
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
            parent_coefficient,
            additive,
            float(-natural[4]),
        )
    else:
        natural = np.zeros(5, dtype=np.float64)
        messages = parent
        coefficient = float(parent_coefficient)
    return EndpointLaw(
        messages=messages,
        quadratic_coefficient=coefficient,
        covariance=covariance,
        natural_parameter=np.ascontiguousarray(natural),
        amplitude=float(amplitude),
        diagnostics={
            "configuration_endpoint_positive": bool(configuration["positive"]),
            "quotient_endpoint_positive": bool(quotient["positive"]),
            "configuration_endpoint": configuration,
            "quotient_endpoint": quotient,
            "raw_endpoint_fisher_inner_product": raw_inner,
            "canonical_acute_orientation": orientation,
            "midpoint_denominator": denominator,
            "midpoint_unit_variance": midpoint_variance,
            "configuration_reliability": configuration_score_reliability(columns),
            "finite_chain_normalization": True,
            "new_fitted_parameter_count": 0,
        },
    )


__all__ = (
    "EndpointLaw",
    "configuration_score_reliability",
    "fisher_covariance",
    "fisher_rao_bernoulli_midpoint",
    "phase_protected_configuration_endpoint",
    "spherical_equal_information_endpoint",
)
