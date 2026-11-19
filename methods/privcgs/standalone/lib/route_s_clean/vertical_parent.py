"""Intrinsic soft-vertical and anti-closure parent laws.

The vertical conditional lives on the unfiltered sparse completion chart.  Its
first parent is the Gaussian empirical-Bayes soft occupation law; its second
parent adds the efficient joint phase/Palm score and the opposing closure score.
Every quantity is a checkpoint- or finite-chain-measurable sufficient statistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape
from .finite_chain import (
    PackedGaussianMessages,
    build_guarded_gaussian_messages,
)


@dataclass(frozen=True)
class SoftVerticalParent:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    carrier_reliability: float
    posterior_noise_signal_odds: float
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class VerticalAntiClosureParent:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    covariance: np.ndarray
    natural_parameter: np.ndarray
    joint_score_parameter: np.ndarray
    anti_closure_score_parameter: np.ndarray
    diagnostics: Mapping[str, object]


def orthogonal_degree_carrier_reliability(
    continuous_noisy_degree,
    operator_basis,
    degree_noise_std: float,
):
    """Positive-part Gaussian EB reliability off the released operator span."""

    observation = np.asarray(continuous_noisy_degree, dtype=np.float64)
    basis = np.asarray(operator_basis, dtype=np.float64)
    sigma = float(degree_noise_std)
    structural_basis = basis[:, 1:]
    centered = observation - float(np.mean(observation))
    residual = centered.copy()
    if structural_basis.shape[1]:
        residual -= structural_basis @ (structural_basis.T @ centered)
    residual_dimension = int(observation.size - basis.shape[1])
    residual_energy = float(np.dot(residual, residual))
    noise_variance = float(sigma * sigma)
    prior_variance = float(
        max(residual_energy / float(residual_dimension) - noise_variance, 0.0)
    )
    denominator = float(prior_variance + noise_variance)
    reliability = float(prior_variance / denominator) if denominator > 0.0 else 0.0
    return reliability, {
        "residual_dimension": residual_dimension,
        "residual_energy": residual_energy,
        "noise_variance": noise_variance,
        "positive_part_prior_variance": prior_variance,
        "posterior_mean_coefficient": reliability,
        "residual_constant_absolute_error": abs(float(np.sum(residual))),
        "residual_structural_l2_error": float(
            np.linalg.norm(structural_basis.T @ residual)
            if structural_basis.shape[1]
            else 0.0
        ),
        "checkpoint_only": True,
    }


def _views(flat, offsets):
    values = np.asarray(flat)
    boundary = np.asarray(offsets, dtype=np.int64)
    return tuple(
        values[int(boundary[index]) : int(boundary[index + 1])]
        for index in range(int(boundary.size - 1))
    )


def _logsumexp(values, axis=None):
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(array - maximum), axis=axis, keepdims=True)
    )
    return float(result.reshape(-1)[0]) if axis is None else np.squeeze(result, axis)


def _vector_chain_moments(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    quadratic_coefficient: float,
    additive,
):
    """Exact vector sufficient-statistic/degree Fisher moments."""

    matrix = np.asarray(additive, dtype=np.float64)
    states = _views(messages.state_value, messages.state_offsets)
    forward = _views(messages.forward_log_message, messages.state_offsets)
    columns = _views(matrix, messages.state_offsets)
    dimension = int(matrix.shape[1] + 1)
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    inverse_two_variance = 1.0 / (2.0 * float(degree_noise_std) ** 2)
    coefficient = float(quadratic_coefficient)
    first = int(order[0])
    d_first = (degree[first] - states[0] - target[first]) ** 2 * inverse_two_variance
    mean = np.concatenate((columns[0], d_first[:, None]), axis=1)
    second = np.einsum("si,sj->sij", mean, mean)
    for index in range(1, len(states)):
        node = int(order[index])
        previous = states[index - 1]
        current = states[index]
        d_added = (
            degree[node] + previous[:, None] - current[None, :] - target[node]
        ) ** 2 * inverse_two_variance
        log_weight = forward[index - 1][:, None] - coefficient * d_added
        normalizer = _logsumexp(log_weight, axis=0)
        probability = np.exp(log_weight - normalizer[None, :])
        added_column = np.asarray(columns[index], dtype=np.float64)
        new_mean = np.zeros((current.size, dimension), dtype=np.float64)
        new_second = np.zeros((current.size, dimension, dimension), dtype=np.float64)
        for current_index in range(current.size):
            added = np.concatenate(
                (
                    np.repeat(
                        added_column[current_index][None, :], previous.size, axis=0
                    ),
                    d_added[:, current_index, None],
                ),
                axis=1,
            )
            weight = probability[:, current_index]
            accumulated = mean + added
            new_mean[current_index] = weight @ accumulated
            cross = (
                np.einsum("si,sj->sij", mean, added)
                + np.einsum("si,sj->sij", added, mean)
                + np.einsum("si,sj->sij", added, added)
            )
            new_second[current_index] = np.einsum(
                "s,sij->ij", weight, second + cross
            )
        mean, second = new_mean, new_second
    last = int(order[-1])
    terminal_d = (
        degree[last] + states[-1] - target[last]
    ) ** 2 * inverse_two_variance
    terminal = forward[-1] - coefficient * terminal_d
    log_normalizer = float(_logsumexp(terminal))
    probability = np.exp(terminal - log_normalizer)
    terminal_added = np.zeros_like(mean)
    terminal_added[:, -1] = terminal_d
    final_mean_state = mean + terminal_added
    final_second_state = (
        second
        + np.einsum("si,sj->sij", mean, terminal_added)
        + np.einsum("si,sj->sij", terminal_added, mean)
        + np.einsum("si,sj->sij", terminal_added, terminal_added)
    )
    expectation = probability @ final_mean_state
    second_moment = np.einsum("s,sij->ij", probability, final_second_state)
    covariance = second_moment - np.outer(expectation, expectation)
    covariance = 0.5 * (covariance + covariance.T)
    return expectation, np.ascontiguousarray(covariance), {
        "dimension": dimension,
        "log_normalizer_error": float(abs(log_normalizer - messages.log_normalizer)),
    }


def _vector_chain_centered_covariance(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    quadratic_coefficient: float,
    additive,
):
    """Evaluate the same covariance by conditional total-covariance recursion.

    This path is used only when global second-moment subtraction has already
    produced a non-PSD matrix.  It accumulates conditional covariances and centered
    outer products, avoiding cancellation between ``E[SS^T]`` and ``E[S]E[S]^T``.
    """

    matrix = np.asarray(additive, dtype=np.float64)
    states = _views(messages.state_value, messages.state_offsets)
    forward = _views(messages.forward_log_message, messages.state_offsets)
    columns = _views(matrix, messages.state_offsets)
    dimension = int(matrix.shape[1] + 1)
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    inverse_two_variance = 1.0 / (2.0 * float(degree_noise_std) ** 2)
    coefficient = float(quadratic_coefficient)
    first = int(order[0])
    d_first = (degree[first] - states[0] - target[first]) ** 2 * inverse_two_variance
    mean = np.concatenate((columns[0], d_first[:, None]), axis=1)
    covariance = np.zeros(
        (states[0].size, dimension, dimension), dtype=np.float64
    )
    for index in range(1, len(states)):
        node = int(order[index])
        previous = states[index - 1]
        current = states[index]
        d_added = (
            degree[node] + previous[:, None] - current[None, :] - target[node]
        ) ** 2 * inverse_two_variance
        log_weight = forward[index - 1][:, None] - coefficient * d_added
        normalizer = _logsumexp(log_weight, axis=0)
        probability = np.exp(log_weight - normalizer[None, :])
        added_column = np.asarray(columns[index], dtype=np.float64)
        new_mean = np.empty((current.size, dimension), dtype=np.float64)
        new_covariance = np.empty(
            (current.size, dimension, dimension), dtype=np.float64
        )
        for current_index in range(current.size):
            added = np.empty((previous.size, dimension), dtype=np.float64)
            added[:, :-1] = added_column[current_index]
            added[:, -1] = d_added[:, current_index]
            conditional_mean = mean + added
            weight = probability[:, current_index]
            mixture_mean = weight @ conditional_mean
            centered = conditional_mean - mixture_mean
            new_mean[current_index] = mixture_mean
            new_covariance[current_index] = np.einsum(
                "s,sij->ij",
                weight,
                covariance + np.einsum("si,sj->sij", centered, centered),
            )
        mean, covariance = new_mean, new_covariance
    last = int(order[-1])
    terminal_d = (
        degree[last] + states[-1] - target[last]
    ) ** 2 * inverse_two_variance
    terminal = forward[-1] - coefficient * terminal_d
    log_normalizer = float(_logsumexp(terminal))
    probability = np.exp(terminal - log_normalizer)
    final_mean_state = mean.copy()
    final_mean_state[:, -1] += terminal_d
    expectation = probability @ final_mean_state
    centered = final_mean_state - expectation
    result = np.einsum(
        "s,sij->ij",
        probability,
        covariance + np.einsum("si,sj->sij", centered, centered),
    )
    result = 0.5 * (result + result.T)
    return expectation, np.ascontiguousarray(result), {
        "dimension": dimension,
        "log_normalizer_error": float(abs(log_normalizer - messages.log_normalizer)),
        "centered_total_covariance_recursion": True,
    }


def _vector_chain_expectation(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    quadratic_coefficient: float,
    additive,
):
    """Exact first moment without constructing an unused second moment."""

    matrix = np.asarray(additive, dtype=np.float64)
    states = _views(messages.state_value, messages.state_offsets)
    forward = _views(messages.forward_log_message, messages.state_offsets)
    columns = _views(matrix, messages.state_offsets)
    dimension = int(matrix.shape[1] + 1)
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    inverse_two_variance = 1.0 / (2.0 * float(degree_noise_std) ** 2)
    coefficient = float(quadratic_coefficient)
    first = int(order[0])
    d_first = (degree[first] - states[0] - target[first]) ** 2 * inverse_two_variance
    mean = np.concatenate((columns[0], d_first[:, None]), axis=1)
    for index in range(1, len(states)):
        node = int(order[index])
        previous = states[index - 1]
        current = states[index]
        d_added = (
            degree[node] + previous[:, None] - current[None, :] - target[node]
        ) ** 2 * inverse_two_variance
        log_weight = forward[index - 1][:, None] - coefficient * d_added
        normalizer = _logsumexp(log_weight, axis=0)
        probability = np.exp(log_weight - normalizer[None, :])
        added_column = np.asarray(columns[index], dtype=np.float64)
        new_mean = np.zeros((current.size, dimension), dtype=np.float64)
        for current_index in range(current.size):
            added = np.concatenate(
                (
                    np.repeat(
                        added_column[current_index][None, :], previous.size, axis=0
                    ),
                    d_added[:, current_index, None],
                ),
                axis=1,
            )
            weight = probability[:, current_index]
            accumulated = mean + added
            new_mean[current_index] = weight @ accumulated
        mean = new_mean
    last = int(order[-1])
    terminal_d = (
        degree[last] + states[-1] - target[last]
    ) ** 2 * inverse_two_variance
    terminal = forward[-1] - coefficient * terminal_d
    log_normalizer = float(_logsumexp(terminal))
    probability = np.exp(terminal - log_normalizer)
    terminal_added = np.zeros_like(mean)
    terminal_added[:, -1] = terminal_d
    final_mean_state = mean + terminal_added
    return probability @ final_mean_state


def _vector_tilt(
    path,
    tape: PackedRelocationTape,
    parent: PackedGaussianMessages,
    parent_coefficient: float,
    additive,
    natural_parameter,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
):
    matrix = np.asarray(additive, dtype=np.float64)
    natural = np.asarray(natural_parameter, dtype=np.float64)
    gain = np.ascontiguousarray(
        parent.structural_gain + matrix @ natural[:-1], dtype=np.float64
    )
    coefficient = float(parent_coefficient - float(natural[-1]))
    messages = build_guarded_gaussian_messages(
        path,
        replace(tape, structural_gain=gain),
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=coefficient,
        structural_temperature=1.0,
    )
    return messages, coefficient


def gaussian_soft_vertical_parent(
    path,
    tape: PackedRelocationTape,
    full_messages: PackedGaussianMessages,
    full_coefficient: float,
    horizontal_coefficient: float,
    vertical_occupation,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    carrier_reliability: float,
    diagnostics_required: bool = True,
) -> SoftVerticalParent:
    """Gaussian-odds soft law on the unfiltered vertical completion support."""

    reliability = float(carrier_reliability)
    vertical = np.asarray(vertical_occupation, dtype=np.float64)
    if vertical.shape == (tape.state_value.size,):
        vertical = vertical[:, None]
    posterior_odds = float((1.0 - reliability) / reliability)
    natural = np.asarray(
        [-posterior_odds, float(full_coefficient - horizontal_coefficient)],
        dtype=np.float64,
    )
    messages, coefficient = _vector_tilt(
        path,
        tape,
        full_messages,
        full_coefficient,
        vertical,
        natural,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
    )
    if not bool(diagnostics_required):
        return SoftVerticalParent(
            messages=messages,
            quadratic_coefficient=coefficient,
            carrier_reliability=reliability,
            posterior_noise_signal_odds=posterior_odds,
            diagnostics={},
        )
    expectation, _, moment_diagnostics = _vector_chain_moments(
        path,
        messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        coefficient,
        vertical,
    )
    return SoftVerticalParent(
        messages=messages,
        quadratic_coefficient=coefficient,
        carrier_reliability=reliability,
        posterior_noise_signal_odds=posterior_odds,
        diagnostics={
            "expected_vertical_completion_occupation": float(expectation[0]),
            "horizontal_degree_coefficient_preserved": bool(
                coefficient == float(horizontal_coefficient)
            ),
            "strictly_positive_soft_support": True,
            "new_fitted_parameter_count": 0,
            **dict(moment_diagnostics),
        },
    )


def vertical_anti_closure_parent(
    path,
    tape: PackedRelocationTape,
    parent: SoftVerticalParent,
    joint_phase_palm_additive,
    closure_additive,
    vertical_occupation,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    diagnostics_required: bool = True,
) -> VerticalAntiClosureParent:
    """Efficient joint phase/Palm score plus negative efficient closure score."""

    joint_mark = np.asarray(joint_phase_palm_additive, dtype=np.float64)
    closure = np.asarray(closure_additive, dtype=np.float64)
    if joint_mark.ndim == 2 and joint_mark.shape[1] == 1:
        joint_mark = joint_mark[:, 0]
    if closure.ndim == 2 and closure.shape[1] == 1:
        closure = closure[:, 0]
    joint = np.ascontiguousarray(np.column_stack((joint_mark, closure)))
    expectation, covariance, moment_diagnostics = _vector_chain_moments(
        path,
        parent.messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        parent.quadratic_coefficient,
        joint,
    )
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ArithmeticError("vertical Fisher covariance is invalid")
    covariance_scale = float(max(1.0, np.max(np.abs(covariance))))
    tolerance = float(1024.0 * np.finfo(np.float64).eps * covariance_scale)
    minimum = float(np.min(np.linalg.eigvalsh(covariance)))
    raw_minimum = minimum
    raw_tolerance = tolerance
    centered_fallback_used = False
    if minimum < -tolerance:
        expectation, covariance, centered_diagnostics = (
            _vector_chain_centered_covariance(
                path,
                parent.messages,
                source_degree,
                target_endpoint_mass,
                degree_noise_std,
                parent.quadratic_coefficient,
                joint,
            )
        )
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ArithmeticError("vertical centered Fisher covariance is invalid")
        covariance_scale = float(max(1.0, np.max(np.abs(covariance))))
        tolerance = float(1024.0 * np.finfo(np.float64).eps * covariance_scale)
        minimum = float(np.min(np.linalg.eigvalsh(covariance)))
        if minimum < -tolerance:
            raise ArithmeticError("vertical centered Fisher covariance is not PSD")
        centered_fallback_used = True
        moment_diagnostics = {
            **dict(moment_diagnostics),
            **dict(centered_diagnostics),
        }
    nuisance = np.asarray(covariance[1:, 1:], dtype=np.float64)
    nuisance_eigenvalues = np.linalg.eigvalsh(nuisance)
    nuisance_minimum = float(np.min(nuisance_eigenvalues))
    rank_adaptive = bool(nuisance_minimum <= tolerance)
    if nuisance_minimum < -tolerance:
        raise ArithmeticError("vertical closure-degree nuisance is not PSD")
    if not rank_adaptive:
        beta = np.linalg.solve(nuisance, covariance[1:, 0])
    else:
        # The Fisher metric is a seminorm on sufficient statistics.  Work in
        # its quotient: a statistic with zero variance on this support is the
        # zero tangent and contributes no normalized score.  The Moore--Penrose
        # projection is the unique minimum-norm representative of the remaining
        # efficient joint direction.
        singular = np.linalg.svd(nuisance, compute_uv=False)
        maximum_singular = float(np.max(singular, initial=0.0))
        inverse = np.linalg.pinv(
            nuisance,
            rcond=(
                tolerance / maximum_singular
                if maximum_singular > 0.0
                else np.finfo(np.float64).eps
            ),
        )
        beta = inverse @ covariance[1:, 0]
    joint_vector = np.asarray([1.0, -beta[0], -beta[1]], dtype=np.float64)
    joint_variance = float(joint_vector @ covariance @ joint_vector)
    if joint_variance < -tolerance:
        raise ArithmeticError("vertical joint phase/Palm variance is negative")
    joint_rank_positive = bool(joint_variance > tolerance)
    if not rank_adaptive and not joint_rank_positive:
        raise ArithmeticError("vertical joint phase/Palm rank is zero")
    degree_variance = float(covariance[2, 2])
    if degree_variance < -tolerance:
        raise ArithmeticError("vertical degree nuisance variance is negative")
    if not rank_adaptive and not degree_variance > tolerance:
        raise ArithmeticError("vertical degree nuisance has zero rank")
    closure_alpha = (
        float(covariance[1, 2] / degree_variance)
        if degree_variance > tolerance
        else 0.0
    )
    closure_vector = np.asarray([0.0, 1.0, -closure_alpha], dtype=np.float64)
    closure_variance = float(closure_vector @ covariance @ closure_vector)
    if closure_variance < -tolerance:
        raise ArithmeticError("vertical closure residual variance is negative")
    closure_rank_positive = bool(closure_variance > tolerance)
    if not rank_adaptive and not closure_rank_positive:
        raise ArithmeticError("vertical closure residual rank is zero")
    joint_score = (
        joint_vector / float(math.sqrt(joint_variance))
        if joint_rank_positive
        else np.zeros(3, dtype=np.float64)
    )
    anti_closure_score = (
        -closure_vector / float(math.sqrt(closure_variance))
        if closure_rank_positive
        else np.zeros(3, dtype=np.float64)
    )
    natural = np.asarray(joint_score + anti_closure_score, dtype=np.float64)
    if joint_rank_positive or closure_rank_positive:
        messages, coefficient = _vector_tilt(
            path,
            tape,
            parent.messages,
            parent.quadratic_coefficient,
            joint,
            natural,
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
        )
    else:
        messages = parent.messages
        coefficient = float(parent.quadratic_coefficient)
    vertical_expectation = None
    if bool(diagnostics_required):
        vertical = np.asarray(vertical_occupation, dtype=np.float64)
        if vertical.ndim == 1:
            vertical = vertical[:, None]
        vertical_expectation = _vector_chain_expectation(
            path,
            messages,
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
            coefficient,
            vertical,
        )
    return VerticalAntiClosureParent(
        messages=messages,
        quadratic_coefficient=coefficient,
        covariance=covariance,
        natural_parameter=natural,
        joint_score_parameter=np.ascontiguousarray(joint_score),
        anti_closure_score_parameter=np.ascontiguousarray(anti_closure_score),
        diagnostics={
            "fisher_minimum_eigenvalue": minimum,
            "fisher_machine_tolerance": tolerance,
            "fisher_raw_minimum_eigenvalue": raw_minimum,
            "fisher_raw_machine_tolerance": raw_tolerance,
            "fisher_centered_fallback_used": centered_fallback_used,
            "joint_closure_projection_coefficient": float(beta[0]),
            "joint_degree_projection_coefficient": float(beta[1]),
            "closure_degree_projection_coefficient": closure_alpha,
            "joint_residual_variance": joint_variance,
            "closure_residual_variance": closure_variance,
            "joint_unit_fisher_variance": float(
                joint_score @ covariance @ joint_score
            ),
            "closure_unit_fisher_variance": float(
                (-anti_closure_score) @ covariance @ (-anti_closure_score)
            ),
            "joint_closure_score_cross_covariance": float(
                joint_score @ covariance @ (-anti_closure_score)
            ),
            "expected_joint_phase_palm_score": float(expectation[0]),
            "anti_closure_orientation_is_negative": True,
            "new_fitted_parameter_count": 0,
            **dict(moment_diagnostics),
            **(
                {
                    "expected_vertical_completion_occupation": float(
                        vertical_expectation[0]
                    )
                }
                if vertical_expectation is not None
                else {}
            ),
            **(
                {
                    "fisher_quotient_rank_adaptive_branch": True,
                    "joint_score_rank_positive": joint_rank_positive,
                    "closure_score_rank_positive": closure_rank_positive,
                    "nuisance_rank": int(
                        np.count_nonzero(nuisance_eigenvalues > tolerance)
                    ),
                }
                if rank_adaptive
                else {}
            ),
        },
    )


__all__ = (
    "SoftVerticalParent",
    "VerticalAntiClosureParent",
    "gaussian_soft_vertical_parent",
    "orthogonal_degree_carrier_reliability",
    "vertical_anti_closure_parent",
)
