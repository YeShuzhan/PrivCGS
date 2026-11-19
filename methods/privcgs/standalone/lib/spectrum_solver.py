from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np
from types import SimpleNamespace
import spectrum_model
from spectrum_types import ReleasedSpectrum
import gfl_mpe_rkss1_pprs1_v1 as native
v1=SimpleNamespace(spectrum_impl=spectrum_model, ReleasedSpectrum=ReleasedSpectrum)


def _logsumexp(value: np.ndarray) -> float:
    finite = np.isfinite(value)
    if not np.any(finite):
        return -math.inf
    maximum = float(np.max(value[finite]))
    return maximum + math.log(float(np.sum(np.exp(value[finite] - maximum))))


def _maximum_entropy_log_initialization(
    support: np.ndarray, target_mean: float
) -> tuple[np.ndarray, float]:
    # Retain the frozen 256-step parameter search exactly; only reconstruct the
    # resulting law in log coordinates so its positive tail is not discarded.
    _probability, parameter = v1.spectrum_impl._maximum_entropy_initialization(
        support, target_mean)
    exponent = np.asarray(support, dtype=np.float64) * float(parameter)
    log_probability = exponent - _logsumexp(exponent)
    return np.ascontiguousarray(log_probability, dtype=np.float64), float(parameter)


def _signed_mean_equation_sign(
    log_q: np.ndarray,
    centered: np.ndarray,
    active: np.ndarray,
    tilt: float,
) -> int:
    denominator = 1.0 + float(tilt) * centered
    if np.any(denominator[active] <= 0.0):
        raise RuntimeError("full-log M-step left the open feasible interval")
    log_denominator = np.log(denominator)
    positive = active & (centered > 0.0)
    negative = active & (centered < 0.0)
    log_positive = _logsumexp(
        log_q[positive] + np.log(centered[positive]) - log_denominator[positive])
    log_negative = _logsumexp(
        log_q[negative] + np.log(-centered[negative]) - log_denominator[negative])
    if log_positive > log_negative:
        return 1
    if log_positive < log_negative:
        return -1
    return 0


def _mean_constrained_log_m_step(
    log_expected: np.ndarray,
    support: np.ndarray,
    target_mean: float,
) -> tuple[np.ndarray, float]:
    log_total = _logsumexp(log_expected)
    if not math.isfinite(log_total):
        raise RuntimeError("full-log EM has no released expected mass")
    log_q = np.asarray(log_expected, dtype=np.float64) - log_total
    centered = np.asarray(support, dtype=np.float64) - float(target_mean)
    active = np.isfinite(log_q)
    positive = active & (centered > 0.0)
    negative = active & (centered < 0.0)
    if not np.any(positive) or not np.any(negative):
        raise RuntimeError("released mean is not in the interior of the full-log support")
    open_margin = 64.0 * float(np.finfo(np.float64).eps)
    lower = -1.0 / float(np.max(centered[positive])) * (1.0 - open_margin)
    upper = 1.0 / float(np.max(-centered[negative])) * (1.0 - open_margin)
    if _signed_mean_equation_sign(log_q, centered, active, lower) <= 0:
        raise RuntimeError("full-log M-step lower endpoint does not bracket the mean")
    if _signed_mean_equation_sign(log_q, centered, active, upper) >= 0:
        raise RuntimeError("full-log M-step upper endpoint does not bracket the mean")
    for _ in range(128):
        middle = 0.5 * (lower + upper)
        if _signed_mean_equation_sign(log_q, centered, active, middle) > 0:
            lower = middle
        else:
            upper = middle
    tilt = 0.5 * (lower + upper)
    denominator = 1.0 + tilt * centered
    log_probability = np.full(log_q.size, -math.inf, dtype=np.float64)
    log_probability[active] = log_q[active] - np.log(denominator[active])
    log_probability -= _logsumexp(log_probability)
    return np.ascontiguousarray(log_probability), float(tilt)


def _linear_oracle_from_log_gradient(model, log_gradient: np.ndarray):
    scale = float(np.max(log_gradient[np.isfinite(log_gradient)]))
    scaled = np.zeros(log_gradient.size, dtype=np.float64)
    finite = np.isfinite(log_gradient)
    scaled[finite] = np.exp(log_gradient[finite] - scale)
    oracle_value, pair, weights = model.linear_oracle(scaled)
    return float(scale), float(oracle_value), pair, weights


def _finite_from_log(value: float) -> float:
    if value == -math.inf:
        return 0.0
    if value > math.log(float(np.finfo(np.float64).max)):
        return math.inf
    return math.exp(value)


def released_spectrum_in_memory_v41(checkpoint):
    observation = np.asarray(checkpoint.continuous_noisy_degree, dtype=np.float64)
    node_count = int(checkpoint.n_nodes)
    sigma = float(checkpoint.degree_sigma)
    edge_count = int(round(float(checkpoint.continuous_noisy_edge_mass)))
    endpoint_mass = 2 * edge_count
    target_mean = float(endpoint_mass) / float(node_count)
    machine_epsilon = float(np.finfo(np.float64).eps)
    tail_z = math.sqrt(2.0 * math.log(
        float(node_count) * float(node_count + 1) / machine_epsilon))
    support_maximum = min(
        node_count - 1,
        int(math.ceil(float(np.max(observation)) + sigma * tail_z)),
    )
    half_bandwidth = int(math.ceil(sigma * tail_z)) + 1
    model = v1.spectrum_impl.ReleasedMixture(
        observation, sigma, support_maximum, half_bandwidth, target_mean)
    log_probability, initialization_parameter = _maximum_entropy_log_initialization(
        model.support, target_mean)

    tolerance = 1.0 / math.sqrt(float(node_count))
    threshold_log_gap = math.log(float(node_count) * tolerance)
    check_period = int(math.ceil(math.log2(float(node_count))))
    # A mathematical stop is certificate based for every dataset.  n is only a
    # uniform operational safety ceiling; reaching it is an explicit failure,
    # never a switch to another solver.
    safety_iteration_ceiling = node_count
    previous_likelihood = -math.inf
    converged = False
    final_log_gap = math.inf
    final_pair = (-1, -1)
    final_weights = (0.0, 0.0)
    final_tilt = 0.0
    final_log_expected_mass_error = math.inf
    iterations = 0

    for iteration in range(safety_iteration_ceiling + 1):
        log_expected, row_log_density = native.expected_log_counts(
            model.observation, model.center, half_bandwidth, log_probability, sigma)
        chunk_size = 4096
        likelihood_without_constant = 0.0
        for start in range(0, row_log_density.size, chunk_size):
            likelihood_without_constant += float(np.sum(
                row_log_density[start : start + chunk_size]))
        likelihood = likelihood_without_constant - float(node_count) * math.log(
            sigma * math.sqrt(2.0 * math.pi))
        if likelihood + 128.0 * machine_epsilon * (
            1.0 + abs(likelihood)) < previous_likelihood:
            raise RuntimeError("uniform full-log released likelihood decreased beyond roundoff")
        previous_likelihood = likelihood

        log_gradient = np.full(log_probability.size, -math.inf, dtype=np.float64)
        finite_expected = np.isfinite(log_expected)
        log_gradient[finite_expected] = (
            log_expected[finite_expected] - log_probability[finite_expected])
        if iteration % check_period == 0:
            scale, oracle_scaled, pair, weights = _linear_oracle_from_log_gradient(
                model, log_gradient)
            dot_scaled = float(node_count) * math.exp(-scale)
            scaled_gap = max(0.0, oracle_scaled - dot_scaled)
            final_log_gap = (
                -math.inf if scaled_gap == 0.0 else scale + math.log(scaled_gap))
            final_pair = pair
            final_weights = weights
            final_log_expected_mass_error = (
                _logsumexp(log_expected) - math.log(float(node_count)))
            if final_log_gap <= threshold_log_gap:
                converged = True
                iterations = iteration
                break
        log_probability, final_tilt = _mean_constrained_log_m_step(
            log_expected, model.support, target_mean)
        iterations = iteration + 1

    if not converged:
        raise RuntimeError(
            "uniform full-log released spectrum certificate failed at the single "
            f"safety ceiling: log_gap_per_node={final_log_gap - math.log(float(node_count))} "
            f"> log_tolerance={math.log(tolerance)}")

    probability = np.exp(log_probability)
    probability /= float(np.sum(probability))
    represented_mean = float(np.dot(model.support, probability))
    if abs(represented_mean - target_mean) > 4096.0 * machine_epsilon * (
        1.0 + abs(target_mean)):
        raise RuntimeError(
            f"full-log probability conversion changed released mean: "
            f"observed={represented_mean} target={target_mean}")
    expected_count = float(node_count) * probability
    rounded = v1.spectrum_impl._largest_remainder_counts(expected_count)
    histogram, correction = v1.spectrum_impl._adjacent_mass_correction(
        rounded, expected_count, endpoint_mass)
    if int(np.sum(histogram, dtype=np.int64)) != node_count:
        raise AssertionError("integer spectrum lost node mass")
    if int(np.dot(np.arange(histogram.size), histogram)) != endpoint_mass:
        raise AssertionError("integer spectrum lost endpoint mass")
    graphicality = v1.spectrum_impl._erdos_gallai(histogram)
    final_gap = _finite_from_log(final_log_gap)
    minimum_finite_log_probability = float(np.min(
        log_probability[np.isfinite(log_probability)]))
    return v1.ReleasedSpectrum(
        probability=np.asarray(probability, dtype=np.float64),
        integer_histogram=np.asarray(histogram, dtype=np.int64),
        edge_count=edge_count,
        endpoint_mass=endpoint_mass,
        diagnostics={
            "initialization_parameter": float(initialization_parameter),
            "iterations": int(iterations),
            "duality_gap": final_gap,
            "duality_gap_per_node": final_gap / float(node_count),
            "duality_gap_log": float(final_log_gap),
            "duality_gap_per_node_log": float(
                final_log_gap - math.log(float(node_count))),
            "oracle_pair": [int(final_pair[0]), int(final_pair[1])],
            "oracle_pair_weights": [float(final_weights[0]), float(final_weights[1])],
            "m_step_tilt": float(final_tilt),
            "integer_correction": correction,
            "graphicality": graphicality,
            "uniform_single_path": True,
            "full_log_probability_state": True,
            "support_loss_gate_used": False,
            "probability_floor_used": False,
            "fallback_solver_used": False,
            "dataset_or_size_branch_used": False,
            "minimum_finite_log_probability": minimum_finite_log_probability,
            "log_expected_mass_error": float(final_log_expected_mass_error),
            "released_mean_preserved": True,
            "privacy_release_changed": False,
            "random_stream_changed": False,
            "objective_changed": False,
            "oracle_changed": False,
            "certificate_threshold_changed": False,
            "certificate_relaxed": False,
        },
    )
