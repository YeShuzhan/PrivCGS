"""Checkpoint-only recovery of the normalized explicit finite graph law."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np

from .checkpoint import GaussianCheckpoint, load_gaussian_checkpoint
from .contracts import RecoveredGraphLaw


@dataclass(frozen=True)
class RecoveredCheckpointState:
    checkpoint_path: Path
    checkpoint: GaussianCheckpoint
    temperature: float
    temperature_diagnostics: Mapping[str, object]
    graph_law: RecoveredGraphLaw
    endpoint_target: np.ndarray
    degree_carrier_target: np.ndarray
    recovery_diagnostics: Mapping[str, object]
    public_experiment_id: int
    stage_seconds: Mapping[str, float]
    audit: Mapping[str, int]


@dataclass(frozen=True)
class _CoherentGaussianBase:
    target_edges: int
    endpoint_target: np.ndarray
    raw_operator: np.ndarray
    coherent_operator: np.ndarray
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _ShellPosteriorState:
    coefficient: np.ndarray
    posterior_variance: np.ndarray
    records: Tuple[Mapping[str, object], ...]
    positive_shell_count: int
    retained_coordinate_count: int


@dataclass(frozen=True)
class _RecoveredGaussianState:
    target_edges: int
    degree_target: np.ndarray
    operator: np.ndarray
    node_factors: np.ndarray
    spectrum: np.ndarray
    active_rank: int


@dataclass(frozen=True)
class _HaarLevel:
    left_sizes: np.ndarray
    right_sizes: np.ndarray
    details: np.ndarray
    carried_unpaired: bool


def _sociability_ordered_haar_forward(values: np.ndarray):
    """Orthogonal adjacent-interval Haar transform for arbitrary node counts."""

    current = np.asarray(values, dtype=np.float64)
    current = current.copy()
    sizes = np.ones(current.size, dtype=np.int64)
    levels = []
    while current.size > 1:
        pair_count = current.size // 2
        left_sizes = sizes[0 : 2 * pair_count : 2].copy()
        right_sizes = sizes[1 : 2 * pair_count : 2].copy()
        left = current[0 : 2 * pair_count : 2]
        right = current[1 : 2 * pair_count : 2]
        left_root = np.sqrt(left_sizes.astype(np.float64))
        right_root = np.sqrt(right_sizes.astype(np.float64))
        union_root = np.sqrt((left_sizes + right_sizes).astype(np.float64))
        unions = (left_root * left + right_root * right) / union_root
        details = (right_root * left - left_root * right) / union_root
        carried = bool(current.size % 2)
        if carried:
            next_current = np.concatenate((unions, current[-1:]))
            next_sizes = np.concatenate((left_sizes + right_sizes, sizes[-1:]))
        else:
            next_current = unions
            next_sizes = left_sizes + right_sizes
        levels.append(
            _HaarLevel(
                left_sizes=left_sizes,
                right_sizes=right_sizes,
                details=np.asarray(details, dtype=np.float64),
                carried_unpaired=carried,
            )
        )
        current = np.asarray(next_current, dtype=np.float64)
        sizes = np.asarray(next_sizes, dtype=np.int64)
    return float(current[0]), tuple(levels)


def _sociability_ordered_haar_inverse(
    root_scaling: float,
    levels,
    details_by_level,
) -> np.ndarray:
    current = np.asarray([float(root_scaling)], dtype=np.float64)
    for level, level_details in zip(reversed(levels), reversed(details_by_level)):
        left_sizes = np.asarray(level.left_sizes, dtype=np.int64)
        right_sizes = np.asarray(level.right_sizes, dtype=np.int64)
        detail = np.asarray(level_details, dtype=np.float64)
        pair_count = int(left_sizes.size)
        left_root = np.sqrt(left_sizes.astype(np.float64))
        right_root = np.sqrt(right_sizes.astype(np.float64))
        union_root = np.sqrt((left_sizes + right_sizes).astype(np.float64))
        unions = current[:pair_count]
        left = (left_root * unions + right_root * detail) / union_root
        right = (right_root * unions - left_root * detail) / union_root
        previous = np.empty(
            2 * pair_count + int(level.carried_unpaired), dtype=np.float64
        )
        previous[0 : 2 * pair_count : 2] = left
        previous[1 : 2 * pair_count : 2] = right
        if level.carried_unpaired:
            previous[-1] = current[-1]
        current = previous
    return current


def recover_rank_coupled_degree_carrier(
    checkpoint: GaussianCheckpoint,
    coherent_endpoint_target: np.ndarray,
    sociability: np.ndarray,
    target_edges: int,
):
    """Recover the coefficient-free Gaussian EB carrier and its rank coupling.

    The residual degree channel is Rao--Blackwellized off the released operator
    span.  Its nonnegative fixed-mass empirical marginal is then coupled
    monotonically to a multiscale Haar score ordered by recovered sociability.
    """

    basis = np.asarray(checkpoint.basis, dtype=np.float64)
    observation = np.asarray(checkpoint.continuous_noisy_degree, dtype=np.float64)
    coherent = np.asarray(coherent_endpoint_target, dtype=np.float64)
    atom = np.asarray(sociability, dtype=np.float64)
    n_nodes, rank = basis.shape
    structural_basis = basis[:, 1:]
    centered = observation - float(np.mean(observation))
    residual = centered.copy()
    if structural_basis.shape[1]:
        residual -= structural_basis @ (structural_basis.T @ centered)
    residual_dimension = int(n_nodes - rank)
    residual_energy = float(np.dot(residual, residual))
    noise_variance = float(checkpoint.degree_sigma) ** 2
    prior_variance = max(
        residual_energy / float(residual_dimension) - noise_variance, 0.0
    )
    denominator = prior_variance + noise_variance
    coefficient = prior_variance / denominator if denominator > 0.0 else 0.0
    target_mass = 2.0 * float(target_edges)
    target_mean = target_mass / float(n_nodes)
    centered_coherent = coherent - target_mean
    anchor = np.full(n_nodes, target_mean, dtype=np.float64)
    if structural_basis.shape[1]:
        anchor += structural_basis @ (structural_basis.T @ centered_coherent)
    empirical_target = project_degree_to_mass_simplex(
        anchor + coefficient * residual, target_mass
    )

    sociability_order = np.lexsort(
        (np.arange(n_nodes, dtype=np.int64), atom)
    )
    ordered_observation = np.asarray(centered[sociability_order], dtype=np.float64)
    _root, levels = _sociability_ordered_haar_forward(ordered_observation)
    shrunk_details = []
    level_coefficients = []
    for level in levels:
        detail = np.asarray(level.details, dtype=np.float64)
        energy = float(np.dot(detail, detail))
        level_prior = max(
            energy / float(detail.size) - noise_variance, 0.0
        )
        level_denominator = level_prior + noise_variance
        level_coefficient = (
            level_prior / level_denominator if level_denominator > 0.0 else 0.0
        )
        level_coefficients.append(float(level_coefficient))
        shrunk_details.append(level_coefficient * detail)
    ordered_score = _sociability_ordered_haar_inverse(
        0.0, levels, shrunk_details
    )
    score = np.empty(n_nodes, dtype=np.float64)
    score[sociability_order] = ordered_score
    score_order = np.lexsort(
        (np.arange(n_nodes, dtype=np.int64), atom, score)
    )
    sorted_carrier = np.sort(empirical_target, kind="stable")
    coupled = np.empty(n_nodes, dtype=np.float64)
    coupled[score_order] = sorted_carrier
    # ``coupled`` is now certified to be an exact permutation of the projected
    # empirical carrier.  A second floating sum in permutation order is not a
    # mass certificate: binary64 addition is order-dependent, and can cross the
    # absolute consumer threshold although no element or mathematical mass was
    # changed.  Certify the canonical pre-permutation carrier instead; retain the
    # old reordered sum below as a diagnostic so every previously valid result
    # keeps its exact diagnostics.
    canonical_mass_error = abs(float(np.sum(empirical_target)) - target_mass)
    mass_error = abs(float(np.sum(coupled)) - target_mass)
    if canonical_mass_error > 1.0e-8:
        raise ArithmeticError("rank-coupled degree carrier lost edge mass")
    return np.asarray(coupled, dtype=np.float64), {
        "carrier_residual_dimension": residual_dimension,
        "carrier_residual_energy": residual_energy,
        "carrier_noise_variance": noise_variance,
        "carrier_empirical_prior_variance": prior_variance,
        "carrier_posterior_mean_coefficient": float(coefficient),
        "carrier_postprojection_mass_error": float(
            abs(float(np.sum(empirical_target)) - target_mass)
        ),
        "carrier_rank_coupling_mass_error": float(mass_error),
        "carrier_multiscale_level_count": int(len(levels)),
        "carrier_multiscale_active_level_count": int(
            np.count_nonzero(np.asarray(level_coefficients) > 0.0)
        ),
        "carrier_multiscale_level_coefficients": level_coefficients,
        "carrier_hidden_empirical_constant_count": 0,
        "carrier_metric_feedback": False,
        "carrier_private_graph_access": False,
    }


def operator_matrix_from_packed(values: np.ndarray, rank: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    order = int(rank)
    matrix = np.zeros((order, order), dtype=np.float64)
    index = 0
    for left in range(order):
        matrix[left, left] = float(vector[index])
        index += 1
        for right in range(left + 1, order):
            value = float(vector[index])
            matrix[left, right] = value
            matrix[right, left] = value
            index += 1
    return matrix


def project_degree_to_mass_simplex(
    continuous_noisy_degree: np.ndarray,
    target_degree_mass: float,
) -> np.ndarray:
    values = np.asarray(continuous_noisy_degree, dtype=np.float64)
    target = float(target_degree_mass)
    if values.size == 0 or target == 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    order = np.sort(values)[::-1]
    cumulative = np.cumsum(order)
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    positive = order - (cumulative - target) / indices > 0.0
    if not np.any(positive):
        return np.full(values.shape, target / float(values.size), dtype=np.float64)
    active_count = int(np.flatnonzero(positive)[-1]) + 1
    threshold = float(
        (cumulative[active_count - 1] - target) / float(active_count)
    )
    projected = np.maximum(values - threshold, 0.0)
    residual = target - float(np.sum(projected))
    # The consumer below certifies mass with an absolute 1e-8 tolerance.  Use
    # that same threshold here: the former relative threshold grew to 6e-4 at
    # 300K/3M and could therefore skip the existing active-set correction for a
    # 1e-7 residual that the consumer immediately rejected.  This changes no
    # already-certified projection and only completes the same simplex threshold
    # correction when the old path could not produce a valid recovered state.
    if abs(residual) > 1.0e-8:
        active = np.flatnonzero(projected > 0.0)
        if not active.size:
            active = np.arange(values.size, dtype=np.int64)
        projected[active] += residual / float(active.size)
        projected = np.maximum(projected, 0.0)
    # On multi-million-node graphs the distributed correction above can be
    # smaller than one binary64 ulp at every active coordinate.  Close only the
    # remaining floating-point summation residue on a deterministic positive
    # coordinate.  This preserves the same mathematical simplex projection and
    # makes the downstream absolute 1e-8 mass certificate scale-independent.
    for _ in range(4):
        residual = target - float(np.sum(projected))
        if abs(residual) <= 1.0e-8:
            break
        anchor = int(np.argmax(projected))
        corrected = float(projected[anchor]) + residual
        if corrected < 0.0:
            raise ArithmeticError("simplex roundoff closure would violate nonnegativity")
        projected[anchor] = corrected
    return projected


def _precision_fused_scalar(
    observations: np.ndarray,
    variances: np.ndarray,
) -> Tuple[float, float]:
    values = np.asarray(observations, dtype=np.float64)
    noise = np.asarray(variances, dtype=np.float64)
    exact = noise == 0.0
    if np.any(exact):
        return float(np.mean(values[exact])), 0.0
    precision = 1.0 / noise
    total_precision = float(np.sum(precision))
    return (
        float(np.dot(precision, values) / total_precision),
        float(1.0 / total_precision),
    )


def _constant_mode_sign(basis: np.ndarray) -> float:
    matrix = np.asarray(basis, dtype=np.float64)
    n_nodes = int(matrix.shape[0])
    constant = np.full(n_nodes, 1.0 / math.sqrt(float(n_nodes)), dtype=np.float64)
    overlap = float(np.dot(matrix[:, 0], constant))
    return 1.0 if overlap >= 0.0 else -1.0


def _recover_coherent_gaussian_base(
    checkpoint: GaussianCheckpoint,
) -> _CoherentGaussianBase:
    degree = np.asarray(checkpoint.continuous_noisy_degree, dtype=np.float64)
    raw_vector = np.asarray(checkpoint.raw_noisy_operator_vector, dtype=np.float64)
    basis = np.asarray(checkpoint.basis, dtype=np.float64)
    n_nodes = int(checkpoint.n_nodes)
    rank = int(basis.shape[1])
    sigma_degree = float(checkpoint.degree_sigma)
    sigma_operator = float(checkpoint.operator_noise_std)
    mass_weight = float(checkpoint.mass_coordinate_weight)
    sign = _constant_mode_sign(basis)

    degree_mass = 0.5 * float(np.sum(degree))
    operator_mass = float(n_nodes) * float(raw_vector[0])
    direct_mass = float(checkpoint.continuous_noisy_edge_mass)
    fused_mass, fused_mass_variance = _precision_fused_scalar(
        np.asarray([degree_mass, operator_mass, direct_mass]),
        np.asarray(
            [
                float(n_nodes) * sigma_degree * sigma_degree / 4.0,
                float(n_nodes * n_nodes) * sigma_operator * sigma_operator,
                (sigma_operator / mass_weight) ** 2,
            ]
        ),
    )
    maximum_edges = n_nodes * (n_nodes - 1) // 2
    target_edges = min(max(int(np.rint(fused_mass)), 0), maximum_edges)

    raw_operator = operator_matrix_from_packed(raw_vector, rank)
    structural_basis = basis[:, 1:]
    if rank > 1:
        degree_cross = sign / math.sqrt(float(n_nodes)) * (
            structural_basis.T @ degree
        )
        operator_cross = np.asarray(raw_operator[0, 1:], dtype=np.float64)
        degree_variance = sigma_degree * sigma_degree / float(n_nodes)
        operator_variance = sigma_operator * sigma_operator
        if degree_variance == 0.0:
            fused_cross = degree_cross
            fused_cross_variance = 0.0
        elif operator_variance == 0.0:
            fused_cross = operator_cross
            fused_cross_variance = 0.0
        else:
            degree_precision = 1.0 / degree_variance
            operator_precision = 1.0 / operator_variance
            fused_cross = (
                degree_precision * degree_cross
                + operator_precision * operator_cross
            ) / (degree_precision + operator_precision)
            fused_cross_variance = 1.0 / (
                degree_precision + operator_precision
            )
        projected_degree = degree + structural_basis @ (
            sign * math.sqrt(float(n_nodes)) * fused_cross
            - structural_basis.T @ degree
        )
    else:
        fused_cross = np.asarray([], dtype=np.float64)
        projected_degree = degree.copy()
        fused_cross_variance = 0.0
    target = project_degree_to_mass_simplex(
        projected_degree, 2.0 * float(target_edges)
    )

    coherent = np.zeros((rank, rank), dtype=np.float64)
    coherent[0, 0] = float(target_edges) / float(n_nodes)
    if rank > 1:
        coherent_cross = sign / math.sqrt(float(n_nodes)) * (
            structural_basis.T @ target
        )
        coherent[0, 1:] = coherent_cross
        coherent[1:, 0] = coherent_cross
    else:
        coherent_cross = np.asarray([], dtype=np.float64)
    diagnostics = {
        "gcs_mass_degree_observation": float(degree_mass),
        "gcs_mass_operator_constant_observation": float(operator_mass),
        "gcs_mass_direct_observation": float(direct_mass),
        "gcs_mass_fused_continuous": float(fused_mass),
        "gcs_mass_fused_variance": float(fused_mass_variance),
        "gcs_target_edges": int(target_edges),
        "gcs_constant_mode_sign": float(sign),
        "gcs_cross_coordinate_count": int(fused_cross.size),
        "gcs_cross_fused_variance": float(fused_cross_variance),
        "gcs_cross_pre_simplex_l2_shift": float(
            np.linalg.norm(projected_degree - degree)
        ),
        "gcs_cross_post_simplex_operator_l2_shift": float(
            np.linalg.norm(coherent_cross - raw_operator[0, 1:])
        ),
        "gcs_degree_mass_coherence_absolute_error": float(
            abs(float(np.sum(target)) - 2.0 * float(target_edges))
        ),
        "gcs_operator_mass_coherence_absolute_error": float(
            abs(float(coherent[0, 0]) - float(target_edges) / float(n_nodes))
        ),
        "gcs_operator_degree_cross_coherence_l2_error": float(
            np.linalg.norm(
                coherent[0, 1:]
                - sign
                / math.sqrt(float(n_nodes))
                * (structural_basis.T @ target)
            )
        ),
        "gcs_active_hidden_empirical_constant_count": 0,
        "gcs_metric_feedback": False,
        "gcs_private_graph_access": False,
    }
    return _CoherentGaussianBase(
        target_edges=int(target_edges),
        endpoint_target=np.asarray(target, dtype=np.float64),
        raw_operator=raw_operator,
        coherent_operator=coherent,
        diagnostics=diagnostics,
    )


def _compile_shell_posterior_state(
    raw_operator: np.ndarray,
    noise_std: float,
) -> _ShellPosteriorState:
    operator = np.asarray(raw_operator, dtype=np.float64)
    rank = int(operator.shape[0])
    sigma_squared = float(noise_std) ** 2
    coefficient = np.zeros(rank, dtype=np.float64)
    posterior_variance = np.zeros_like(operator)
    shell_records = []
    positive_shells = 0
    retained_coordinates = 0
    for outer in range(1, rank):
        values = np.asarray(
            [operator[inner, outer] for inner in range(1, outer + 1)],
            dtype=np.float64,
        )
        dimension = int(values.size)
        energy = float(np.dot(values, values))
        prior_variance = max(energy / float(dimension) - sigma_squared, 0.0)
        shell_coefficient = (
            prior_variance / (prior_variance + sigma_squared)
            if prior_variance > 0.0
            else 0.0
        )
        coefficient[outer] = shell_coefficient
        variance = shell_coefficient * sigma_squared
        posterior_variance[1 : outer + 1, outer] = variance
        posterior_variance[outer, 1 : outer + 1] = variance
        if shell_coefficient > 0.0:
            positive_shells += 1
            retained_coordinates += dimension
        shell_records.append(
            {
                "shell": int(outer),
                "dimension": dimension,
                "energy": energy,
                "prior_variance": float(prior_variance),
                "posterior_coefficient": float(shell_coefficient),
                "posterior_variance": float(variance),
            }
        )
    return _ShellPosteriorState(
        coefficient=coefficient,
        posterior_variance=posterior_variance,
        records=tuple(shell_records),
        positive_shell_count=int(positive_shells),
        retained_coordinate_count=int(retained_coordinates),
    )


def _recover_operator_shell_state(
    checkpoint: GaussianCheckpoint,
    base: _CoherentGaussianBase,
    shell: _ShellPosteriorState,
) -> Tuple[_RecoveredGaussianState, Mapping[str, object]]:
    rank = int(np.asarray(checkpoint.basis).shape[1])
    coherent = np.asarray(base.coherent_operator, dtype=np.float64).copy()
    for outer in range(1, rank):
        shell_coefficient = float(shell.coefficient[outer])
        for inner in range(1, outer + 1):
            value = shell_coefficient * float(base.raw_operator[inner, outer])
            coherent[inner, outer] = value
            coherent[outer, inner] = value
    eigenvalues, eigenvectors = np.linalg.eigh(coherent)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    spectrum = np.asarray(eigenvalues[order], dtype=np.float64)
    node_factors = np.asarray(
        np.asarray(checkpoint.basis, dtype=np.float64) @ eigenvectors[:, order],
        dtype=np.float64,
    )
    tolerance = (
        np.finfo(np.float64).eps
        * max(rank, 1)
        * max(float(np.max(np.abs(spectrum))), 1.0)
    )
    active_rank = int(np.count_nonzero(np.abs(spectrum) > tolerance))
    if active_rank == 0:
        active_rank = 1
    recovered = _RecoveredGaussianState(
        target_edges=int(base.target_edges),
        degree_target=np.asarray(base.endpoint_target, dtype=np.float64),
        operator=coherent,
        node_factors=node_factors,
        spectrum=spectrum,
        active_rank=int(active_rank),
    )
    diagnostics = {
        **dict(base.diagnostics),
        "operator_formula_ablation": "fixed-basis-shell empirical Bayes",
        "operator_formula_shell_count": int(rank - 1),
        "operator_formula_positive_shell_count": int(shell.positive_shell_count),
        "operator_formula_retained_coordinate_count": int(
            shell.retained_coordinate_count
        ),
        "operator_formula_algebraic_active_rank": int(active_rank),
        "operator_formula_shell_records": shell.records,
        "operator_formula_private_graph_access_after_checkpoint": False,
        "operator_formula_full_dyad_enumeration": False,
        "gcs_operator_algebraic_active_rank": int(active_rank),
        "gcs_operator_numerical_rank_tolerance": float(tolerance),
    }
    return recovered, diagnostics


def _packed_feature_variance(
    basis: np.ndarray,
    endpoint_probability: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float64)
    probability = np.asarray(endpoint_probability, dtype=np.float64).reshape(-1)
    probability = probability / float(np.sum(probability))
    probability_squared = probability * probability
    distinct_mass = 1.0 - float(np.sum(probability_squared))
    rank = int(matrix.shape[1])
    variance = np.zeros((rank, rank), dtype=np.float64)
    first = probability @ matrix
    second = probability @ np.square(matrix)
    for left in range(1, rank):
        x = matrix[:, left]
        x_squared = x * x
        for right in range(left, rank):
            y = matrix[:, right]
            if left == right:
                mean = (
                    first[left] * first[left]
                    - float(np.dot(probability_squared, x_squared))
                ) / distinct_mass
                second_moment = (
                    second[left] * second[left]
                    - float(np.dot(probability_squared, x_squared * x_squared))
                ) / distinct_mass
            else:
                xy = x * y
                mean = 2.0 * (
                    first[left] * first[right]
                    - float(np.dot(probability_squared, xy))
                ) / distinct_mass
                second_moment = (
                    2.0 * second[left] * second[right]
                    + 2.0 * float(np.dot(probability, xy)) ** 2
                    - 4.0 * float(np.dot(probability_squared, xy * xy))
                ) / distinct_mass
            coordinate_variance = max(second_moment - mean * mean, 0.0)
            variance[left, right] = coordinate_variance
            variance[right, left] = coordinate_variance
    return variance


def derive_source_temperature(
    checkpoint: GaussianCheckpoint,
    target_edges: int,
    endpoint_target: np.ndarray,
    shell: _ShellPosteriorState,
) -> Tuple[float, Mapping[str, object]]:
    rank = int(np.asarray(checkpoint.basis).shape[1])
    endpoint_denominator = (
        2.0 * float(target_edges)
        + 0.5 * float(checkpoint.n_nodes)
    )
    endpoint_probability = (
        np.asarray(endpoint_target, dtype=np.float64) + 0.5
    ) / endpoint_denominator
    feature_variance = _packed_feature_variance(
        np.asarray(checkpoint.basis, dtype=np.float64), endpoint_probability
    )
    rows, columns = np.triu_indices(rank, k=0)
    residual = (rows >= 1) & (columns >= 1)
    trace = float(
        np.dot(
            shell.posterior_variance[rows[residual], columns[residual]],
            feature_variance[rows[residual], columns[residual]],
        )
    )
    trace = max(trace, 0.0)
    temperature = math.sqrt(6.0 * trace) / math.pi
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ArithmeticError("zero-temperature weak-limit executor is unavailable")
    active_shell_count = int(shell.positive_shell_count)
    return float(temperature), {
        "formula": "sqrt(6 * tr(Sigma_O C_nu)) / pi",
        "formula_temperature": float(temperature),
        "operator_posterior_feature_trace": trace,
        "fixed_public_basis_shell_count": int(len(shell.records)),
        "positive_posterior_shell_count": int(active_shell_count),
        "operator_coordinate_count": int(rank * (rank + 1) // 2),
        "residual_operator_coordinate_count": int(rank * (rank - 1) // 2),
        "jeffreys_endpoint_offset": 0.5,
        "full_dyad_enumeration": False,
        "private_graph_access_after_checkpoint": False,
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - float(np.mean(array))) / max(float(np.std(array)), 1.0e-8)


def _public_generator(public_experiment_id: int, domain: int) -> np.random.Generator:
    identity = int(public_experiment_id)
    stream = int(domain)
    seed = np.random.SeedSequence(
        [identity & 0xFFFFFFFF, (identity >> 32) & 0xFFFFFFFF, stream & 0xFFFFFFFF]
    )
    return np.random.Generator(np.random.Philox(seed))


def _freeze_base_graph_law(
    checkpoint: GaussianCheckpoint,
    recovered: _RecoveredGaussianState,
    endpoint_target: np.ndarray,
    temperature: float,
) -> Tuple[RecoveredGraphLaw, Mapping[str, object]]:
    degree = np.maximum(np.asarray(recovered.degree_target, dtype=np.float64), 0.0)
    standardized = _standardize(np.log1p(degree))
    sociability = np.exp(np.clip(0.50 * standardized, -3.0, 3.0)) + 1.0e-4
    star = np.sqrt(degree + 0.5)
    star /= max(float(np.mean(star)), 1.0e-12)

    spectrum = np.asarray(recovered.spectrum, dtype=np.float64)
    factors = np.asarray(recovered.node_factors, dtype=np.float64)
    active = np.arange(int(recovered.active_rank), dtype=np.int64)
    interaction = factors[:, active] * np.sqrt(
        np.abs(spectrum[active]) + 1.0e-12
    )[None, :]
    norm = np.linalg.norm(interaction, axis=1, keepdims=True)
    interaction = interaction / np.maximum(norm, 1.0e-12)
    diagonal = np.tanh(
        spectrum[active]
        / max(float(np.max(np.abs(spectrum[active]))), 1.0e-12)
    )

    bin_count = max(int(math.ceil(math.log2(max(checkpoint.n_nodes, 2)))), 2)
    cuts = np.quantile(
        sociability, np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    )
    parent = np.searchsorted(cuts, sociability, side="right").astype(np.int64)
    leading = interaction[:, : min(2, interaction.shape[1])]
    component = np.zeros(checkpoint.n_nodes, dtype=np.int64)
    if leading.shape[1] >= 1:
        component += (leading[:, 0] >= 0.0).astype(np.int64)
    if leading.shape[1] >= 2:
        component += 2 * (leading[:, 1] >= 0.0).astype(np.int64)

    membership = np.zeros((checkpoint.n_nodes, bin_count), dtype=np.float64)
    membership[np.arange(checkpoint.n_nodes), parent] = 1.0
    node_mass = sociability / max(float(np.sum(sociability)), 1.0e-12)
    fiber_mass = membership.T @ node_mass
    aggregate = membership.T @ interaction
    aggregate /= np.maximum(np.sum(membership, axis=0)[:, None], 1.0)
    quotient = aggregate @ np.diag(diagonal) @ aggregate.T
    composed = quotient @ (fiber_mass[:, None] * quotient)
    raw_flux = np.clip(composed - quotient, -4.0, 4.0)
    raw_flux -= float(
        np.sum(fiber_mass[:, None] * fiber_mass[None, :] * raw_flux)
    )
    parent_flux = 0.018 * raw_flux / max(float(np.std(raw_flux)), 1.0e-8)

    parent_potential = np.zeros(bin_count, dtype=np.float64)
    for current in range(bin_count):
        nodes = np.flatnonzero(parent == current)
        if nodes.size:
            parent_potential[current] = float(np.mean(standardized[nodes]))
    parent_potential -= float(np.sum(fiber_mass * parent_potential))
    marked_shift = 0.05 * parent_potential / max(
        float(np.std(parent_potential)), 1.0e-8
    )
    component_shift = np.asarray([0.030, 0.020, 0.0, -0.015], dtype=np.float64)
    confidence = np.linalg.norm(interaction, axis=1)
    confidence = np.clip(
        confidence / max(float(np.max(confidence)), 1.0e-12), 0.0, 1.0
    )
    dust = max(
        float(recovered.target_edges) / max(float(checkpoint.n_nodes**2), 1.0) * 0.02,
        1.0e-8,
    )
    digest = hashlib.sha256()
    for value in (
        sociability,
        star,
        interaction,
        diagonal,
        parent,
        component,
        marked_shift,
        component_shift,
        parent_flux,
    ):
        digest.update(np.asarray(value).tobytes())
    law = RecoveredGraphLaw(
        target_edges=int(recovered.target_edges),
        degree_noise_std=float(checkpoint.degree_sigma),
        temperature=float(temperature),
        target_endpoint_mass=np.asarray(endpoint_target, dtype=np.float64),
        sociability=sociability,
        star=star,
        interaction=interaction,
        interaction_diagonal=diagonal,
        dust=float(dust),
        parent_fiber=parent,
        component_fiber=component,
        marked_shift=marked_shift,
        component_shift=component_shift,
        parent_flux=parent_flux,
        noisy_degree=degree,
        teacher_confidence=confidence,
        state_hash=digest.hexdigest(),
    )
    return law, {
        "active_rank": int(active.size),
        "parent_fiber_count": int(bin_count),
        "component_fiber_count": 4,
    }


def _softplus(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.maximum(array, 0.0) + np.log1p(np.exp(-np.abs(array))) + 1.0e-6


def _partition_rescale(
    diagonal: np.ndarray,
    left_coordinate: np.ndarray,
    right_coordinate: np.ndarray,
    endpoint_weight: np.ndarray,
    target: float,
) -> float:
    if not diagonal.size or float(np.max(np.abs(diagonal))) <= 1.0e-12:
        return 1.0
    unit_bilinear = np.sum(
        left_coordinate * diagonal[None, :] * right_coordinate, axis=1
    )
    low = 0.25
    high = 4.0
    for _ in range(48):
        middle = 0.5 * (low + high)
        value = float(
            np.mean(endpoint_weight * _softplus(middle * unit_bilinear))
        )
        if value < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _freeze_commutant_functional(
    base_law: RecoveredGraphLaw,
    checkpoint: GaussianCheckpoint,
    recovered: _RecoveredGaussianState,
    public_experiment_id: int,
) -> Tuple[RecoveredGraphLaw, Mapping[str, object]]:
    diagonal = np.asarray(base_law.interaction_diagonal, dtype=np.float64)
    scale_values = np.asarray((1, 2, 4), dtype=np.float64)
    rank = int(diagonal.size)
    noise_edge = float(checkpoint.operator_noise_std) * math.sqrt(
        max(float(checkpoint.basis.shape[1]), 1.0)
    )
    spectrum = np.abs(np.asarray(recovered.spectrum[:rank], dtype=np.float64))
    reliability = np.maximum(spectrum - noise_edge, 0.0) / np.maximum(
        spectrum + noise_edge, 1.0e-12
    )
    mean_reliability = float(np.mean(reliability)) if reliability.size else 0.0
    gamma = float(np.clip(0.18 * mean_reliability, 0.0, 0.18))
    normalized = np.abs(diagonal) / max(
        float(np.max(np.abs(diagonal))), 1.0e-12
    )
    evidence = np.asarray(
        [
            float(np.mean(reliability * np.exp(-scale * (1.0 - normalized))))
            if rank
            else 0.0
            for scale in scale_values
        ],
        dtype=np.float64,
    )
    posterior_mean = evidence + 1.0 / float(scale_values.size)
    posterior_mean /= max(float(np.sum(posterior_mean)), 1.0e-12)
    concentration = 8.0 + 32.0 * mean_reliability
    generator = _public_generator(int(public_experiment_id), 0x47434652)
    scale_mass = generator.dirichlet(
        np.maximum(concentration * posterior_mean, 1.0e-3)
    )
    semigroup = np.zeros(rank, dtype=np.float64)
    for weight, scale in zip(scale_mass, scale_values):
        semigroup += float(weight) * np.exp(
            -float(scale) * (1.0 - normalized)
        )
    raw_diagonal = diagonal * ((1.0 - gamma) + gamma * semigroup)

    dyad_count = 65_536
    left = generator.integers(
        0, base_law.n_nodes, size=dyad_count, dtype=np.int64
    )
    right = generator.integers(
        0, base_law.n_nodes - 1, size=dyad_count, dtype=np.int64
    )
    right += (right >= left).astype(np.int64)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    base_bilinear = np.sum(
        base_law.interaction[low]
        * diagonal[None, :]
        * base_law.interaction[high],
        axis=1,
    )
    endpoint_weight = base_law.sociability[low] * base_law.sociability[high]
    target_partition = float(np.mean(endpoint_weight * _softplus(base_bilinear)))
    rescale = _partition_rescale(
        raw_diagonal,
        base_law.interaction[low],
        base_law.interaction[high],
        endpoint_weight,
        target_partition,
    )
    transformed_diagonal = raw_diagonal * rescale

    # The promoted graph law explicitly disables endpoint-measure flow.  Keep the
    # original binary64 log/exp normalization below because even a zero mixture
    # weight has a specified rounding path in the frozen probability law.
    # Enabling this term is a new graph law, not a correctness repair.
    endpoint_strength = 0.0
    target_endpoint = np.power(
        np.maximum(base_law.noisy_degree, 0.0) + 0.5, 0.85
    )
    target_endpoint /= max(float(np.mean(target_endpoint)), 1.0e-12)
    sociability = np.exp(
        (1.0 - endpoint_strength)
        * np.log(np.maximum(base_law.sociability, 1.0e-12))
        + endpoint_strength * np.log(np.maximum(target_endpoint, 1.0e-12))
    )
    sociability *= float(np.mean(base_law.sociability)) / max(
        float(np.mean(sociability)), 1.0e-12
    )
    digest = hashlib.sha256()
    digest.update(base_law.state_hash.encode("ascii"))
    digest.update(transformed_diagonal.tobytes())
    digest.update(sociability.tobytes())
    digest.update(scale_mass.tobytes())
    law = replace(
        base_law,
        sociability=sociability,
        interaction_diagonal=transformed_diagonal,
        state_hash=digest.hexdigest(),
    )
    return law, {
        "functional_scale_mass": scale_mass.tolist(),
        "functional_active_rank": int(np.count_nonzero(reliability > 0.0)),
        "functional_mean_reliability": mean_reliability,
        "functional_strength_gamma": gamma,
        "functional_spectral_rescale": float(rescale),
        "endpoint_measure_flow_strength": endpoint_strength,
        "interaction_partition_before": target_partition,
        "private_graph_access_after_checkpoint": False,
        "full_dyad_enumeration": False,
    }


def recover_checkpoint_state(
    checkpoint_path: Path,
    *,
    public_experiment_id: int,
) -> RecoveredCheckpointState:
    """Load one checkpoint once and materialize one authoritative P2 state."""

    path = Path(checkpoint_path).resolve()
    load_started = time.perf_counter()
    checkpoint = load_gaussian_checkpoint(path)
    load_seconds = time.perf_counter() - load_started

    coherent_started = time.perf_counter()
    coherent = _recover_coherent_gaussian_base(checkpoint)
    shell = _compile_shell_posterior_state(
        coherent.raw_operator, float(checkpoint.operator_noise_std)
    )
    temperature, temperature_diagnostics = derive_source_temperature(
        checkpoint, coherent.target_edges, coherent.endpoint_target, shell
    )
    temperature_seconds = time.perf_counter() - coherent_started

    recovery_started = time.perf_counter()
    recovered, operator_diagnostics = _recover_operator_shell_state(
        checkpoint, coherent, shell
    )
    endpoint_target = coherent.endpoint_target
    base_law, base_diagnostics = _freeze_base_graph_law(
        checkpoint, recovered, endpoint_target, temperature
    )
    graph_law, functional_diagnostics = _freeze_commutant_functional(
        base_law, checkpoint, recovered, int(public_experiment_id)
    )
    degree_carrier_target, carrier_diagnostics = recover_rank_coupled_degree_carrier(
        checkpoint,
        endpoint_target,
        graph_law.sociability,
        graph_law.target_edges,
    )
    graph_law = replace(
        graph_law,
        target_endpoint_mass=np.asarray(degree_carrier_target, dtype=np.float64),
        noisy_degree=np.asarray(degree_carrier_target, dtype=np.float64),
    )
    carrier_digest = hashlib.sha256()
    carrier_digest.update(graph_law.state_hash.encode("ascii"))
    carrier_digest.update(
        np.ascontiguousarray(degree_carrier_target, dtype=np.float64).tobytes()
    )
    graph_law = replace(graph_law, state_hash=carrier_digest.hexdigest())
    recovery_seconds = time.perf_counter() - recovery_started
    diagnostics = {
        **dict(operator_diagnostics),
        **dict(base_diagnostics),
        **dict(functional_diagnostics),
        **dict(carrier_diagnostics),
        "degree_carrier": (
            "operator-orthogonal Gaussian empirical-Bayes marginal with "
            "sociability-ordered multiscale rank coupling"
        ),
        "source_temperature": "posterior-predictive formula",
        "private_graph_access_after_checkpoint": False,
        "complete_dyad_enumerated": False,
    }
    return RecoveredCheckpointState(
        checkpoint_path=path,
        checkpoint=checkpoint,
        temperature=float(temperature),
        temperature_diagnostics=dict(temperature_diagnostics),
        graph_law=graph_law,
        endpoint_target=np.asarray(degree_carrier_target, dtype=np.float64),
        degree_carrier_target=np.asarray(degree_carrier_target, dtype=np.float64),
        recovery_diagnostics=diagnostics,
        public_experiment_id=int(public_experiment_id),
        stage_seconds={
            "checkpoint_load_seconds": float(load_seconds),
            "temperature_formula_seconds": float(temperature_seconds),
            "p2_recovery_seconds": float(recovery_seconds),
        },
        audit={
            "checkpoint_load_requests": 1,
            "checkpoint_load_materializations": 1,
            "p2_recovery_requests": 1,
            "p2_recovery_materializations": 1,
        },
    )


__all__ = (
    "RecoveredCheckpointState",
    "operator_matrix_from_packed",
    "project_degree_to_mass_simplex",
    "recover_rank_coupled_degree_carrier",
    "recover_checkpoint_state",
)
