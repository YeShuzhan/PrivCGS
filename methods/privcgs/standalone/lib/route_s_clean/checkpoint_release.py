"""One-shot Gaussian Edge-DP release from an undirected CSR graph.

The private graph is touched only in this module.  It releases a degree block
and one joint operator-plus-mass block under additive zCDP composition; every
later GCFR-Scale stage is postprocessing of the returned checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.sparse import csr_matrix

from .checkpoint import (
    GaussianCheckpoint,
    checkpoint_integrity_hash,
    operator_l2_sensitivity_from_basis,
    validate_gaussian_checkpoint,
)
from .checkpoint_native import validate_simple_undirected_csr


@dataclass(frozen=True)
class CheckpointRelease:
    checkpoint: GaussianCheckpoint
    diagnostics: Mapping[str, object]


def rho_from_epsilon_delta(epsilon: float, delta: float) -> float:
    """Largest rho whose standard zCDP conversion fits (epsilon, delta)."""

    epsilon_value = float(epsilon)
    delta_value = float(delta)
    if (
        not math.isfinite(epsilon_value)
        or not math.isfinite(delta_value)
        or epsilon_value <= 0.0
        or not 0.0 < delta_value < 1.0
    ):
        raise ValueError("epsilon must be positive and delta must lie in (0,1)")
    logarithm = math.log(1.0 / delta_value)
    return float(
        (math.sqrt(logarithm + epsilon_value) - math.sqrt(logarithm)) ** 2
    )


def split_rho(rho_total: float, degree_fraction: float) -> tuple[float, float]:
    total = float(rho_total)
    fraction = float(degree_fraction)
    if (
        not math.isfinite(total)
        or not math.isfinite(fraction)
        or total <= 0.0
        or not 0.0 < fraction < 1.0
    ):
        raise ValueError("rho and its degree fraction must be strictly interior")
    return float(total * fraction), float(total * (1.0 - fraction))


def _public_fourier_basis(
    n_nodes: int, rank: int, public_rng: np.random.Generator
) -> np.ndarray:
    latent = (
        np.arange(int(n_nodes), dtype=np.float64) + public_rng.random(int(n_nodes))
    ) / float(int(n_nodes) + 1)
    columns = [np.ones(int(n_nodes), dtype=np.float64)]
    frequency = 1
    while len(columns) < int(rank):
        angle = 2.0 * math.pi * float(frequency) * latent
        columns.append(math.sqrt(2.0) * np.cos(angle))
        if len(columns) < int(rank):
            columns.append(math.sqrt(2.0) * np.sin(angle))
        frequency += 1
    raw = np.column_stack(columns[: int(rank)])
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return np.asarray(basis[:, : int(rank)], dtype=np.float64)


def _degree_orthogonal_basis(basis: np.ndarray, activity: np.ndarray) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float64)
    anchor = np.asarray(activity, dtype=np.float64)
    centered = anchor - float(np.mean(anchor)) if anchor.size else anchor
    denominator = float(centered @ centered)
    if denominator <= np.finfo(np.float64).eps:
        return matrix
    adjusted = matrix - np.outer(centered, (centered @ matrix) / denominator)
    norms = np.linalg.norm(adjusted, axis=0)
    safe = norms > 1.0e-10
    adjusted[:, safe] /= norms[safe]
    adjusted[:, ~safe] = matrix[:, ~safe]
    orthogonal, _ = np.linalg.qr(adjusted, mode="reduced")
    return np.asarray(orthogonal[:, : matrix.shape[1]], dtype=np.float64)


def operator_observation_from_csr(
    offsets: np.ndarray, neighbors: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Packed edge-additive statistic via B.T @ A @ B, without edge features."""

    matrix = np.asarray(basis, dtype=np.float64)
    n_nodes, rank = matrix.shape
    adjacency = csr_matrix(
        (
            np.ones(np.asarray(neighbors).size, dtype=np.float64),
            np.asarray(neighbors, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
        ),
        shape=(n_nodes, n_nodes),
    )
    gram = np.asarray(matrix.T @ (adjacency @ matrix), dtype=np.float64)
    packed = np.empty(rank * (rank + 1) // 2, dtype=np.float64)
    cursor = 0
    for left in range(rank):
        packed[cursor] = 0.5 * gram[left, left]
        cursor += 1
        for right in range(left + 1, rank):
            packed[cursor] = gram[left, right]
            cursor += 1
    return packed


def _rng_state_fingerprint(generator: np.random.Generator) -> str:
    """Hash a NumPy bit-generator state without consuming private randomness."""

    digest = hashlib.sha256()

    def update(value) -> None:
        if isinstance(value, Mapping):
            digest.update(b"{")
            for key in sorted(value, key=str):
                digest.update(str(key).encode("utf-8"))
                update(value[key])
            digest.update(b"}")
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(b"[")
            for item in value:
                update(item)
            digest.update(b"]")
        else:
            digest.update(repr(value).encode("utf-8"))

    update(generator.bit_generator.state)
    return digest.hexdigest()


def _validate_private_gaussian_generators(
    degree_private_rng: np.random.Generator,
    structural_private_rng: np.random.Generator,
) -> None:
    """Reject the reproducible identical-state failure of two Gaussian blocks.

    Reusing one generator object is safe because the blocks consume disjoint
    consecutive draws.  Two distinct objects at the same state instead repeat
    the same standard-normal coordinates and invalidate the additive zCDP
    accountant through noise cancellation.
    """

    if not isinstance(degree_private_rng, np.random.Generator) or not isinstance(
        structural_private_rng, np.random.Generator
    ):
        raise TypeError("private Gaussian streams must be NumPy Generator objects")
    if degree_private_rng is not structural_private_rng and (
        _rng_state_fingerprint(degree_private_rng)
        == _rng_state_fingerprint(structural_private_rng)
    ):
        raise ValueError(
            "the two private Gaussian generators have identical states; use one "
            "shared generator or independently spawned substreams"
        )


def release_gaussian_checkpoint_from_csr(
    offsets,
    neighbors,
    *,
    epsilon: float,
    delta: float,
    public_experiment_id: int,
    degree_rho_fraction: float,
    operator_basis_rank: int,
    degree_private_rng: np.random.Generator,
    structural_private_rng: np.random.Generator,
) -> CheckpointRelease:
    """Release exactly one detached Gaussian checkpoint from the private CSR."""

    started = time.perf_counter()
    _validate_private_gaussian_generators(
        degree_private_rng, structural_private_rng
    )
    row_offsets, column_indices = validate_simple_undirected_csr(offsets, neighbors)
    n_nodes = int(row_offsets.size - 1)
    true_degree = np.diff(row_offsets).astype(np.float64, copy=False)
    true_edge_count = int(column_indices.size // 2)
    rank = max(1, min(n_nodes - 1, int(operator_basis_rank)))
    rho_total = rho_from_epsilon_delta(epsilon, delta)
    rho_degree, rho_structural = split_rho(rho_total, degree_rho_fraction)

    degree_started = time.perf_counter()
    degree_sigma = 1.0 / math.sqrt(rho_degree)
    continuous_degree = true_degree + degree_private_rng.normal(
        0.0, degree_sigma, size=n_nodes
    )
    activity = np.arcsinh(continuous_degree / degree_sigma)
    activity -= float(np.mean(activity))
    degree_seconds = time.perf_counter() - degree_started

    structural_started = time.perf_counter()
    public_rng = np.random.default_rng(int(public_experiment_id) * 10_000 + 2003)
    basis = _degree_orthogonal_basis(
        _public_fourier_basis(n_nodes, rank, public_rng), activity
    )
    observation_started = time.perf_counter()
    observed = operator_observation_from_csr(row_offsets, column_indices, basis)
    observation_seconds = time.perf_counter() - observation_started
    operator_sensitivity = operator_l2_sensitivity_from_basis(basis)
    mass_weight = max(operator_sensitivity, np.finfo(np.float64).eps)
    joint_sensitivity = math.hypot(operator_sensitivity, mass_weight)
    operator_noise_std = joint_sensitivity / math.sqrt(2.0 * rho_structural)
    joint_observation = np.concatenate(
        (observed, np.asarray([mass_weight * true_edge_count], dtype=np.float64))
    )
    joint_release = joint_observation + structural_private_rng.normal(
        0.0, operator_noise_std, size=joint_observation.shape
    )
    raw_operator = np.asarray(joint_release[:-1], dtype=np.float64)
    noisy_mass = float(joint_release[-1] / mass_weight)
    structural_seconds = time.perf_counter() - structural_started

    diagnostics = {
        "p1_seconds": float(time.perf_counter() - started),
        "degree_release_seconds": float(degree_seconds),
        "structural_release_seconds": float(structural_seconds),
        "sparse_operator_observation_seconds": float(observation_seconds),
        "n_nodes": n_nodes,
        "input_edge_count": true_edge_count,
        "operator_basis_rank": rank,
        "operator_coordinate_count": int(rank * (rank + 1) // 2),
        "checkpoint_schema": "route-s-gaussian-checkpoint-v4",
        "serialized_gaussian_coordinate_count": int(
            n_nodes + rank * (rank + 1) // 2 + 1
        ),
        "p1_derived_shadow_field_count": 0,
        "graphical_degree_materialized": False,
        "p1_target_edges_materialized": False,
        "p1_posterior_operator_materialized": False,
        "p1_eigensystem_materialized": False,
        "degree_l2_sensitivity": math.sqrt(2.0),
        "operator_l2_sensitivity": operator_sensitivity,
        "mass_coordinate_l2_sensitivity": mass_weight,
        "joint_structural_l2_sensitivity": joint_sensitivity,
        "privacy_adjacency": "edge_add_remove",
        "probability_mechanism": "two-block Gaussian zCDP checkpoint",
        "rho_total": rho_total,
        "rho_degree": rho_degree,
        "rho_structural": rho_structural,
        "zcdp_implied_epsilon": float(
            rho_total + 2.0 * math.sqrt(rho_total * math.log(1.0 / delta))
        ),
        "single_checkpoint": True,
        "additional_private_query": False,
        "private_graph_access_after_checkpoint": False,
        "edge_feature_table_materialized": False,
        "complete_dyad_enumerated": False,
    }
    checkpoint = GaussianCheckpoint(
        n_nodes=n_nodes,
        basis=np.asarray(basis, dtype=np.float64),
        continuous_noisy_degree=np.asarray(continuous_degree, dtype=np.float64),
        raw_noisy_operator_vector=raw_operator,
        continuous_noisy_edge_mass=noisy_mass,
        mass_coordinate_weight=mass_weight,
        degree_sigma=degree_sigma,
        operator_noise_std=operator_noise_std,
        rho_total=rho_total,
        rho_degree=rho_degree,
        rho_structural=rho_structural,
        diagnostics=diagnostics,
        epsilon=float(epsilon),
        delta=float(delta),
        privacy_adjacency="edge_add_remove",
    )
    integrity_hash = checkpoint_integrity_hash(checkpoint)
    diagnostics = {
        **diagnostics,
        "checkpoint_integrity_hash": integrity_hash,
    }
    checkpoint = replace(
        checkpoint,
        diagnostics=diagnostics,
        integrity_hash=integrity_hash,
    )
    validate_gaussian_checkpoint(checkpoint)
    return CheckpointRelease(checkpoint=checkpoint, diagnostics=diagnostics)


__all__ = (
    "CheckpointRelease",
    "operator_observation_from_csr",
    "release_gaussian_checkpoint_from_csr",
    "rho_from_epsilon_delta",
    "split_rho",
)
