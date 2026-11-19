"""JT034: tight (W,3T) Gaussian release, frozen legacy cap selection.

Conditional on the released degree vector, the mutual-top-D projection has
joint count sensitivity sqrt(40)*(D-1). This bound is sharp when n>=2(D+1).
The degree/operator release and cap selector remain unchanged. All previous
release objects remain distinct; new closure noise uses a separate stream.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import time
from typing import Mapping, Sequence

import numpy as np

from route_s_clean.checkpoint import (
    GaussianCheckpoint,
    operator_l2_sensitivity_from_basis,
    validate_gaussian_checkpoint,
)
from route_s_clean.checkpoint_native import validate_simple_undirected_csr
from route_s_clean.checkpoint_release import (
    _degree_orthogonal_basis,
    _public_fourier_basis,
    _validate_private_gaussian_generators,
    operator_observation_from_csr,
    rho_from_epsilon_delta,
)
from cctf_closure_checkpoint_v1 import projected_closure_counts


SCHEMA = "jt034-mutual-top-d-count40-v1"
CLOSURE_RHO_FRACTION_OF_NONDEGREE = 0.10
LOWER_CONFIDENCE_Z = 3.0
# Four-sigma protection is used because one production campaign screens many
# dataset/epsilon cells. The earlier three-sigma gate produced a wrong-direction
# Enron activation at z=3.22 in the cross-scale audit.
EVIDENCE_Z = 4.0
MIN_RETAINED_WEDGE_PROXY_FRACTION = 0.95


@dataclass(frozen=True)
class StableWedgeRatioTranscript:
    noisy_closure_ratio: float
    closure_noise_std: float
    projection_degree_cap: int
    denominator_stabilizer: float
    closure_l2_sensitivity: float
    retained_wedge_proxy_fraction: float
    rho_total: float
    rho_degree: float
    rho_operator: float
    rho_closure: float
    public_projection_seed: int
    integrity_hash: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class StableWedgeRatioRelease:
    checkpoint: GaussianCheckpoint
    transcript: StableWedgeRatioTranscript
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class StableWedgeRatioTarget:
    closure_target: float
    correction_weight: float
    identity_fallback: bool
    diagnostics: Mapping[str, object]


def closure_ratio_sensitivity(degree_cap: int, stabilizer: float) -> float:
    """Legacy direct-ratio bound retained for audit compatibility only."""
    cap = int(degree_cap)
    lower = float(stabilizer)
    if cap < 2 or not math.isfinite(lower) or lower < 1.0:
        raise ValueError("invalid stable-ratio sensitivity arguments")
    return float(min(1.0, 15.0 * float(cap - 1) / lower))


def closure_count_l2_sensitivity(degree_cap: int) -> float:
    cap=int(degree_cap)
    if cap<2:raise ValueError("projection degree cap must be at least two")
    return float(math.sqrt(40.0)*float(cap-1))


def legacy_count_l2_sensitivity(degree_cap: int) -> float:
    """Joint sensitivity of projected (wedge, triangle-incidence) counts."""

    cap = int(degree_cap)
    if cap < 2:
        raise ValueError("projection degree cap must be at least two")
    return float(math.sqrt(117.0) * float(cap - 1))


def _projected_wedge_proxy(
    noisy_degree: np.ndarray,
    degree_sigma: float,
    degree_cap: int,
    *,
    lower_confidence_z: float = LOWER_CONFIDENCE_Z,
) -> tuple[float, float, float]:
    """Return a cap-matched wedge proxy, its analytic SD, and a lower bound.

    Interior coordinates use the Gaussian debiasing term sigma^2.  At a clipping
    boundary the formula is deliberately conservative rather than claimed to be
    exactly unbiased.  This state is DP post-processing and is never a new query.
    """

    degree = np.asarray(noisy_degree, dtype=np.float64).reshape(-1)
    sigma = float(degree_sigma)
    cap = int(degree_cap)
    if not np.all(np.isfinite(degree)) or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("invalid released degree channel")
    clipped = np.clip(degree, 0.0, float(cap))
    interior = (degree > 0.0) & (degree < float(cap))
    proxy_terms = 0.5 * (
        clipped * clipped - clipped - sigma * sigma * interior.astype(np.float64)
    )
    proxy = max(float(np.sum(proxy_terms, dtype=np.float64)), 1.0)
    # Var[.5((d+Z)^2-(d+Z)-sigma^2)]
    variance_terms = np.where(
        interior,
        sigma * sigma * (clipped - 0.5) ** 2 + 0.5 * sigma**4,
        0.0,
    )
    standard_deviation = math.sqrt(max(float(np.sum(variance_terms)), 0.0))
    lower = max(proxy - float(lower_confidence_z) * standard_deviation, 1.0)
    return proxy, standard_deviation, lower


def choose_projection_cap_and_stabilizer(
    noisy_degree: np.ndarray,
    degree_sigma: float,
    rho_closure: float,
    *,
    candidate_caps: Sequence[int] | None = None,
) -> tuple[int, float, float, Mapping[str, object]]:
    """Select D using only released degree state and the actual scalar variance."""

    degree = np.asarray(noisy_degree, dtype=np.float64).reshape(-1)
    n_nodes = int(degree.size)
    rho = float(rho_closure)
    if n_nodes < 3 or not np.all(np.isfinite(degree)) or rho <= 0.0:
        raise ValueError("invalid released state for projection selection")

    if candidate_caps is None:
        # A released union-bound envelope normally includes the unprojected graph,
        # while remaining much smaller than n on sparse large graphs.
        envelope_z = math.sqrt(2.0 * math.log(max(20.0 * n_nodes, 3.0)))
        released_upper = int(
            math.ceil(float(np.max(degree, initial=2.0)) + degree_sigma * envelope_z)
        )
        upper = min(n_nodes - 1, max(8, released_upper))
        values = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, upper)
    else:
        values = tuple(int(value) for value in candidate_caps)
    caps = sorted({min(max(int(value), 2), n_nodes - 1) for value in values})

    full_proxy, full_sd, _full_lower = _projected_wedge_proxy(
        degree, degree_sigma, n_nodes - 1
    )
    records: list[dict[str, float | int]] = []
    for cap in caps:
        proxy, proxy_sd, lower = _projected_wedge_proxy(degree, degree_sigma, cap)
        retained = float(np.clip(proxy / max(full_proxy, 1.0), 0.0, 1.0))
        sensitivity = closure_count_l2_sensitivity(cap)
        noise_std = sensitivity / math.sqrt(2.0 * rho)
        tail_loss = 1.0 - retained
        relative_variance = 2.0 * (noise_std / max(proxy, 1.0)) ** 2
        score = tail_loss * tail_loss + relative_variance
        records.append(
            {
                "degree_cap": cap,
                "projected_wedge_proxy": proxy,
                "projected_wedge_proxy_sd": proxy_sd,
                "denominator_stabilizer": lower,
                "retained_wedge_proxy_fraction": retained,
                "released_tail_loss": tail_loss,
                "closure_l2_sensitivity": sensitivity,
                "predicted_count_noise_std": noise_std,
                "predicted_relative_ratio_variance_proxy": relative_variance,
                "score": score,
            }
        )
    chosen = min(records, key=lambda row: (float(row["score"]), int(row["degree_cap"])))
    return (
        int(chosen["degree_cap"]),
        float(chosen["denominator_stabilizer"]),
        float(chosen["retained_wedge_proxy_fraction"]),
        {
            "selection_rule": "released wedge-tail squared + joint-count relative variance",
            "released_full_wedge_proxy": full_proxy,
            "released_full_wedge_proxy_sd": full_sd,
            "lower_confidence_z": LOWER_CONFIDENCE_Z,
            "rho_closure": rho,
            "candidate_records": records,
            "chosen_record": chosen,
            "private_graph_access": False,
        },
    )


def _validate_three_streams(*streams: np.random.Generator) -> None:
    if len(streams) != 3:
        raise ValueError("stable-ratio release needs three Gaussian streams")
    for first in range(3):
        for second in range(first + 1, 3):
            _validate_private_gaussian_generators(streams[first], streams[second])


def _transcript_hash(base_hash: str, values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("ascii"))
    digest.update(base_hash.encode("ascii"))
    for value in values:
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def release_count40_from_csr(
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
    closure_private_rng: np.random.Generator,
    closure_rho_fraction_of_nondegree: float = CLOSURE_RHO_FRACTION_OF_NONDEGREE,
    projection_candidate_caps: Sequence[int] | None = None,
) -> StableWedgeRatioRelease:
    """Release the unchanged base and a fresh count40 closure block at total rho."""

    started = time.perf_counter()
    _validate_three_streams(
        degree_private_rng, structural_private_rng, closure_private_rng
    )
    row_offsets, column_indices = validate_simple_undirected_csr(offsets, neighbors)
    n_nodes = int(row_offsets.size - 1)
    true_degree = np.subtract(row_offsets[1:], row_offsets[:-1], dtype=np.float64)
    true_edge_count = int(column_indices.size // 2)
    degree_fraction = float(degree_rho_fraction)
    closure_fraction = float(closure_rho_fraction_of_nondegree)
    if not 0.0 < degree_fraction < 1.0 or not 0.0 < closure_fraction < 1.0:
        raise ValueError("rho fractions must be strictly interior")
    rho_total = rho_from_epsilon_delta(epsilon, delta)
    rho_degree = rho_total * degree_fraction
    rho_nondegree = rho_total - rho_degree
    rho_closure = rho_nondegree * closure_fraction
    rho_operator = rho_nondegree - rho_closure

    timer = time.perf_counter()
    degree_sigma = 1.0 / math.sqrt(rho_degree)
    continuous_degree = true_degree + degree_private_rng.normal(
        0.0, degree_sigma, size=n_nodes
    )
    activity = np.arcsinh(continuous_degree / degree_sigma)
    activity -= float(np.mean(activity))
    degree_seconds = time.perf_counter() - timer

    rank = max(1, min(n_nodes - 1, int(operator_basis_rank)))
    public_rng = np.random.default_rng(int(public_experiment_id) * 10_000 + 2003)
    basis = _degree_orthogonal_basis(
        _public_fourier_basis(n_nodes, rank, public_rng), activity
    )

    timer = time.perf_counter()
    observed = operator_observation_from_csr(row_offsets, column_indices, basis)
    operator_sensitivity = operator_l2_sensitivity_from_basis(basis)
    mass_weight = max(operator_sensitivity, np.finfo(np.float64).eps)
    joint_sensitivity = math.hypot(operator_sensitivity, mass_weight)
    operator_sigma = joint_sensitivity / math.sqrt(2.0 * rho_operator)
    joint_observation = np.concatenate(
        (observed, np.asarray([mass_weight * true_edge_count], dtype=np.float64))
    )
    joint_release = joint_observation + structural_private_rng.normal(
        0.0, operator_sigma, size=joint_observation.shape
    )
    operator_seconds = time.perf_counter() - timer

    cap, _released_proxy_lower, retained, cap_diagnostics = choose_projection_cap_and_stabilizer(
        continuous_degree,
        degree_sigma,
        rho_closure,
        candidate_caps=projection_candidate_caps,
    )
    projection_seed = int(public_experiment_id) * 10_000 + 4409
    timer = time.perf_counter()
    exact = projected_closure_counts(
        row_offsets,
        column_indices,
        degree_cap=cap,
        public_seed=projection_seed,
    )
    closure_sensitivity = closure_count_l2_sensitivity(cap)
    count_sigma = closure_sensitivity / math.sqrt(2.0 * rho_closure)
    count_noise = closure_private_rng.normal(0.0, count_sigma, size=2)
    noisy_wedge = float(exact["projected_wedge_count"] + count_noise[0])
    noisy_incidence = float(3 * exact["projected_triangle_count"] + count_noise[1])
    stabilizer = max(3.0 * count_sigma, 1.0)
    released_denominator = max(noisy_wedge, stabilizer)
    noisy_ratio = float(np.clip(noisy_incidence / released_denominator, 0.0, 1.0))
    ratio_sigma_upper = float(
        count_sigma
        * math.sqrt(1.0 + noisy_ratio * noisy_ratio)
        / max(noisy_wedge - 3.0 * count_sigma, stabilizer)
    )
    closure_seconds = time.perf_counter() - timer

    base_diagnostics = {
        "checkpoint_schema": "route-s-gaussian-checkpoint-v4-v41-swr-base",
        "rho_total": rho_degree + rho_operator,
        "rho_degree": rho_degree,
        "rho_structural": rho_operator,
        "private_graph_access_after_checkpoint": False,
        "full_transcript_required": True,
    }
    base = GaussianCheckpoint(
        n_nodes=n_nodes,
        basis=np.asarray(basis, dtype=np.float64),
        continuous_noisy_degree=np.asarray(continuous_degree, dtype=np.float64),
        raw_noisy_operator_vector=np.asarray(joint_release[:-1], dtype=np.float64),
        continuous_noisy_edge_mass=float(joint_release[-1] / mass_weight),
        mass_coordinate_weight=float(mass_weight),
        degree_sigma=float(degree_sigma),
        operator_noise_std=float(operator_sigma),
        rho_total=float(rho_degree + rho_operator),
        rho_degree=float(rho_degree),
        rho_structural=float(rho_operator),
        diagnostics=base_diagnostics,
        epsilon=None,
        delta=None,
        privacy_adjacency="edge_add_remove",
    )
    try:
        validated = validate_gaussian_checkpoint(
            base, _operator_sensitivity=operator_sensitivity
        )
    except TypeError as error:
        if "_operator_sensitivity" not in str(error):
            raise
        validated = validate_gaussian_checkpoint(base)
    if isinstance(validated, str):
        base_hash = validated
    else:
        from route_s_clean.checkpoint import checkpoint_integrity_hash

        base_hash = checkpoint_integrity_hash(base)
    base = replace(base, integrity_hash=base_hash)
    transcript_hash = _transcript_hash(
        base_hash,
        (
            noisy_ratio,
            noisy_wedge,
            noisy_incidence,
            count_sigma,
            ratio_sigma_upper,
            cap,
            stabilizer,
            closure_sensitivity,
            retained,
            rho_total,
            rho_degree,
            rho_operator,
            rho_closure,
            projection_seed,
        ),
    )
    diagnostics = {
        "schema": SCHEMA,
        "method": "MCR1-D40SCORE1 count40 release and count40 cap-selection score",
        "cap_selection_uses_frozen_legacy_117_variance": False,
        "cap_selection_variance_model": "count40",
        "n_nodes": n_nodes,
        "operator_basis_rank": rank,
        "degree_rho_fraction": degree_fraction,
        "closure_rho_fraction_of_nondegree": closure_fraction,
        "rho_total": rho_total,
        "rho_degree": rho_degree,
        "rho_operator": rho_operator,
        "rho_closure": rho_closure,
        "rho_composition_exact": bool(
            np.isclose(rho_total, rho_degree + rho_operator + rho_closure)
        ),
        "degree_noise_std": degree_sigma,
        "operator_noise_std": operator_sigma,
        "operator_noise_ratio_vs_v41": float(1.0 / math.sqrt(1.0 - closure_fraction)),
        "closure_count_noise_std": count_sigma,
        "closure_ratio_noise_std_upper": ratio_sigma_upper,
        "closure_joint_count_l2_sensitivity": closure_sensitivity,
        "projection_degree_cap": cap,
        "denominator_stabilizer": stabilizer,
        "retained_wedge_proxy_fraction": retained,
        "public_projection_seed": projection_seed,
        "noisy_closure_ratio": noisy_ratio,
        "noisy_projected_wedge_count": noisy_wedge,
        "noisy_projected_triangle_incidence": noisy_incidence,
        "degree_cap_selection": cap_diagnostics,
        "exact_private_triangle_and_wedge_counts_serialized": False,
        "degree_release_seconds": degree_seconds,
        "operator_release_seconds": operator_seconds,
        "closure_release_seconds": closure_seconds,
        "p1_seconds": time.perf_counter() - started,
        "additional_closure_rho_relative_to_legacy_transcript": rho_closure,
        "private_graph_access_after_checkpoint": False,
        "single_uniform_path": True,
        "base_checkpoint_integrity_hash": base_hash,
        "transcript_integrity_hash": transcript_hash,
    }
    transcript = StableWedgeRatioTranscript(
        noisy_closure_ratio=noisy_ratio,
        closure_noise_std=ratio_sigma_upper,
        projection_degree_cap=cap,
        denominator_stabilizer=stabilizer,
        closure_l2_sensitivity=closure_sensitivity,
        retained_wedge_proxy_fraction=retained,
        rho_total=rho_total,
        rho_degree=rho_degree,
        rho_operator=rho_operator,
        rho_closure=rho_closure,
        public_projection_seed=projection_seed,
        integrity_hash=transcript_hash,
        diagnostics=diagnostics,
    )
    return StableWedgeRatioRelease(base, transcript, diagnostics)

