"""CCTF-V1: the minimal closure-calibrated extension of the V41 checkpoint.

The V41 degree channel is unchanged.  The former structural rho block is split
once into an operator block and a two-coordinate closure block; their sum is
the original structural rho.  The closure statistic is evaluated on a stable
mutual-top-D projection, where D is selected only from the released degree
vector.  No private graph access occurs after this function returns.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from route_s_clean.checkpoint import (
    GaussianCheckpoint,
    operator_l2_sensitivity_from_basis,
    validate_gaussian_checkpoint,
)
from route_s_clean.checkpoint_native import validate_simple_undirected_csr
from route_s_clean.checkpoint_release import (
    CheckpointRelease,
    _degree_orthogonal_basis,
    _public_fourier_basis,
    _validate_private_gaussian_generators,
    operator_observation_from_csr,
    rho_from_epsilon_delta,
)


SCHEMA = "e-ncufd-v41-cctf-checkpoint-v1"
CLOSURE_RHO_FRACTION_OF_STRUCTURAL = 0.10
_SOURCE = Path(__file__).with_name("cctf_closure_native_v1.cpp")
_NATIVE = None


@dataclass(frozen=True)
class CCTFTranscript:
    noisy_projected_wedge_count: float
    noisy_projected_triangle_incidence: float
    projection_degree_cap: int
    closure_noise_std: float
    closure_l2_sensitivity: float
    rho_total: float
    rho_degree: float
    rho_operator: float
    rho_closure: float
    retained_wedge_proxy_fraction: float
    public_projection_seed: int
    integrity_hash: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class CCTFRelease:
    checkpoint: GaussianCheckpoint
    transcript: CCTFTranscript
    diagnostics: Mapping[str, object]


def closure_l2_sensitivity(degree_cap: int) -> float:
    """Sensitivity of (projected wedges, 3 * projected triangles).

    One source-edge edit changes a mutual-top-D projection by at most three
    projected edges.  One projected-edge edit changes wedge mass by at most
    2(D-1) and triangle-incidence mass by at most 3(D-1).  The joint L2 bound is
    therefore sqrt((6(D-1))^2 + (9(D-1))^2).
    """

    cap = int(degree_cap)
    if cap < 2:
        raise ValueError("CCTF projection degree cap must be at least two")
    return float(math.sqrt(117.0) * (cap - 1))


def _load_native():
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE
    from route_s_clean._native_build import build_shared_library

    target = build_shared_library(
        "e-ncufd-v41-cctf-closure-native-v1",
        _SOURCE,
        compiler_flags=(
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-shared",
            "-static-libgcc",
            "-static-libstdc++",
        ),
    )
    library = ctypes.CDLL(str(target))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    function = library.cctf_closure_counts_v1
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int64,
        i64,
        i64,
        ctypes.c_int,
        ctypes.c_uint64,
        i64,
    ]
    function.restype = ctypes.c_int
    _NATIVE = function
    return function


def projected_closure_counts(
    offsets,
    neighbors,
    *,
    degree_cap: int,
    public_seed: int,
) -> Mapping[str, int]:
    row_offsets, column_indices = validate_simple_undirected_csr(offsets, neighbors)
    n_nodes = int(row_offsets.size - 1)
    cap = int(degree_cap)
    if cap < 2 or cap >= n_nodes:
        raise ValueError("CCTF projection cap must lie in [2,n)")
    counters = np.zeros(5, dtype=np.int64)
    status = int(
        _load_native()(
            n_nodes,
            int(column_indices.size),
            row_offsets,
            column_indices,
            cap,
            int(public_seed) & ((1 << 64) - 1),
            counters,
        )
    )
    if status:
        raise RuntimeError(f"CCTF native closure counter returned status {status}")
    edges, triangles, wedges, maximum_degree, discarded = (
        int(value) for value in counters
    )
    if maximum_degree > cap or 3 * triangles > wedges:
        raise ArithmeticError("CCTF projected closure certificate failed")
    return {
        "projected_edge_count": edges,
        "projected_triangle_count": triangles,
        "projected_wedge_count": wedges,
        "projected_maximum_degree": maximum_degree,
        "discarded_edge_count": discarded,
    }


def choose_projection_cap(
    continuous_noisy_degree: np.ndarray,
    *,
    rho_closure: float,
    candidate_caps: Sequence[int] | None = None,
) -> tuple[int, float, Mapping[str, object]]:
    """Minimize a released-only wedge-tail plus relative-noise proxy."""

    degree = np.asarray(continuous_noisy_degree, dtype=np.float64).reshape(-1)
    n_nodes = int(degree.size)
    if n_nodes < 3 or not np.all(np.isfinite(degree)):
        raise ValueError("CCTF requires at least three finite released degrees")
    estimate = np.clip(np.rint(np.maximum(degree, 0.0)), 0, n_nodes - 1)
    full_wedge = float(np.sum(estimate * np.maximum(estimate - 1.0, 0.0) / 2.0))
    denominator = max(full_wedge, 1.0)
    if candidate_caps is None:
        upper = min(n_nodes - 1, max(8, int(np.max(estimate, initial=2.0))))
        values = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, upper)
    else:
        values = tuple(int(value) for value in candidate_caps)
    caps = sorted({max(2, min(n_nodes - 1, value)) for value in values})
    records: list[dict[str, float | int]] = []
    for cap in caps:
        clipped = np.minimum(estimate, float(cap))
        retained = float(np.sum(clipped * np.maximum(clipped - 1.0, 0.0) / 2.0))
        retained_fraction = float(np.clip(retained / denominator, 0.0, 1.0))
        tail_fraction = 1.0 - retained_fraction
        sensitivity = closure_l2_sensitivity(cap)
        sigma = sensitivity / math.sqrt(2.0 * float(rho_closure))
        relative_variance = 2.0 * (sigma / max(retained, 1.0)) ** 2
        score = tail_fraction * tail_fraction + relative_variance
        records.append(
            {
                "degree_cap": int(cap),
                "retained_wedge_proxy": retained,
                "retained_wedge_proxy_fraction": retained_fraction,
                "released_tail_wedge_fraction": tail_fraction,
                "relative_noise_variance_proxy": relative_variance,
                "score": score,
            }
        )
    chosen = min(records, key=lambda row: (float(row["score"]), int(row["degree_cap"])))
    return int(chosen["degree_cap"]), float(chosen["retained_wedge_proxy_fraction"]), {
        "selection_rule": "released wedge-tail squared + two-coordinate relative variance",
        "released_total_wedge_proxy": full_wedge,
        "candidate_records": records,
        "chosen_record": chosen,
        "private_graph_access": False,
    }


def _validate_three_streams(*streams: np.random.Generator) -> None:
    if len(streams) != 3:
        raise ValueError("CCTF needs exactly three private Gaussian streams")
    for first in range(3):
        for second in range(first + 1, 3):
            _validate_private_gaussian_generators(streams[first], streams[second])


def _transcript_hash(base_hash: str, values: Sequence[float | int]) -> str:
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("ascii"))
    digest.update(base_hash.encode("ascii"))
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def release_cctf_checkpoint_from_csr(
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
    closure_rho_fraction_of_structural: float = CLOSURE_RHO_FRACTION_OF_STRUCTURAL,
    projection_candidate_caps: Sequence[int] | None = None,
) -> CCTFRelease:
    """Release the single logical V41-CCTF checkpoint under unchanged total rho."""

    started = time.perf_counter()
    _validate_three_streams(
        degree_private_rng, structural_private_rng, closure_private_rng
    )
    row_offsets, column_indices = validate_simple_undirected_csr(offsets, neighbors)
    n_nodes = int(row_offsets.size - 1)
    if n_nodes < 3:
        raise ValueError("CCTF requires at least three nodes")
    true_degree = np.subtract(row_offsets[1:], row_offsets[:-1], dtype=np.float64)
    true_edge_count = int(column_indices.size // 2)
    degree_fraction = float(degree_rho_fraction)
    closure_fraction = float(closure_rho_fraction_of_structural)
    if not 0.0 < degree_fraction < 1.0 or not 0.0 < closure_fraction < 1.0:
        raise ValueError("CCTF rho fractions must be strictly interior")
    rho_total = rho_from_epsilon_delta(epsilon, delta)
    rho_degree = rho_total * degree_fraction
    rho_nondegree = rho_total - rho_degree
    rho_closure = rho_nondegree * closure_fraction
    rho_operator = rho_nondegree - rho_closure

    degree_started = time.perf_counter()
    degree_sigma = 1.0 / math.sqrt(rho_degree)
    continuous_degree = true_degree + degree_private_rng.normal(
        0.0, degree_sigma, size=n_nodes
    )
    activity = np.arcsinh(continuous_degree / degree_sigma)
    activity -= float(np.mean(activity))
    degree_seconds = time.perf_counter() - degree_started

    rank = max(1, min(n_nodes - 1, int(operator_basis_rank)))
    public_rng = np.random.default_rng(int(public_experiment_id) * 10_000 + 2003)
    basis = _degree_orthogonal_basis(
        _public_fourier_basis(n_nodes, rank, public_rng), activity
    )
    operator_started = time.perf_counter()
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
    operator_seconds = time.perf_counter() - operator_started

    cap, retained_fraction, cap_diagnostics = choose_projection_cap(
        continuous_degree,
        rho_closure=rho_closure,
        candidate_caps=projection_candidate_caps,
    )
    projection_seed = int(public_experiment_id) * 10_000 + 4409
    closure_started = time.perf_counter()
    exact = projected_closure_counts(
        row_offsets,
        column_indices,
        degree_cap=cap,
        public_seed=projection_seed,
    )
    closure_sensitivity = closure_l2_sensitivity(cap)
    closure_sigma = closure_sensitivity / math.sqrt(2.0 * rho_closure)
    closure_observation = np.asarray(
        [
            float(exact["projected_wedge_count"]),
            float(3 * exact["projected_triangle_count"]),
        ],
        dtype=np.float64,
    )
    closure_release = closure_observation + closure_private_rng.normal(
        0.0, closure_sigma, size=2
    )
    closure_seconds = time.perf_counter() - closure_started

    base_diagnostics = {
        "checkpoint_schema": "route-s-gaussian-checkpoint-v4-cctf-base",
        "rho_total": rho_degree + rho_operator,
        "rho_degree": rho_degree,
        "rho_structural": rho_operator,
        "private_graph_access_after_checkpoint": False,
        "cctf_combined_checkpoint": True,
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
    base_hash = validate_gaussian_checkpoint(
        base, _operator_sensitivity=operator_sensitivity
    )
    base = replace(base, integrity_hash=base_hash)
    transcript_hash = _transcript_hash(
        base_hash,
        (
            float(closure_release[0]),
            float(closure_release[1]),
            cap,
            closure_sigma,
            closure_sensitivity,
            rho_total,
            rho_degree,
            rho_operator,
            rho_closure,
            retained_fraction,
            projection_seed,
        ),
    )
    diagnostics = {
        "schema": SCHEMA,
        "method": "V41 minimal checkpoint-calibrated triangle fiber",
        "n_nodes": n_nodes,
        "operator_basis_rank": rank,
        "degree_rho_fraction": degree_fraction,
        "closure_rho_fraction_of_structural": closure_fraction,
        "rho_total": rho_total,
        "rho_degree": rho_degree,
        "rho_operator": rho_operator,
        "rho_closure": rho_closure,
        "rho_composition_exact": bool(
            np.isclose(rho_total, rho_degree + rho_operator + rho_closure)
        ),
        "degree_noise_std": degree_sigma,
        "operator_noise_std": operator_sigma,
        "closure_noise_std": closure_sigma,
        "closure_l2_sensitivity": closure_sensitivity,
        "projection_degree_cap": cap,
        "retained_wedge_proxy_fraction": retained_fraction,
        "public_projection_seed": projection_seed,
        "noisy_projected_wedge_count": float(closure_release[0]),
        "noisy_projected_triangle_incidence": float(closure_release[1]),
        "degree_cap_selection": cap_diagnostics,
        "degree_release_seconds": degree_seconds,
        "operator_release_seconds": operator_seconds,
        "closure_release_seconds": closure_seconds,
        "p1_seconds": time.perf_counter() - started,
        "base_checkpoint_integrity_hash": base_hash,
        "cctf_integrity_hash": transcript_hash,
        "probability_mechanism": "three-block Gaussian zCDP with unchanged total rho",
        "privacy_adjacency": "edge_add_remove",
        "private_graph_access_after_checkpoint": False,
        "exact_private_closure_counts_serialized": False,
        "single_uniform_path": True,
    }
    transcript = CCTFTranscript(
        noisy_projected_wedge_count=float(closure_release[0]),
        noisy_projected_triangle_incidence=float(closure_release[1]),
        projection_degree_cap=cap,
        closure_noise_std=closure_sigma,
        closure_l2_sensitivity=closure_sensitivity,
        rho_total=rho_total,
        rho_degree=rho_degree,
        rho_operator=rho_operator,
        rho_closure=rho_closure,
        retained_wedge_proxy_fraction=retained_fraction,
        public_projection_seed=projection_seed,
        integrity_hash=transcript_hash,
        diagnostics=diagnostics,
    )
    return CCTFRelease(checkpoint=base, transcript=transcript, diagnostics=diagnostics)


__all__ = (
    "CCTFRelease",
    "CCTFTranscript",
    "CLOSURE_RHO_FRACTION_OF_STRUCTURAL",
    "SCHEMA",
    "choose_projection_cap",
    "closure_l2_sensitivity",
    "projected_closure_counts",
    "release_cctf_checkpoint_from_csr",
)
