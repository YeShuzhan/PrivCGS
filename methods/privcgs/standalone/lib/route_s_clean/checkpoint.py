"""Minimal sufficient-state serialization for one Gaussian Edge-DP checkpoint."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


_SCHEMA_VERSION = "route-s-gaussian-checkpoint-v4"
_LEGACY_SCHEMA_VERSIONS = {
    None,
    "route-s-gaussian-checkpoint-v2",
    "route-s-gaussian-checkpoint-v3",
}
_SENSITIVITY_WORKING_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class GaussianCheckpoint:
    """The Gaussian coordinates and privacy metadata sufficient for P2.

    GCFR-Scale 1.8.1 no longer serializes P1-derived degree graphification,
    provisional mass fusion, posterior operators, or eigensystems.  P2 recovers
    all of those quantities, when needed, from the coordinates below.
    """

    n_nodes: int
    basis: np.ndarray
    continuous_noisy_degree: np.ndarray
    raw_noisy_operator_vector: np.ndarray
    continuous_noisy_edge_mass: float
    mass_coordinate_weight: float
    degree_sigma: float
    operator_noise_std: float
    rho_total: float
    rho_degree: float
    rho_structural: float
    diagnostics: Mapping[str, object]
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    privacy_adjacency: str = "edge_add_remove"
    integrity_hash: Optional[str] = None


def operator_l2_sensitivity_from_basis(
    basis: np.ndarray,
    *,
    working_bytes: int = _SENSITIVITY_WORKING_BYTES,
) -> float:
    """Return ``2 * max_i ||basis[i]||^2`` with bounded scratch memory."""

    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("basis must be a nonempty two-dimensional array")
    rank = int(matrix.shape[1])
    bytes_per_row = max(rank * matrix.dtype.itemsize, matrix.dtype.itemsize)
    rows_per_block = max(1, int(working_bytes) // bytes_per_row)
    maximum = 0.0
    for start in range(0, int(matrix.shape[0]), rows_per_block):
        stop = min(start + rows_per_block, int(matrix.shape[0]))
        row_norm_squared = np.sum(np.square(matrix[start:stop]), axis=1)
        maximum = max(maximum, float(np.max(row_norm_squared)))
    return float(2.0 * maximum)


def _update_digest_array(
    digest: "hashlib._Hash", label: str, value, dtype
) -> None:
    array = np.ascontiguousarray(value, dtype=dtype)
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    raw = memoryview(array).cast("B")
    block = 16 * 1024 * 1024
    for start in range(0, raw.nbytes, block):
        digest.update(raw[start : start + block])


def checkpoint_integrity_hash(checkpoint: GaussianCheckpoint) -> str:
    """Hash every serialized GCFR-Scale 1.8.1 checkpoint field."""

    digest = hashlib.sha256()
    digest.update(_SCHEMA_VERSION.encode("utf-8"))
    _update_digest_array(digest, "n_nodes", [checkpoint.n_nodes], np.int64)
    _update_digest_array(digest, "basis", checkpoint.basis, np.float64)
    _update_digest_array(
        digest,
        "continuous_noisy_degree",
        checkpoint.continuous_noisy_degree,
        np.float64,
    )
    _update_digest_array(
        digest,
        "raw_noisy_operator_vector",
        checkpoint.raw_noisy_operator_vector,
        np.float64,
    )
    _update_digest_array(
        digest,
        "continuous_noisy_edge_mass",
        [checkpoint.continuous_noisy_edge_mass],
        np.float64,
    )
    _update_digest_array(
        digest,
        "mass_coordinate_weight",
        [checkpoint.mass_coordinate_weight],
        np.float64,
    )
    _update_digest_array(
        digest, "degree_sigma", [checkpoint.degree_sigma], np.float64
    )
    _update_digest_array(
        digest, "operator_noise_std", [checkpoint.operator_noise_std], np.float64
    )
    _update_digest_array(digest, "rho_total", [checkpoint.rho_total], np.float64)
    _update_digest_array(digest, "rho_degree", [checkpoint.rho_degree], np.float64)
    _update_digest_array(
        digest, "rho_structural", [checkpoint.rho_structural], np.float64
    )
    _update_digest_array(
        digest,
        "epsilon",
        [] if checkpoint.epsilon is None else [checkpoint.epsilon],
        np.float64,
    )
    _update_digest_array(
        digest,
        "delta",
        [] if checkpoint.delta is None else [checkpoint.delta],
        np.float64,
    )
    digest.update(b"privacy_adjacency\0")
    digest.update(str(checkpoint.privacy_adjacency).encode("utf-8"))
    return digest.hexdigest()


def _legacy_release_hash(payload) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(payload["noisy_degree"], dtype=np.int32).tobytes())
    digest.update(
        np.asarray(payload["continuous_noisy_degree"], dtype=np.float64).tobytes()
    )
    digest.update(
        np.asarray(payload["raw_noisy_operator_vector"], dtype=np.float64).tobytes()
    )
    digest.update(np.asarray(payload["operator"], dtype=np.float64).tobytes())
    digest.update(
        np.asarray(
            [
                int(payload["target_edges"][0]),
                float(payload["continuous_noisy_edge_mass"][0]),
            ],
            dtype=np.float64,
        ).tobytes()
    )
    return digest.hexdigest()


def _legacy_v3_integrity_hash(payload) -> str:
    """Validate a v3 transcript before discarding its derived shadow fields."""

    digest = hashlib.sha256()
    digest.update(b"route-s-gaussian-checkpoint-v3")
    _update_digest_array(digest, "n_nodes", payload["n_nodes"], np.int64)
    _update_digest_array(digest, "noisy_degree", payload["noisy_degree"], np.int64)
    _update_digest_array(digest, "target_edges", payload["target_edges"], np.int64)
    _update_digest_array(digest, "basis", payload["basis"], np.float64)
    _update_digest_array(digest, "operator", payload["operator"], np.float64)
    _update_digest_array(digest, "node_factors", payload["node_factors"], np.float64)
    _update_digest_array(digest, "spectrum", payload["spectrum"], np.float64)
    _update_digest_array(digest, "degree_sigma", payload["degree_sigma"], np.float64)
    _update_digest_array(
        digest, "operator_noise_std", payload["operator_noise_std"], np.float64
    )
    _update_digest_array(digest, "rho_total", payload["rho_total"], np.float64)
    _update_digest_array(digest, "rho_degree", payload["rho_degree"], np.float64)
    _update_digest_array(
        digest, "rho_structural", payload["rho_structural"], np.float64
    )
    _update_digest_array(
        digest,
        "continuous_noisy_degree",
        payload["continuous_noisy_degree"],
        np.float64,
    )
    _update_digest_array(
        digest,
        "raw_noisy_operator_vector",
        payload["raw_noisy_operator_vector"],
        np.float64,
    )
    _update_digest_array(
        digest,
        "continuous_noisy_edge_mass",
        payload["continuous_noisy_edge_mass"],
        np.float64,
    )
    _update_digest_array(
        digest,
        "mass_coordinate_weight",
        payload["mass_coordinate_weight"],
        np.float64,
    )
    _update_digest_array(digest, "epsilon", payload["epsilon"], np.float64)
    _update_digest_array(digest, "delta", payload["delta"], np.float64)
    digest.update(b"privacy_adjacency\0")
    digest.update(str(payload["privacy_adjacency"][0]).encode("utf-8"))
    digest.update(b"\0release_hash\0")
    digest.update(str(payload["release_hash"][0]).encode("ascii"))
    return digest.hexdigest()


def load_gaussian_checkpoint(path: Path) -> GaussianCheckpoint:
    """Load one minimal checkpoint; validated v2/v3 files migrate at the boundary."""

    with np.load(Path(path), allow_pickle=False) as payload:
        schema_version = (
            str(payload["schema_version"][0])
            if "schema_version" in payload.files
            else None
        )
        if schema_version != _SCHEMA_VERSION and schema_version not in _LEGACY_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported checkpoint schema: {schema_version!r}")

        source_integrity = (
            str(payload["integrity_hash"][0])
            if "integrity_hash" in payload.files
            else None
        )
        if schema_version == _SCHEMA_VERSION and source_integrity is None:
            raise ValueError("current checkpoint schema requires an integrity hash")

        legacy = schema_version != _SCHEMA_VERSION
        if legacy:
            required = {
                "noisy_degree",
                "target_edges",
                "operator",
                "node_factors",
                "spectrum",
                "release_hash",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"legacy checkpoint lacks fields: {missing}")
            if str(payload["release_hash"][0]) != _legacy_release_hash(payload):
                raise ValueError("legacy checkpoint release hash does not match")
            if schema_version == "route-s-gaussian-checkpoint-v3":
                if source_integrity is None:
                    raise ValueError("v3 checkpoint requires an integrity hash")
                if source_integrity != _legacy_v3_integrity_hash(payload):
                    raise ValueError("legacy checkpoint integrity hash does not match")

        epsilon = np.asarray(
            payload["epsilon"] if "epsilon" in payload.files else [],
            dtype=np.float64,
        )
        delta = np.asarray(
            payload["delta"] if "delta" in payload.files else [],
            dtype=np.float64,
        )
        checkpoint = GaussianCheckpoint(
            n_nodes=int(payload["n_nodes"][0]),
            basis=np.asarray(payload["basis"], dtype=np.float64),
            continuous_noisy_degree=np.asarray(
                payload["continuous_noisy_degree"], dtype=np.float64
            ),
            raw_noisy_operator_vector=np.asarray(
                payload["raw_noisy_operator_vector"], dtype=np.float64
            ),
            continuous_noisy_edge_mass=float(
                payload["continuous_noisy_edge_mass"][0]
            ),
            mass_coordinate_weight=float(payload["mass_coordinate_weight"][0]),
            degree_sigma=float(payload["degree_sigma"][0]),
            operator_noise_std=float(payload["operator_noise_std"][0]),
            rho_total=float(payload["rho_total"][0]),
            rho_degree=float(payload["rho_degree"][0]),
            rho_structural=float(payload["rho_structural"][0]),
            diagnostics={
                "schema_version": _SCHEMA_VERSION,
                "source_schema_version": schema_version or "legacy-unspecified",
                "legacy_shadow_fields_discarded": bool(legacy),
                "integrity_hash_present": True,
            },
            epsilon=float(epsilon[0]) if epsilon.size else None,
            delta=float(delta[0]) if delta.size else None,
            privacy_adjacency=(
                str(payload["privacy_adjacency"][0])
                if "privacy_adjacency" in payload.files
                else "edge_add_remove"
            ),
            integrity_hash=source_integrity if not legacy else None,
        )

    if legacy:
        checkpoint = replace(
            checkpoint, integrity_hash=checkpoint_integrity_hash(checkpoint)
        )
    validate_gaussian_checkpoint(checkpoint)
    return checkpoint


def validate_gaussian_checkpoint(checkpoint: GaussianCheckpoint) -> None:
    n_nodes = int(checkpoint.n_nodes)
    basis = np.asarray(checkpoint.basis, dtype=np.float64)
    if n_nodes < 2 or basis.ndim != 2 or basis.shape[0] != n_nodes:
        raise ValueError("invalid checkpoint node or basis dimensions")
    rank = int(basis.shape[1])
    if not 1 <= rank < n_nodes or not np.all(np.isfinite(basis)):
        raise ValueError("checkpoint basis must be finite with rank in [1,n)")
    expected = rank * (rank + 1) // 2
    continuous = np.asarray(checkpoint.continuous_noisy_degree, dtype=np.float64)
    raw_operator = np.asarray(checkpoint.raw_noisy_operator_vector, dtype=np.float64)
    noisy_mass = float(checkpoint.continuous_noisy_edge_mass)
    mass_weight = float(checkpoint.mass_coordinate_weight)
    if continuous.shape != (n_nodes,) or raw_operator.shape != (expected,):
        raise ValueError("checkpoint Gaussian channels have invalid shapes")
    if (
        not np.all(np.isfinite(continuous))
        or not np.all(np.isfinite(raw_operator))
        or not math.isfinite(noisy_mass)
        or not math.isfinite(mass_weight)
        or mass_weight <= 0.0
    ):
        raise ValueError("checkpoint Gaussian channels must be finite")

    degree_sigma = float(checkpoint.degree_sigma)
    operator_sigma = float(checkpoint.operator_noise_std)
    if (
        not math.isfinite(degree_sigma)
        or not math.isfinite(operator_sigma)
        or degree_sigma <= 0.0
        or operator_sigma <= 0.0
    ):
        raise ValueError("checkpoint Gaussian scales must be finite and positive")
    if checkpoint.privacy_adjacency != "edge_add_remove":
        raise ValueError("checkpoint uses an unsupported privacy adjacency")

    rho_total = float(checkpoint.rho_total)
    rho_degree = float(checkpoint.rho_degree)
    rho_structural = float(checkpoint.rho_structural)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (rho_total, rho_degree, rho_structural)
    ):
        raise ValueError("checkpoint rho allocation must be finite and positive")
    if not np.isclose(
        rho_total,
        rho_degree + rho_structural,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("checkpoint privacy allocation does not compose")
    if not np.isclose(
        degree_sigma,
        1.0 / math.sqrt(rho_degree),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("checkpoint degree scale is inconsistent with rho_degree")
    operator_sensitivity = operator_l2_sensitivity_from_basis(basis)
    if not np.isclose(
        mass_weight, operator_sensitivity, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("checkpoint mass weight is inconsistent with its public basis")
    expected_operator_sigma = math.hypot(
        operator_sensitivity, mass_weight
    ) / math.sqrt(2.0 * rho_structural)
    if not np.isclose(
        operator_sigma, expected_operator_sigma, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("checkpoint structural scale is inconsistent with rho_structural")

    if (checkpoint.epsilon is None) != (checkpoint.delta is None):
        raise ValueError("checkpoint epsilon and delta must be present together")
    if checkpoint.epsilon is not None:
        epsilon = float(checkpoint.epsilon)
        delta = float(checkpoint.delta)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("checkpoint epsilon must be finite and positive")
        if not math.isfinite(delta) or not 0.0 < delta < 1.0:
            raise ValueError("checkpoint delta must lie in (0,1)")
        logarithm = math.log(1.0 / delta)
        expected_rho = (
            math.sqrt(logarithm + epsilon) - math.sqrt(logarithm)
        ) ** 2
        if not np.isclose(
            rho_total, expected_rho, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError("checkpoint rho_total is inconsistent with epsilon and delta")

    if checkpoint.integrity_hash is not None and (
        str(checkpoint.integrity_hash) != checkpoint_integrity_hash(checkpoint)
    ):
        raise ValueError("checkpoint integrity hash does not match its serialized fields")


def save_gaussian_checkpoint(path: Path, checkpoint: GaussianCheckpoint) -> None:
    """Persist one immutable detached minimal checkpoint."""

    validate_gaussian_checkpoint(checkpoint)
    integrity = checkpoint_integrity_hash(checkpoint)
    if checkpoint.integrity_hash is not None and checkpoint.integrity_hash != integrity:
        raise ValueError("checkpoint integrity hash changed before serialization")
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([_SCHEMA_VERSION]),
            n_nodes=np.asarray([checkpoint.n_nodes], dtype=np.int64),
            basis=np.asarray(checkpoint.basis, dtype=np.float64),
            continuous_noisy_degree=np.asarray(
                checkpoint.continuous_noisy_degree, dtype=np.float64
            ),
            raw_noisy_operator_vector=np.asarray(
                checkpoint.raw_noisy_operator_vector, dtype=np.float64
            ),
            continuous_noisy_edge_mass=np.asarray(
                [checkpoint.continuous_noisy_edge_mass], dtype=np.float64
            ),
            mass_coordinate_weight=np.asarray(
                [checkpoint.mass_coordinate_weight], dtype=np.float64
            ),
            degree_sigma=np.asarray([checkpoint.degree_sigma], dtype=np.float64),
            operator_noise_std=np.asarray(
                [checkpoint.operator_noise_std], dtype=np.float64
            ),
            rho_total=np.asarray([checkpoint.rho_total], dtype=np.float64),
            rho_degree=np.asarray([checkpoint.rho_degree], dtype=np.float64),
            rho_structural=np.asarray(
                [checkpoint.rho_structural], dtype=np.float64
            ),
            epsilon=(
                np.asarray([checkpoint.epsilon], dtype=np.float64)
                if checkpoint.epsilon is not None
                else np.asarray([], dtype=np.float64)
            ),
            delta=(
                np.asarray([checkpoint.delta], dtype=np.float64)
                if checkpoint.delta is not None
                else np.asarray([], dtype=np.float64)
            ),
            privacy_adjacency=np.asarray([checkpoint.privacy_adjacency]),
            integrity_hash=np.asarray([integrity]),
        )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.link(temporary, target)
        temporary.unlink()
    except BaseException:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise


__all__ = (
    "GaussianCheckpoint",
    "checkpoint_integrity_hash",
    "load_gaussian_checkpoint",
    "operator_l2_sensitivity_from_basis",
    "save_gaussian_checkpoint",
    "validate_gaussian_checkpoint",
)
