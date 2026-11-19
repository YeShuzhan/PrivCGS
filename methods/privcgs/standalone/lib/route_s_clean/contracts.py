"""Semantic data contracts shared by the GCFR-Scale stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class PipelineConfig:
    epsilon: float
    delta: float
    public_experiment_id: int
    degree_rho_fraction: float
    operator_basis_rank: int


@dataclass(frozen=True)
class RecoveredGraphLaw:
    """Normalized finite graph law reconstructed solely from a DP checkpoint."""

    target_edges: int
    degree_noise_std: float
    temperature: float
    target_endpoint_mass: np.ndarray
    sociability: np.ndarray
    star: np.ndarray
    interaction: np.ndarray
    interaction_diagonal: np.ndarray
    dust: float
    parent_fiber: np.ndarray
    component_fiber: np.ndarray
    marked_shift: np.ndarray
    component_shift: np.ndarray
    parent_flux: np.ndarray
    phase_interaction: Optional[np.ndarray] = None
    noisy_degree: Optional[np.ndarray] = None
    teacher_confidence: Optional[np.ndarray] = None
    state_hash: str = ""

    @property
    def n_nodes(self) -> int:
        return int(self.sociability.size)

    @property
    def rank(self) -> int:
        return int(self.interaction.shape[1])


@dataclass(frozen=True)
class SourceOrderTree:
    """Flat sufficient tree used by the exact sparse source-order race."""

    child_left: np.ndarray
    child_right: np.ndarray
    subtree_size: np.ndarray
    singleton_vertex: np.ndarray
    coordinate_low: np.ndarray
    coordinate_high: np.ndarray
    sociability_max: np.ndarray
    star_max: np.ndarray
    parent_mask: np.ndarray
    depth: int


@dataclass(frozen=True)
class PackedRelocationTape:
    bundle_count: int
    path_left: np.ndarray
    path_right: np.ndarray
    negative_offsets: np.ndarray
    negative_bundle: np.ndarray
    negative_rank: np.ndarray
    negative_old_edge: np.ndarray
    negative_new_edge: np.ndarray
    negative_energy: np.ndarray
    positive_offsets: np.ndarray
    positive_bundle: np.ndarray
    positive_rank: np.ndarray
    positive_old_edge: np.ndarray
    positive_new_edge: np.ndarray
    positive_energy: np.ndarray
    state_offsets: np.ndarray
    state_value: np.ndarray
    structural_gain: np.ndarray


@dataclass(frozen=True)
class StageTimings:
    checkpoint_release_seconds: float
    law_recovery_seconds: float
    source_order_seconds: float
    degree_chain_compile_seconds: float
    degree_chain_draw_seconds: float
    phase_compile_seconds: float
    terminal_seconds: float


@dataclass(frozen=True)
class EndToEndResult:
    edge_array: np.ndarray
    checkpoint_path: Path
    recovered_state_hash: str
    timings: StageTimings
    diagnostics: Mapping[str, object]
