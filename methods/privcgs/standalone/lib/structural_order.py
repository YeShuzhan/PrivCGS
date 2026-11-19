from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


def source_rank_direct(law: Any) -> np.ndarray:
    node_count = int(law.n_nodes)
    latent_rank = int(law.rank)
    interaction = np.asarray(law.interaction, dtype=np.float64)
    weights = np.arange(1, latent_rank + 1, dtype=np.float64)
    projection = interaction @ (weights / np.linalg.norm(weights))
    phase = law.phase_interaction
    if phase is not None and phase.shape[1]:
        phase = np.asarray(phase, dtype=np.float64)
        phase_weights = np.arange(1, phase.shape[1] + 1, dtype=np.float64)
        projection += phase @ (phase_weights / np.linalg.norm(phase_weights))
    vertex = np.arange(node_count, dtype=np.int64)
    order = np.lexsort(
        (
            vertex,
            projection,
            np.asarray(law.component_fiber, dtype=np.int32),
            np.asarray(law.parent_fiber, dtype=np.int32),
            np.asarray(law.sociability, dtype=np.float64),
        )
    ).astype(np.int64, copy=False)
    rank = np.empty(node_count, dtype=np.int64)
    rank[order] = vertex
    return rank


def structural_order(f5_degree: np.ndarray, law: Any, source_rank: np.ndarray) -> np.ndarray:
    vertex = np.arange(f5_degree.size, dtype=np.int64)
    return np.lexsort(
        (
            vertex,
            source_rank,
            np.asarray(law.component_fiber, dtype=np.int32),
            np.asarray(law.parent_fiber, dtype=np.int32),
            f5_degree,
        )
    ).astype(np.int64, copy=False)
