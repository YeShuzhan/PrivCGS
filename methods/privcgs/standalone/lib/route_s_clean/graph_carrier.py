"""Immutable simple-graph CSR carrier and relocation pushforward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape


def canonical_edges(edges, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.uint64)
    left = np.minimum(array[:, 0], array[:, 1])
    right = np.maximum(array[:, 0], array[:, 1])
    if (
        np.any(left < 0)
        or np.any(right >= int(n_nodes))
        or np.any(left == right)
    ):
        raise ValueError("edges must be simple dyads inside the fixed atom domain")
    keys = left.astype(np.uint64) * np.uint64(n_nodes) + right.astype(np.uint64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if sorted_keys.size > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("edge array contains duplicate dyads")
    canonical = np.column_stack((left[order], right[order])).astype(np.int64, copy=False)
    return np.ascontiguousarray(canonical), np.ascontiguousarray(sorted_keys)


def _csr(edges: np.ndarray, n_nodes: int):
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    target = np.concatenate((edges[:, 1], edges[:, 0]))
    order = np.lexsort((target, source))
    neighbors = np.ascontiguousarray(target[order], dtype=np.int64)
    degree = np.bincount(source, minlength=n_nodes).astype(np.int64)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(degree, dtype=np.int64))
    )
    return offsets, neighbors, degree


@dataclass(frozen=True)
class CSRGraph:
    n_nodes: int
    edge_array: np.ndarray
    edge_keys: np.ndarray
    csr_offsets: np.ndarray
    csr_neighbors: np.ndarray
    degree_array: np.ndarray

    @classmethod
    def from_edges(cls, edges, n_nodes: int) -> "CSRGraph":
        canonical, keys = canonical_edges(edges, int(n_nodes))
        offsets, neighbors, degree = _csr(canonical, int(n_nodes))
        return cls(
            int(n_nodes), canonical, keys, offsets, neighbors, degree
        )

    def neighbors(self, node: int) -> np.ndarray:
        index = int(node)
        if index < 0 or index >= self.n_nodes:
            raise IndexError("node is outside the fixed atom domain")
        return self.csr_neighbors[
            int(self.csr_offsets[index]) : int(self.csr_offsets[index + 1])
        ]

    @property
    def edge_count(self) -> int:
        return int(self.edge_array.shape[0])


def apply_relocation_state(
    source: CSRGraph,
    tape: PackedRelocationTape,
    state,
) -> CSRGraph:
    flow = np.asarray(state, dtype=np.int64)
    positive_mask = tape.positive_rank <= np.maximum(
        flow[tape.positive_bundle], 0
    )
    negative_mask = tape.negative_rank <= np.maximum(
        -flow[tape.negative_bundle], 0
    )
    removals = np.concatenate(
        (tape.negative_old_edge[negative_mask], tape.positive_old_edge[positive_mask]),
        axis=0,
    )
    additions = np.concatenate(
        (tape.negative_new_edge[negative_mask], tape.positive_new_edge[positive_mask]),
        axis=0,
    )
    selected_count = int(np.sum(np.abs(flow)))
    if selected_count == 0:
        return source
    _, remove_keys = canonical_edges(removals, source.n_nodes)
    add_edges, add_keys = canonical_edges(additions, source.n_nodes)
    positions = np.searchsorted(source.edge_keys, remove_keys)
    retained = np.ones(source.edge_keys.size, dtype=bool)
    retained[positions] = False
    final = CSRGraph.from_edges(
        np.concatenate((source.edge_array[retained], add_edges), axis=0),
        source.n_nodes,
    )
    return final


def graph_carrier_diagnostics(source: CSRGraph) -> Mapping[str, object]:
    return {
        "source_edge_array_bytes": int(source.edge_array.nbytes),
        "source_edge_key_bytes": int(source.edge_keys.nbytes),
        "source_csr_bytes": int(
            source.csr_offsets.nbytes
            + source.csr_neighbors.nbytes
            + source.degree_array.nbytes
        ),
        "python_adjacency_set_materialized": False,
        "dynamic_membership_query_count": 0,
        "complete_dyad_enumeration": False,
        "complete_wedge_enumeration": False,
        "complexity": "O(n+m log m) construction and O(m+R log m) immutable pushforward",
    }


__all__ = (
    "CSRGraph",
    "apply_relocation_state",
    "canonical_edges",
    "graph_carrier_diagnostics",
)
