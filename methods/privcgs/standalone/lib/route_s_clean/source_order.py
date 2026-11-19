"""Exact sparse source-order stage for the GCFR-Scale pipeline."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Mapping, Tuple

import numpy as np

from .contracts import RecoveredGraphLaw, SourceOrderTree
from .source_order_native import load_source_order_library


def _root_path_hash() -> int:
    value = 0x9E3779B9
    value ^= 0x9E3779B9 + ((value << 6) & 0xFFFFFFFF) + (value >> 2)
    return value & 0xFFFFFFFF


def _as_array(value, dtype, *, ndim: int, name: str) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=dtype)


def build_source_order_tree(
    law: RecoveredGraphLaw,
) -> Tuple[SourceOrderTree, Mapping[str, object]]:
    """Build the exact flat sufficient tree without Python node objects."""

    n_nodes = law.n_nodes
    rank = law.rank
    interaction = _as_array(law.interaction, np.float64, ndim=2, name="interaction")
    sociability = _as_array(law.sociability, np.float64, ndim=1, name="sociability")
    star = _as_array(law.star, np.float64, ndim=1, name="star")
    parent_fiber = _as_array(law.parent_fiber, np.int32, ndim=1, name="parent_fiber")
    component_fiber = _as_array(
        law.component_fiber, np.int32, ndim=1, name="component_fiber"
    )

    projection_weights = np.arange(1, rank + 1, dtype=np.float64)
    projection = interaction @ (projection_weights / np.linalg.norm(projection_weights))
    if law.phase_interaction is not None and law.phase_interaction.shape[1]:
        phase = _as_array(
            law.phase_interaction, np.float64, ndim=2, name="phase_interaction"
        )
        phase_weights = np.arange(1, phase.shape[1] + 1, dtype=np.float64)
        projection += phase @ (phase_weights / np.linalg.norm(phase_weights))
    order = np.lexsort(
        (
            np.arange(n_nodes, dtype=np.int64),
            projection,
            component_fiber,
            parent_fiber,
            sociability,
        )
    ).astype(np.int64, copy=False)

    count = 2 * n_nodes - 1
    child_left = np.full(count, -1, dtype=np.int32)
    child_right = np.full(count, -1, dtype=np.int32)
    subtree_size = np.empty(count, dtype=np.int32)
    singleton_vertex = np.full(count, -1, dtype=np.int32)
    coordinate_low = np.empty((count, rank), dtype=np.float64)
    coordinate_high = np.empty((count, rank), dtype=np.float64)
    sociability_max = np.empty(count, dtype=np.float64)
    star_max = np.empty(count, dtype=np.float64)
    parent_mask = np.empty(count, dtype=np.uint64)

    stack = [(0, 0, n_nodes)]
    while stack:
        node_id, start, stop = stack.pop()
        width = stop - start
        subtree_size[node_id] = width
        if width == 1:
            vertex = int(order[start])
            singleton_vertex[node_id] = vertex
            coordinate_low[node_id] = interaction[vertex]
            coordinate_high[node_id] = interaction[vertex]
            sociability_max[node_id] = sociability[vertex]
            star_max[node_id] = star[vertex]
            parent_mask[node_id] = np.uint64(1) << np.uint64(parent_fiber[vertex])
            continue
        midpoint = start + width // 2
        left_width = midpoint - start
        left = node_id + 1
        right = node_id + 2 * left_width
        child_left[node_id] = left
        child_right[node_id] = right
        stack.append((right, midpoint, stop))
        stack.append((left, start, midpoint))

    for node_id in range(count - 1, -1, -1):
        left = int(child_left[node_id])
        if left < 0:
            continue
        right = int(child_right[node_id])
        np.minimum(coordinate_low[left], coordinate_low[right], out=coordinate_low[node_id])
        np.maximum(coordinate_high[left], coordinate_high[right], out=coordinate_high[node_id])
        sociability_max[node_id] = max(sociability_max[left], sociability_max[right])
        star_max[node_id] = max(star_max[left], star_max[right])
        parent_mask[node_id] = parent_mask[left] | parent_mask[right]

    depth = 1 if n_nodes == 1 else int((n_nodes - 1).bit_length() + 1)
    tree = SourceOrderTree(
        child_left,
        child_right,
        subtree_size,
        singleton_vertex,
        coordinate_low,
        coordinate_high,
        sociability_max,
        star_max,
        parent_mask,
        depth,
    )
    return tree, {
        "tree_node_count": count,
        "tree_leaf_count": n_nodes,
        "tree_depth": depth,
        "tree_order": (
            "sociability,parent_fiber,component_fiber,"
            "interaction_and_phase_projection,vertex_id"
        ),
        "python_tree_object_count": 0,
    }


def generate_source_edges(
    law: RecoveredGraphLaw,
    tree: SourceOrderTree,
    *,
    order_seed: int,
    mark_seed: int,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Run the exact frontier-free source race and return canonical edges."""

    started = time.perf_counter()
    target_edges = int(law.target_edges)
    output_edges = np.empty((target_edges, 2), dtype=np.int64)
    output_utilities = np.empty(target_edges, dtype=np.float64)
    integer_diagnostics = np.zeros(5, dtype=np.int64)
    real_diagnostics = np.zeros(9, dtype=np.float64)

    library = load_source_order_library()
    status = library.route_s_generate_edges(
        law.n_nodes,
        law.rank,
        int(np.size(law.marked_shift)),
        int(np.size(law.component_shift)),
        target_edges,
        float(law.degree_noise_std),
        float(law.temperature),
        _as_array(law.target_endpoint_mass, np.float64, ndim=1, name="target_endpoint_mass"),
        _as_array(law.sociability, np.float64, ndim=1, name="sociability"),
        _as_array(law.star, np.float64, ndim=1, name="star"),
        _as_array(law.interaction, np.float64, ndim=2, name="interaction"),
        _as_array(law.interaction_diagonal, np.float64, ndim=1, name="interaction_diagonal"),
        float(law.dust),
        _as_array(law.parent_fiber, np.int32, ndim=1, name="parent_fiber"),
        _as_array(law.component_fiber, np.int32, ndim=1, name="component_fiber"),
        _as_array(law.marked_shift, np.float64, ndim=1, name="marked_shift"),
        _as_array(law.component_shift, np.float64, ndim=1, name="component_shift"),
        _as_array(law.parent_flux, np.float64, ndim=2, name="parent_flux"),
        int(tree.subtree_size.size),
        _as_array(tree.child_left, np.int32, ndim=1, name="child_left"),
        _as_array(tree.child_right, np.int32, ndim=1, name="child_right"),
        _as_array(tree.subtree_size, np.int32, ndim=1, name="subtree_size"),
        _as_array(tree.singleton_vertex, np.int32, ndim=1, name="singleton_vertex"),
        _as_array(tree.coordinate_low, np.float64, ndim=2, name="coordinate_low"),
        _as_array(tree.coordinate_high, np.float64, ndim=2, name="coordinate_high"),
        _as_array(tree.sociability_max, np.float64, ndim=1, name="sociability_max"),
        _as_array(tree.star_max, np.float64, ndim=1, name="star_max"),
        _as_array(tree.parent_mask, np.uint64, ndim=1, name="parent_mask"),
        _root_path_hash(),
        int(order_seed),
        int(mark_seed),
        output_edges.reshape(-1),
        output_utilities,
        integer_diagnostics,
        real_diagnostics,
    )
    if int(status):
        raise RuntimeError(f"clean source-order kernel returned status {status}")

    canonical_edges = np.sort(output_edges, axis=1)
    edge_order = np.lexsort((canonical_edges[:, 1], canonical_edges[:, 0]))
    canonical_edges = np.ascontiguousarray(canonical_edges[edge_order], dtype=np.int64)

    diagnostic_functions = {
        "candidate_prefix_limit": "route_s_last_source_candidate_prefix_limit",
        "maximum_materialized_segment_size": (
            "route_s_last_source_maximum_materialized_segment_size"
        ),
        "maximum_dfs_stack_size": "route_s_last_source_maximum_dfs_stack_size",
        "topk_round_count": "route_s_last_source_topk_round_count",
        "used_frontier_free_topk": "route_s_last_source_used_frontier_free_topk",
        "used_monolithic_fallback": "route_s_last_source_used_monolithic_fallback",
        "binary64_serial_tie_comparison_count": (
            "route_s_last_source_binary64_serial_tie_comparison_count"
        ),
        "operator_bound_call_count": "operator_bound_last_joint_bound_call_count",
        "operator_tightened_cell_count": "operator_bound_last_tightened_cell_count",
        "operator_only_prune_count": "operator_bound_last_joint_only_prune_count",
    }
    diagnostics = {
        key: int(getattr(library, name)())
        for key, name in diagnostic_functions.items()
    }
    diagnostics.update(
        {
            "visited_cell_count": int(integer_diagnostics[0]),
            "processed_singleton_count": int(integer_diagnostics[2]),
            "rejected_mark_count": int(integer_diagnostics[3]),
            "forced_completion_mark_count": int(integer_diagnostics[4]),
            "mean_mark_probability": float(real_diagnostics[0]),
            "minimum_mark_probability": float(real_diagnostics[1]),
            "maximum_mark_probability": float(real_diagnostics[2]),
            "terminal_degree_target_l1": float(real_diagnostics[5]),
            "fraction_of_total_dyads_materialized": float(real_diagnostics[6]),
            "selected_random_utility_threshold": float(real_diagnostics[7]),
            "maximum_unexpanded_cell_priority": float(real_diagnostics[8]),
            "selected_random_utility_minimum": float(
                np.min(output_utilities, initial=math.inf)
            ),
            "selected_random_utility_maximum": float(
                np.max(output_utilities, initial=-math.inf)
            ),
            "edge_sha256": hashlib.sha256(canonical_edges.tobytes()).hexdigest(),
            "source_order_seconds": time.perf_counter() - started,
            "full_dyad_array_created": False,
            "exact_stop_certificate": True,
        }
    )
    return canonical_edges, diagnostics


__all__ = (
    "build_source_order_tree",
    "generate_source_edges",
)
