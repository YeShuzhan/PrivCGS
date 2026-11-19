"""Sparse path-aligned relocation charts for the degree-conditioned law."""

from __future__ import annotations

import heapq
import time
from typing import Mapping

import numpy as np

from .candidate_chart_native import (
    load_candidate_chart_library,
    load_scarcity_stratified_descent_library,
)
from .contracts import PackedRelocationTape, RecoveredGraphLaw
from .degree_chain import assemble_packed_relocation_tape
from .graph_carrier import CSRGraph
from .source_order_native import load_source_order_library


def phase_fiber_path(
    residual,
    component_fiber,
    parent_fiber,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Exact lazy-heap realization of the phase/parent lexicographic path."""

    values = np.asarray(residual, dtype=np.float64)
    component = np.asarray(component_fiber, dtype=np.int64)
    parent = np.asarray(parent_fiber, dtype=np.int64)
    tolerance = 1.0e-12
    state = np.zeros(values.size, dtype=np.int8)
    state[values > tolerance] = 1
    state[values < -tolerance] = -1
    alive = np.ones(values.size, dtype=np.bool_)
    counts = {sign: int(np.count_nonzero(state == sign)) for sign in (-1, 0, 1)}
    global_heaps = {sign: [] for sign in (-1, 0, 1)}
    component_heaps: dict[tuple[int, int], list[tuple[float, int]]] = {}
    parent_heaps: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for node in range(values.size):
        sign = int(state[node])
        key = (-abs(float(values[node])), int(node))
        global_heaps[sign].append(key)
        component_heaps.setdefault((sign, int(component[node])), []).append(key)
        parent_heaps.setdefault((sign, int(parent[node])), []).append(key)
    for collection in (global_heaps, component_heaps, parent_heaps):
        for heap in collection.values():
            heapq.heapify(heap)

    def peek(heap):
        if heap is None:
            return None
        while heap and not bool(alive[int(heap[0][1])]):
            heapq.heappop(heap)
        return heap[0] if heap else None

    def choose(sign: int, previous: int | None) -> int:
        selected = None
        if previous is not None:
            selected = peek(component_heaps.get((sign, int(component[previous]))))
            if selected is None:
                selected = peek(parent_heaps.get((sign, int(parent[previous]))))
        if selected is None:
            selected = peek(global_heaps[sign])
        node = int(selected[1])
        alive[node] = False
        counts[sign] -= 1
        return node

    order = np.empty(values.size, dtype=np.int64)
    cursor = 0
    prefix = 0.0
    maximum_prefix = 0.0
    previous = None
    while counts[-1] > 0 or counts[1] > 0:
        if previous is None:
            positive = peek(global_heaps[1]) if counts[1] > 0 else None
            negative = peek(global_heaps[-1]) if counts[-1] > 0 else None
            sign = 1 if negative is None or (
                positive is not None
                and (-float(positive[0]), -int(positive[1]))
                >= (-float(negative[0]), -int(negative[1]))
            ) else -1
        elif prefix > tolerance and counts[-1] > 0:
            sign = -1
        elif prefix < -tolerance and counts[1] > 0:
            sign = 1
        elif counts[1] > 0 and counts[-1] > 0:
            positive = peek(global_heaps[1])
            negative = peek(global_heaps[-1])
            sign = 1 if (
                (-float(positive[0]), -int(positive[1]))
                >= (-float(negative[0]), -int(negative[1]))
            ) else -1
        else:
            sign = 1 if counts[1] > 0 else -1
        node = choose(sign, previous)
        order[cursor] = node
        cursor += 1
        prefix += float(values[node])
        maximum_prefix = max(maximum_prefix, abs(prefix))
        previous = node
    while counts[0] > 0:
        node = choose(0, previous)
        order[cursor] = node
        cursor += 1
        previous = node
    return order, {
        "phase_fiber_path_complete": True,
        "maximum_absolute_path_prefix_residual": float(maximum_prefix),
        "terminal_path_prefix_residual": float(prefix),
        "path_tradeoff_parameter_count": 0,
        "complexity": "O(n log n) time and O(n) state",
    }


def edge_eta_batch(law: RecoveredGraphLaw, left, right) -> np.ndarray:
    left = np.ascontiguousarray(left, dtype=np.int64)
    right = np.ascontiguousarray(right, dtype=np.int64)
    output = np.empty(left.size, dtype=np.float64)
    status = int(
        load_source_order_library().route_s_edge_eta_batch(
            law.n_nodes,
            law.rank,
            int(np.size(law.marked_shift)),
            int(np.size(law.component_shift)),
            np.ascontiguousarray(law.sociability, dtype=np.float64),
            np.ascontiguousarray(law.star, dtype=np.float64),
            np.ascontiguousarray(law.interaction, dtype=np.float64),
            np.ascontiguousarray(law.interaction_diagonal, dtype=np.float64),
            float(law.dust),
            np.ascontiguousarray(law.parent_fiber, dtype=np.int32),
            np.ascontiguousarray(law.component_fiber, dtype=np.int32),
            np.ascontiguousarray(law.marked_shift, dtype=np.float64),
            np.ascontiguousarray(law.component_shift, dtype=np.float64),
            np.ascontiguousarray(law.parent_flux, dtype=np.float64),
            int(left.size),
            left,
            right,
            output,
        )
    )
    if status:
        raise ArithmeticError(f"clean edge-eta kernel returned {status}")
    return output


def _pack_selected(path, path_index, direction, old_edge, new_edge, gain):
    bundle_count = max(int(path.size) - 1, 0)
    directional = []
    for sign in (-1, 1):
        selected = np.flatnonzero(direction == sign)
        if selected.size:
            selected = selected[np.argsort(path_index[selected], kind="stable")]
        bundle = np.ascontiguousarray(path_index[selected], dtype=np.int64)
        counts = np.bincount(bundle, minlength=bundle_count).astype(np.int64)
        directional.append(
            (
                counts,
                np.ascontiguousarray(old_edge[selected], dtype=np.int64),
                np.ascontiguousarray(new_edge[selected], dtype=np.int64),
                np.ascontiguousarray(gain[selected], dtype=np.float64),
            )
        )
    negative, positive = directional
    return assemble_packed_relocation_tape(
        path[:-1], path[1:], *negative, *positive
    )[0]


def _candidate_arrays(source: CSRGraph, path: np.ndarray):
    library = load_candidate_chart_library()
    count_out = np.zeros(1, dtype=np.int64)
    status = int(library.route_s_count_relocation_candidates(
        source.n_nodes, path, source.csr_offsets, source.csr_neighbors,
        source.edge_keys.size, source.edge_keys, count_out,
    ))
    if status:
        raise RuntimeError(f"candidate count returned {status}")
    count = int(count_out[0])
    path_index = np.empty(count, dtype=np.int64)
    direction = np.empty(count, dtype=np.int8)
    old_left = np.empty(count, dtype=np.int64)
    old_right = np.empty(count, dtype=np.int64)
    new_left = np.empty(count, dtype=np.int64)
    new_right = np.empty(count, dtype=np.int64)
    status = int(library.route_s_fill_relocation_candidates(
        source.n_nodes, path, source.csr_offsets, source.csr_neighbors,
        source.edge_keys.size, source.edge_keys, count, path_index, direction,
        old_left, old_right, new_left, new_right,
    ))
    if status:
        raise RuntimeError(f"candidate fill returned {status}")
    return path_index, direction, old_left, old_right, new_left, new_right


def build_unique_dyad_chart(
    law: RecoveredGraphLaw,
    source: CSRGraph,
    path,
) -> tuple[PackedRelocationTape, Mapping[str, object]]:
    started = time.perf_counter()
    order = np.ascontiguousarray(path, dtype=np.int64)
    arrays = _candidate_arrays(source, order)
    path_index, direction, old_left, old_right, new_left, new_right = arrays
    count = int(path_index.size)
    old_key = old_left.astype(np.uint64) * np.uint64(source.n_nodes) + old_right.astype(np.uint64)
    new_key = new_left.astype(np.uint64) * np.uint64(source.n_nodes) + new_right.astype(np.uint64)
    unique_key, inverse = np.unique(np.concatenate((old_key, new_key)), return_inverse=True)
    inverse = np.ascontiguousarray(inverse, dtype=np.int64)
    eta = edge_eta_batch(
        law,
        np.ascontiguousarray(unique_key // np.uint64(source.n_nodes), dtype=np.int64),
        np.ascontiguousarray(unique_key % np.uint64(source.n_nodes), dtype=np.int64),
    )
    gain = np.ascontiguousarray(eta[inverse[count:]] - eta[inverse[:count]], dtype=np.float64)
    selected_buffer = np.empty(count, dtype=np.int64)
    selected_count = np.zeros(1, dtype=np.int64)
    status = int(load_candidate_chart_library().route_s_sort_and_admit_relocation_ids(
        count, gain, int(unique_key.size),
        np.ascontiguousarray(inverse[:count], dtype=np.int64),
        np.ascontiguousarray(inverse[count:], dtype=np.int64),
        selected_buffer, selected_count,
    ))
    if status:
        raise RuntimeError(f"candidate admission returned {status}")
    selected = selected_buffer[: int(selected_count[0])]
    tape = _pack_selected(
        order, path_index[selected], direction[selected],
        np.column_stack((old_left, old_right))[selected],
        np.column_stack((new_left, new_right))[selected], gain[selected],
    )
    return tape, {
        "raw_candidate_count": count,
        "unique_eta_dyad_count": int(unique_key.size),
        "admitted_unit_count": int(selected.size),
        "seconds": float(time.perf_counter() - started),
        "global_dyad_disjoint": True,
        "complete_dyad_enumeration": False,
        "complete_wedge_enumeration": False,
    }


def build_descent_support_chart(
    law: RecoveredGraphLaw,
    source: CSRGraph,
    target_endpoint_mass,
    path,
) -> tuple[PackedRelocationTape, Mapping[str, object]]:
    started = time.perf_counter()
    order = np.ascontiguousarray(path, dtype=np.int64)
    target = np.ascontiguousarray(target_endpoint_mass, dtype=np.float64)
    degree = source.degree_array.astype(np.float64)
    residual = target - degree
    desired_flow = -np.cumsum(residual[order])[:-1]
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(desired_flow), initial=0.0)), 1.0
    )
    requested = np.ceil(np.abs(desired_flow)).astype(np.int64)
    requested[np.abs(desired_flow) <= tolerance] = 0
    direction = np.sign(desired_flow).astype(np.int8)
    direction[requested == 0] = 0
    requested = np.ascontiguousarray(requested)
    direction = np.ascontiguousarray(direction)
    capacity = min(
        int(np.sum(requested, dtype=np.int64)), int(source.edge_keys.size)
    )
    storage = max(capacity, 1)
    path_index = np.empty(storage, dtype=np.int64)
    old_left = np.empty(storage, dtype=np.int64)
    old_right = np.empty(storage, dtype=np.int64)
    new_left = np.empty(storage, dtype=np.int64)
    new_right = np.empty(storage, dtype=np.int64)
    gain = np.empty(storage, dtype=np.float64)
    admitted = np.zeros(max(source.n_nodes - 1, 0), dtype=np.int64)
    selected_count = np.zeros(1, dtype=np.int64)
    stratified_diagnostics = np.zeros(10, dtype=np.int64)
    status = int(
        load_scarcity_stratified_descent_library().route_s_scarcity_stratified_descent(
            law.n_nodes,
            law.rank,
            int(np.size(law.marked_shift)),
            int(np.size(law.component_shift)),
            np.ascontiguousarray(law.sociability, dtype=np.float64),
            np.ascontiguousarray(law.star, dtype=np.float64),
            np.ascontiguousarray(law.interaction, dtype=np.float64),
            np.ascontiguousarray(law.interaction_diagonal, dtype=np.float64),
            float(law.dust),
            np.ascontiguousarray(law.parent_fiber, dtype=np.int32),
            np.ascontiguousarray(law.component_fiber, dtype=np.int32),
            np.ascontiguousarray(law.marked_shift, dtype=np.float64),
            np.ascontiguousarray(law.component_shift, dtype=np.float64),
            np.ascontiguousarray(law.parent_flux, dtype=np.float64),
            order,
            direction,
            requested,
            source.csr_offsets,
            source.csr_neighbors,
            source.edge_keys.size,
            source.edge_keys,
            capacity,
            path_index,
            old_left,
            old_right,
            new_left,
            new_right,
            gain,
            admitted,
            selected_count,
            stratified_diagnostics,
        )
    )
    if status:
        raise RuntimeError(f"scarcity-stratified descent returned {status}")
    selected = int(selected_count[0])
    path_index = np.ascontiguousarray(path_index[:selected])
    old_left = np.ascontiguousarray(old_left[:selected])
    old_right = np.ascontiguousarray(old_right[:selected])
    new_left = np.ascontiguousarray(new_left[:selected])
    new_right = np.ascontiguousarray(new_right[:selected])
    gain = np.ascontiguousarray(gain[:selected])
    tape = _pack_selected(
        order, path_index, direction[path_index],
        np.column_stack((old_left, old_right)),
        np.column_stack((new_left, new_right)), gain,
    )
    real_shortfall = np.maximum(np.abs(desired_flow) - admitted.astype(np.float64), 0.0)
    return tape, {
        "raw_candidate_count": int(stratified_diagnostics[0]),
        "unique_eta_dyad_count": None,
        "requested_capacity_sum": int(np.sum(requested)),
        "admitted_capacity_sum": int(np.sum(admitted)),
        "real_capacity_shortfall_sum": float(np.sum(real_shortfall)),
        "exact_real_target_flow_inside_admitted_box": bool(np.all(real_shortfall <= tolerance)),
        "seconds": float(time.perf_counter() - started),
        "scarcity_stratified_descent": True,
        "active_bundle_count": int(stratified_diagnostics[1]),
        "scarcity_stratum_count": int(stratified_diagnostics[2]),
        "maximum_stratum_candidate_count": int(stratified_diagnostics[3]),
        "admitted_unit_count": int(stratified_diagnostics[4]),
        "reserved_dyad_count": int(stratified_diagnostics[5]),
        "examined_candidate_count": int(stratified_diagnostics[6]),
        "provably_unread_tail_count": int(stratified_diagnostics[7]),
        "maximum_local_candidate_bytes": int(stratified_diagnostics[8]),
        "direct_eta_call_count": int(stratified_diagnostics[9]),
        "complete_dyad_enumeration": False,
        "complete_wedge_enumeration": False,
    }


__all__ = (
    "build_descent_support_chart",
    "build_unique_dyad_chart",
    "edge_eta_batch",
    "phase_fiber_path",
)
