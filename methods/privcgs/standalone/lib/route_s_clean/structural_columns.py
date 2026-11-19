"""Flag, spectral, phase-block, and vertical-occupation chain columns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape, RecoveredGraphLaw
from .graph_carrier import CSRGraph
from .native import ragged_prefix
from .source_order_native import load_source_order_library
from .spectral import batched_binary64_row_dot


def _decode_relocations(old_edge, new_edge):
    old = np.asarray(old_edge, dtype=np.int64).reshape((-1, 2))
    new = np.asarray(new_edge, dtype=np.int64).reshape((-1, 2))
    old0_common = (old[:, 0] == new[:, 0]) | (old[:, 0] == new[:, 1])
    old1_common = (old[:, 1] == new[:, 0]) | (old[:, 1] == new[:, 1])
    common = np.where(old0_common, old[:, 0], old[:, 1])
    donor = np.where(old0_common, old[:, 1], old[:, 0])
    receiver = np.where(new[:, 0] == common, new[:, 1], new[:, 0])
    return donor, receiver, common


def flag_face_delta(source: CSRGraph, donors, receivers, common):
    donor = np.ascontiguousarray(donors, dtype=np.int64)
    receiver = np.ascontiguousarray(receivers, dtype=np.int64)
    shared = np.ascontiguousarray(common, dtype=np.int64)
    output = np.empty(donor.size, dtype=np.int64)
    status = int(load_source_order_library().route_s_flag_face_delta_batch(
        source.n_nodes,
        source.csr_offsets,
        source.csr_neighbors,
        int(donor.size),
        donor,
        receiver,
        shared,
        output,
    ))
    if status:
        raise RuntimeError(f"clean flag-face kernel returned {status}")
    return output


def _direction_flag_order(source, offsets, old_edge, new_edge, energy):
    donor, receiver, common = _decode_relocations(old_edge, new_edge)
    score = flag_face_delta(source, donor, receiver, common)
    permutation = np.empty(score.size, dtype=np.int64)
    changed = 0
    for index in range(len(offsets) - 1):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if stop == start:
            continue
        local = np.lexsort((
            new_edge[start:stop, 1],
            new_edge[start:stop, 0],
            old_edge[start:stop, 1],
            old_edge[start:stop, 0],
            -energy[start:stop],
            -score[start:stop],
        )).astype(np.int64, copy=False)
        permutation[start:stop] = start + local
        changed += int(not np.array_equal(local, np.arange(stop - start, dtype=np.int64)))
    return permutation, score, changed


def reorder_flag_faces(
    source: CSRGraph,
    tape: PackedRelocationTape,
) -> tuple[PackedRelocationTape, Mapping[str, object]]:
    negative_order, negative_score, negative_changed = _direction_flag_order(
        source, tape.negative_offsets, tape.negative_old_edge,
        tape.negative_new_edge, tape.negative_energy,
    )
    positive_order, positive_score, positive_changed = _direction_flag_order(
        source, tape.positive_offsets, tape.positive_old_edge,
        tape.positive_new_edge, tape.positive_energy,
    )
    negative_energy = np.ascontiguousarray(tape.negative_energy[negative_order])
    positive_energy = np.ascontiguousarray(tape.positive_energy[positive_order])
    gain = ragged_prefix(
        tape.state_offsets, tape.negative_offsets, tape.positive_offsets,
        negative_energy, positive_energy,
    )
    output = replace(
        tape,
        negative_old_edge=np.ascontiguousarray(tape.negative_old_edge[negative_order]),
        negative_new_edge=np.ascontiguousarray(tape.negative_new_edge[negative_order]),
        negative_energy=negative_energy,
        positive_old_edge=np.ascontiguousarray(tape.positive_old_edge[positive_order]),
        positive_new_edge=np.ascontiguousarray(tape.positive_new_edge[positive_order]),
        positive_energy=positive_energy,
        structural_gain=gain,
    )
    scores = np.concatenate((negative_score, positive_score))
    return output, {
        "flag_unit_count": int(scores.size),
        "reordered_negative_bundle_count": int(negative_changed),
        "reordered_positive_bundle_count": int(positive_changed),
        "minimum": float(np.min(scores, initial=0)),
        "maximum": float(np.max(scores, initial=0)),
        "mean": float(np.mean(scores)) if scores.size else 0.0,
        "complete_dyad_enumeration": False,
    }


@dataclass(frozen=True)
class SpectralColumns:
    negative_delta: np.ndarray
    positive_delta: np.ndarray
    additive_gain: np.ndarray
    statistics: Mapping[str, object]


def spectral_columns_from_compiled_tape(
    tape: PackedRelocationTape,
) -> SpectralColumns:
    """Bind exact spectral columns already carried by a compiled tape.

    ``HamiltonianCompletion.final_spectral_tape`` stores spectral unit deltas
    in the two energy fields and their unchanged sequential binary64 ragged
    prefix in ``structural_gain``.  Binding those arrays is therefore the
    minimum sufficient state for a downstream spectral-score consumer; it
    must not rescan the admitted dyads merely to reconstruct the same columns.
    """

    negative = np.asarray(tape.negative_energy, dtype=np.float64)
    positive = np.asarray(tape.positive_energy, dtype=np.float64)
    additive = np.asarray(tape.structural_gain, dtype=np.float64)
    return SpectralColumns(
        negative_delta=negative,
        positive_delta=positive,
        additive_gain=additive,
        statistics={
            "unit_count": int(negative.size + positive.size),
            "hamiltonian_admission_spectral_sufficient_state_reused": True,
            "admitted_dyad_diagnostics_recomputed": False,
            "complete_dyad_enumeration": False,
            "dense_n_by_n_state": False,
        },
    )


def spectral_two_step_features(law: RecoveredGraphLaw):
    interaction = np.asarray(law.interaction, dtype=np.float64)
    diagonal = np.asarray(law.interaction_diagonal, dtype=np.float64)
    weighted = interaction * diagonal[None, :]
    gram = interaction.T @ interaction
    left_feature = weighted @ gram
    if not np.all(np.isfinite(weighted)) or not np.all(np.isfinite(left_feature)):
        raise ArithmeticError("two-step spectral factors are nonfinite")
    return left_feature, weighted, {
        "spectral_rank": int(interaction.shape[1]),
        "materialized_n_by_n_operator": False,
        "materialized_complete_dyad_vector": False,
    }


def compile_spectral_columns(
    law: RecoveredGraphLaw,
    tape: PackedRelocationTape,
    *,
    feature_cache=None,
) -> SpectralColumns:
    if feature_cache is None:
        left_feature, weighted, factor_stats = spectral_two_step_features(law)
    else:
        left_feature, weighted, factor_stats = feature_cache
    edges = np.concatenate((
        tape.negative_old_edge,
        tape.negative_new_edge,
        tape.positive_old_edge,
        tape.positive_new_edge,
    ), axis=0)
    key = edges[:, 0].astype(np.uint64) * np.uint64(law.n_nodes) + edges[:, 1].astype(np.uint64)
    unique, inverse = np.unique(key, return_inverse=True)
    left = np.asarray(unique // np.uint64(law.n_nodes), dtype=np.int64)
    right = np.asarray(unique % np.uint64(law.n_nodes), dtype=np.int64)
    score, row_dot_diagnostics = batched_binary64_row_dot(
        left_feature, weighted, left, right
    )
    negative_count = int(tape.negative_energy.size)
    positive_count = int(tape.positive_energy.size)
    negative_old = score[inverse[:negative_count]]
    negative_new = score[inverse[negative_count : 2 * negative_count]]
    positive_base = 2 * negative_count
    positive_old = score[inverse[positive_base : positive_base + positive_count]]
    positive_new = score[inverse[positive_base + positive_count : positive_base + 2 * positive_count]]
    negative_delta = np.asarray(negative_new - negative_old, dtype=np.float64)
    positive_delta = np.asarray(positive_new - positive_old, dtype=np.float64)
    additive = ragged_prefix(
        tape.state_offsets, tape.negative_offsets, tape.positive_offsets,
        negative_delta, positive_delta,
    )
    values = np.concatenate((negative_delta, positive_delta))
    return SpectralColumns(
        negative_delta,
        positive_delta,
        additive,
        {
            **dict(factor_stats),
            "unit_count": int(values.size),
            "unique_dyad_count": int(unique.size),
            "batched_row_dot_count": int(row_dot_diagnostics["row_dot_count"]),
            "maximum_row_dot_batch_size": int(
                row_dot_diagnostics["maximum_batch_size"]
            ),
            "temporary_operand_byte_cap": int(
                row_dot_diagnostics["temporary_operand_byte_cap"]
            ),
            "promoted_cumulative_batched_row_dot": True,
            "prebatched_scalar_bit_identity_claimed": False,
            "nonzero_increment_count": int(np.count_nonzero(values)),
            "complete_dyad_enumeration": False,
            "dense_n_by_n_state": False,
        },
    )


def _block_index(component, edges, label_count: int):
    edge = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    if edge.size == 0:
        return np.zeros(0, dtype=np.int64)
    labels = np.unique(component)
    left = np.searchsorted(labels, component[edge[:, 0]])
    right = np.searchsorted(labels, component[edge[:, 1]])
    low = np.minimum(left, right).astype(np.int64)
    high = np.maximum(left, right).astype(np.int64)
    start = low * int(label_count) - low * (low - 1) // 2
    return start + high - low


def phase_block_columns(law: RecoveredGraphLaw, tape: PackedRelocationTape):
    component = np.asarray(law.component_fiber, dtype=np.int64)
    labels = np.unique(component)
    label_count = int(labels.size)
    full_dimension = int(label_count * (label_count + 1) // 2)
    dimension = max(full_dimension - 1, 0)
    difference = np.zeros((tape.state_value.size, dimension), dtype=np.float64)

    def add(old_edge, new_edge, bundle, rank, *, negative):
        if not old_edge.size or dimension == 0:
            return np.zeros(old_edge.shape[0], dtype=bool)
        old_block = _block_index(component, old_edge, label_count)
        new_block = _block_index(component, new_edge, label_count)
        negative_count = np.diff(tape.negative_offsets)
        if negative:
            start = tape.state_offsets[bundle]
            stop = start + negative_count[bundle] - rank + 1
        else:
            start = tape.state_offsets[bundle] + negative_count[bundle] + rank
            stop = tape.state_offsets[bundle + 1]
        for block, value in ((new_block, 1.0), (old_block, -1.0)):
            valid = block < dimension
            if np.any(valid):
                np.add.at(difference, (start[valid], block[valid]), value)
                internal = valid & (stop < tape.state_value.size)
                if np.any(internal):
                    np.add.at(difference, (stop[internal], block[internal]), -value)
        return old_block != new_block

    negative_nonzero = add(
        tape.negative_old_edge, tape.negative_new_edge,
        tape.negative_bundle, tape.negative_rank, negative=True,
    )
    positive_nonzero = add(
        tape.positive_old_edge, tape.positive_new_edge,
        tape.positive_bundle, tape.positive_rank, negative=False,
    )
    return np.cumsum(difference, axis=0), {
        "component_fiber_count": label_count,
        "full_phase_block_dimension": full_dimension,
        "raw_phase_block_dimension": dimension,
        "nonzero_phase_block_unit_count": int(
            np.count_nonzero(negative_nonzero) + np.count_nonzero(positive_nonzero)
        ),
        "complete_dyad_enumeration": False,
    }


def vertical_completion_occupation(
    law: RecoveredGraphLaw,
    path,
    core: PackedRelocationTape,
    completion: PackedRelocationTape,
    combined: PackedRelocationTape,
):
    order = np.asarray(path, dtype=np.int64)
    component = np.asarray(law.component_fiber, dtype=np.int64)
    vertical_bundle = component[order[:-1]] != component[order[1:]]
    bundle = np.repeat(
        np.arange(combined.bundle_count, dtype=np.int64),
        np.diff(combined.state_offsets),
    )
    state = np.asarray(combined.state_value, dtype=np.int64)
    core_negative = np.diff(core.negative_offsets)
    core_positive = np.diff(core.positive_offsets)
    value = np.zeros(state.size, dtype=np.float64)
    negative = (state < 0) & vertical_bundle[bundle]
    positive = (state > 0) & vertical_bundle[bundle]
    value[negative] = np.maximum(
        -state[negative] - core_negative[bundle[negative]], 0
    )
    value[positive] = np.maximum(
        state[positive] - core_positive[bundle[positive]], 0
    )
    completion_count = np.diff(completion.negative_offsets) + np.diff(completion.positive_offsets)
    return value[:, None], {
        "vertical_completion_unit_count": int(np.sum(completion_count[vertical_bundle])),
        "vertical_occupation_is_law_statistic": True,
    }


__all__ = (
    "SpectralColumns",
    "compile_spectral_columns",
    "flag_face_delta",
    "phase_block_columns",
    "reorder_flag_faces",
    "spectral_columns_from_compiled_tape",
    "spectral_two_step_features",
    "vertical_completion_occupation",
)
