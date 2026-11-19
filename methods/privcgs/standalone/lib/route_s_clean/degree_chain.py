"""Continuous packed representation for the degree-conditioned finite chain."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .completion_native import admit_completion_units
from .contracts import PackedRelocationTape
from .native import ragged_prefix


def _direction_layout(count: np.ndarray):
    count = np.ascontiguousarray(count, dtype=np.int64)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(count, dtype=np.int64))
    )
    bundle = np.repeat(np.arange(count.size, dtype=np.int64), count)
    rank = np.arange(int(offsets[-1]), dtype=np.int64) - offsets[bundle] + 1
    return offsets, bundle, rank


def assemble_packed_relocation_tape(
    path_left,
    path_right,
    negative_count,
    negative_old_edge,
    negative_new_edge,
    negative_energy,
    positive_count,
    positive_old_edge,
    positive_new_edge,
    positive_energy,
) -> tuple[PackedRelocationTape, dict[str, int | str]]:
    left = np.ascontiguousarray(path_left, dtype=np.int64)
    right = np.ascontiguousarray(path_right, dtype=np.int64)
    negative_count = np.ascontiguousarray(negative_count, dtype=np.int64)
    positive_count = np.ascontiguousarray(positive_count, dtype=np.int64)
    negative = _direction_layout(negative_count)
    positive = _direction_layout(positive_count)
    negative_old = np.ascontiguousarray(negative_old_edge, dtype=np.int64).reshape((-1, 2))
    negative_new = np.ascontiguousarray(negative_new_edge, dtype=np.int64).reshape((-1, 2))
    negative_mark = np.ascontiguousarray(negative_energy, dtype=np.float64)
    positive_old = np.ascontiguousarray(positive_old_edge, dtype=np.int64).reshape((-1, 2))
    positive_new = np.ascontiguousarray(positive_new_edge, dtype=np.int64).reshape((-1, 2))
    positive_mark = np.ascontiguousarray(positive_energy, dtype=np.float64)
    state_count = negative_count + positive_count + 1
    state_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(state_count, dtype=np.int64))
    )
    state_bundle = np.repeat(np.arange(left.size, dtype=np.int64), state_count)
    state_value = (
        np.arange(int(state_offsets[-1]), dtype=np.int64)
        - state_offsets[state_bundle]
        - negative_count[state_bundle]
    )
    structural_gain = ragged_prefix(
        state_offsets,
        negative[0],
        positive[0],
        negative_mark,
        positive_mark,
    )
    tape = PackedRelocationTape(
        bundle_count=int(left.size),
        path_left=left,
        path_right=right,
        negative_offsets=negative[0],
        negative_bundle=negative[1],
        negative_rank=negative[2],
        negative_old_edge=negative_old,
        negative_new_edge=negative_new,
        negative_energy=negative_mark,
        positive_offsets=positive[0],
        positive_bundle=positive[1],
        positive_rank=positive[2],
        positive_old_edge=positive_old,
        positive_new_edge=positive_new,
        positive_energy=positive_mark,
        state_offsets=state_offsets,
        state_value=state_value,
        structural_gain=structural_gain,
    )
    return tape, {
        "bundle_count": int(tape.bundle_count),
        "directional_unit_count": int(negative_mark.size + positive_mark.size),
        "state_count": int(state_value.size),
        "floating_prefix_executor": "route_s_clean native sequential binary64",
    }


def _select_units(
    tape: PackedRelocationTape,
    negative_mask: np.ndarray,
    positive_mask: np.ndarray,
) -> PackedRelocationTape:
    negative_mask = np.asarray(negative_mask, dtype=bool)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    negative_count = np.bincount(
        tape.negative_bundle[negative_mask], minlength=tape.bundle_count
    ).astype(np.int64)
    positive_count = np.bincount(
        tape.positive_bundle[positive_mask], minlength=tape.bundle_count
    ).astype(np.int64)
    return assemble_packed_relocation_tape(
        tape.path_left,
        tape.path_right,
        negative_count,
        tape.negative_old_edge[negative_mask],
        tape.negative_new_edge[negative_mask],
        tape.negative_energy[negative_mask],
        positive_count,
        tape.positive_old_edge[positive_mask],
        tape.positive_new_edge[positive_mask],
        tape.positive_energy[positive_mask],
    )[0]


def directional_prefixes(
    tape: PackedRelocationTape,
    negative_capacity,
    positive_capacity,
) -> PackedRelocationTape:
    negative = np.ascontiguousarray(negative_capacity, dtype=np.int64)
    positive = np.ascontiguousarray(positive_capacity, dtype=np.int64)
    return _select_units(
        tape,
        tape.negative_rank <= negative[tape.negative_bundle],
        tape.positive_rank <= positive[tape.positive_bundle],
    )


def descent_cone(
    path,
    tape: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
) -> tuple[PackedRelocationTape, Mapping[str, object]]:
    order = np.ascontiguousarray(path, dtype=np.int64)
    degree = np.ascontiguousarray(source_degree, dtype=np.float64)
    target = np.ascontiguousarray(target_endpoint_mass, dtype=np.float64)
    residual = target - degree
    desired_flow = -np.cumsum(residual[order])[:-1]
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(desired_flow), initial=0.0)), 1.0
    )
    requested = np.ceil(np.abs(desired_flow)).astype(np.int64)
    requested[np.abs(desired_flow) <= tolerance] = 0
    available_negative = np.diff(tape.negative_offsets)
    available_positive = np.diff(tape.positive_offsets)
    negative_request = np.where(desired_flow < -tolerance, requested, 0)
    positive_request = np.where(desired_flow > tolerance, requested, 0)
    negative = np.minimum(negative_request, available_negative)
    positive = np.minimum(positive_request, available_positive)
    restricted = directional_prefixes(tape, negative, positive)
    admitted = negative + positive
    capacity_shortfall = np.maximum(np.abs(desired_flow) - admitted, 0.0)
    # Endpoint mass is certified in canonical projection order before the exact
    # rank-coupling permutation.  Re-summing the permuted carrier after a second
    # binary64 subtraction is order-dependent and is not a new mass certificate.
    # The descent chart uses only the prefix flow below, so repeating that Gate
    # here can reject a carrier without changing any state or mathematical mass.
    return restricted, {
        "descent_cone_zero_state_in_support": True,
        "descent_cone_one_sided_bundle_count": int(
            np.count_nonzero(np.abs(desired_flow) > tolerance)
        ),
        "desired_positive_capacity_sum": int(np.sum(positive_request)),
        "desired_negative_capacity_sum": int(np.sum(negative_request)),
        "admitted_positive_capacity_sum": int(np.sum(positive)),
        "admitted_negative_capacity_sum": int(np.sum(negative)),
        "real_capacity_shortfall_sum": float(np.sum(capacity_shortfall)),
        "exact_real_target_flow_inside_admitted_box": bool(
            np.all(capacity_shortfall <= tolerance)
        ),
    }


def _merge_direction(
    core: PackedRelocationTape,
    completion: PackedRelocationTape,
    prefix: str,
):
    core_offsets = getattr(core, f"{prefix}_offsets")
    completion_offsets = getattr(completion, f"{prefix}_offsets")
    core_count = np.diff(core_offsets)
    completion_count = np.diff(completion_offsets)
    total = int(core_offsets[-1] + completion_offsets[-1])
    old_edge = np.empty((total, 2), dtype=np.int64)
    new_edge = np.empty((total, 2), dtype=np.int64)
    energy = np.empty(total, dtype=np.float64)
    core_bundle = getattr(core, f"{prefix}_bundle")
    completion_bundle = getattr(completion, f"{prefix}_bundle")
    core_position = (
        np.arange(int(core_offsets[-1]), dtype=np.int64)
        + completion_offsets[core_bundle]
    )
    completion_position = (
        np.arange(int(completion_offsets[-1]), dtype=np.int64)
        + core_offsets[completion_bundle + 1]
    )
    old_edge[core_position] = getattr(core, f"{prefix}_old_edge")
    new_edge[core_position] = getattr(core, f"{prefix}_new_edge")
    energy[core_position] = getattr(core, f"{prefix}_energy")
    old_edge[completion_position] = getattr(completion, f"{prefix}_old_edge")
    new_edge[completion_position] = getattr(completion, f"{prefix}_new_edge")
    energy[completion_position] = getattr(completion, f"{prefix}_energy")
    return core_count + completion_count, old_edge, new_edge, energy


def combine_tapes(
    core: PackedRelocationTape,
    completion: PackedRelocationTape,
) -> PackedRelocationTape:
    negative = _merge_direction(core, completion, "negative")
    positive = _merge_direction(core, completion, "positive")
    return assemble_packed_relocation_tape(
        core.path_left, core.path_right, *negative, *positive
    )[0]


def horizontal_completion(
    component_fiber,
    path,
    completion: PackedRelocationTape,
) -> tuple[PackedRelocationTape, Mapping[str, object]]:
    component = np.ascontiguousarray(component_fiber, dtype=np.int64)
    order = np.ascontiguousarray(path, dtype=np.int64)
    horizontal = component[order[:-1]] == component[order[1:]]
    negative_available = np.diff(completion.negative_offsets)
    positive_available = np.diff(completion.positive_offsets)
    projected = directional_prefixes(
        completion,
        np.where(horizontal, negative_available, 0),
        np.where(horizontal, positive_available, 0),
    )
    unit_count = negative_available + positive_available
    return projected, {
        "full_completion_unit_count": int(np.sum(unit_count)),
        "horizontal_completion_unit_count": int(np.sum(unit_count[horizontal])),
        "vertical_completion_unit_count_excluded": int(
            np.sum(unit_count[~horizontal])
        ),
    }


def completion_from_core_descent(
    path,
    core: PackedRelocationTape,
    descent: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
) -> tuple[PackedRelocationTape, Mapping[str, object]]:
    order = np.ascontiguousarray(path, dtype=np.int64)
    degree = np.ascontiguousarray(source_degree, dtype=np.float64)
    target = np.ascontiguousarray(target_endpoint_mass, dtype=np.float64)
    desired = -np.cumsum((target - degree)[order])[:-1]
    n_nodes = int(order.size)

    positive = np.ascontiguousarray(desired > 0.0)
    requested = np.asarray(np.ceil(np.abs(desired)), dtype=np.int64)
    core_negative_count = np.diff(core.negative_offsets)
    core_positive_count = np.diff(core.positive_offsets)
    core_count = np.where(positive, core_positive_count, core_negative_count)
    required = np.maximum(requested - core_count, 0)
    negative_mask, positive_mask, native = admit_completion_units(
        n_nodes, core, descent, positive, required
    )
    completion = _select_units(descent, negative_mask, positive_mask)
    return completion, {
        "completion_selected_unit_count": int(native["selected_unit_count"]),
        "completion_reserved_dyad_count": int(native["reserved_dyad_count"]),
        "completion_compact_table_capacity": int(native["compact_table_capacity"]),
        "completion_single_pass": True,
        "completion_native_compact_reserved_set": True,
        "completion_python_set_count": 0,
    }


def packed_tape_sha256(tape: PackedRelocationTape) -> str:
    digest = hashlib.sha256()
    digest.update(int(tape.bundle_count).to_bytes(8, "little", signed=False))
    for name in tape.__dataclass_fields__:
        if name == "bundle_count":
            continue
        array = np.ascontiguousarray(getattr(tape, name))
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class PackedStageRegistry:
    _stages: dict[str, PackedRelocationTape] = field(default_factory=dict)

    def bind(self, name: str, tape: PackedRelocationTape) -> None:
        key = str(name)
        self._stages[key] = tape

    def get(self, name: str) -> PackedRelocationTape:
        return self._stages[str(name)]

    def diagnostics(self) -> Mapping[str, object]:
        return {
            "packed_stage_count": int(len(self._stages)),
            "stage_names": tuple(self._stages),
            "stage_hashes": {
                name: packed_tape_sha256(tape)
                for name, tape in self._stages.items()
            },
            "complete_dyad_enumeration": False,
            "complete_wedge_enumeration": False,
            "common_neighbor_intersection_work": 0,
        }


__all__ = (
    "PackedStageRegistry",
    "assemble_packed_relocation_tape",
    "combine_tapes",
    "completion_from_core_descent",
    "descent_cone",
    "directional_prefixes",
    "horizontal_completion",
    "packed_tape_sha256",
)
