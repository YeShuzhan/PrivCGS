"""One independent checkpoint-to-edge execution path for Route S.

The checkpoint is the sole private transcript.  Every operation below is
deterministic postprocessing of that transcript and public randomness.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from .candidate_chart import (
    build_descent_support_chart,
    build_unique_dyad_chart,
    phase_fiber_path,
)
from .chain_moments import (
    efficient_palm_exposure_messages,
    efficient_spectral_score_messages,
)
from .conditional_transport import (
    baseline_preserving_information_transport,
    select_carrier_component,
)
from .contracts import EndToEndResult, PackedRelocationTape, PipelineConfig, StageTimings
from .degree_chain import (
    combine_tapes,
    completion_from_core_descent,
    descent_cone,
    horizontal_completion,
)
from .endpoint_laws import (
    EndpointLaw,
    phase_protected_configuration_endpoint,
    spherical_equal_information_endpoint,
)
from .finite_chain import sample_backward
from .graph_carrier import CSRGraph, apply_relocation_state
from .hamiltonian_completion import positive_hamiltonian_completion
from .p2_recovery import RecoveredCheckpointState, recover_checkpoint_state
from .palm_coordinates import (
    compile_horizontal_palm_score_columns,
    compile_vertical_palm_score_columns,
)
from .parent_laws import (
    closure_degree_moment_substitution_parent,
    hamiltonian_shadow_parent,
    horizontal_anti_closure_parent,
)
from .retraction import counting_measure_retraction_shadow
from .source_rh2b import build_source_order_tree, generate_source_edges
from .structural_columns import (
    reorder_flag_faces,
    spectral_columns_from_compiled_tape,
    spectral_two_step_features,
    vertical_completion_occupation,
)
def _terminal_not_installed(*args, **kwargs):
    raise RuntimeError("Use the standalone run.py MCR1/W4/Cstar entry point")

apply_terminal = prepare_terminal_state = _terminal_not_installed
from .vertical_parent import (
    gaussian_soft_vertical_parent,
    orthogonal_degree_carrier_reliability,
    vertical_anti_closure_parent,
)


_SOURCE_ORDER_SEED_PREFIX = 376_200_000
_SOURCE_MARK_SEED_PREFIX = 376_300_000
_PALM_SEED_PREFIX = 537_000_000
_FLOW_SEED_PREFIX = 404_000_000


@dataclass(frozen=True)
class _SelectedEndpoints:
    tape: PackedRelocationTape
    phase_protected: EndpointLaw
    spherical_reference: EndpointLaw
    diagnostics: Mapping[str, object]


def _horizontal_endpoints(
    recovered: RecoveredCheckpointState,
    source: CSRGraph,
    path: np.ndarray,
    flagged_core: PackedRelocationTape,
    flagged_full_completion: PackedRelocationTape,
    flagged_horizontal_completion: PackedRelocationTape,
    degree_reliability: float,
    palm_seed: int,
) -> _SelectedEndpoints:
    """Compile only the selected horizontal conditional law."""

    law = recovered.graph_law
    degree = source.degree_array.astype(np.float64)
    target = recovered.endpoint_target
    sigma = recovered.checkpoint.degree_sigma
    feature_cache = spectral_two_step_features(law)

    completion = positive_hamiltonian_completion(
        law,
        path,
        flagged_core,
        flagged_horizontal_completion,
        degree,
        target,
        sigma,
        feature_cache=feature_cache,
    )
    full_chart = combine_tapes(flagged_core, flagged_full_completion)
    admitted_chart = combine_tapes(flagged_core, completion.filtered_completion)
    full_score = efficient_spectral_score_messages(
        law, path, full_chart, degree, target, sigma,
        feature_cache=feature_cache,
        _calibration_only=True,
    )
    admitted_score = efficient_spectral_score_messages(
        law, path, admitted_chart, degree, target, sigma,
        feature_cache=feature_cache,
        precompiled_spectral_columns=spectral_columns_from_compiled_tape(
            completion.final_spectral_tape
        ),
    )
    columns = compile_horizontal_palm_score_columns(
        source, admitted_chart, palm_seed
    )
    exposure = efficient_palm_exposure_messages(
        source,
        law,
        path,
        admitted_chart,
        degree,
        target,
        sigma,
        public_seed=palm_seed,
        feature_cache=feature_cache,
        base_score=admitted_score,
        coordinate_columns=columns,
        rebind_coordinate_columns=False,
    )
    moment_parent = closure_degree_moment_substitution_parent(
        path,
        admitted_chart,
        exposure.messages,
        exposure.quadratic_coefficient,
        admitted_score.messages,
        admitted_score.quadratic_coefficient,
        columns.closure_increment_additive,
        degree,
        target,
        sigma,
    )
    del exposure
    anti_closure_parent = horizontal_anti_closure_parent(
        path,
        admitted_chart,
        moment_parent,
        columns.palm_additive,
        columns.closure_increment_additive,
        degree,
        target,
        sigma,
    )
    del moment_parent
    full_spectral = full_score.spectral_columns
    full_spectral_tape = replace(
        full_chart,
        negative_energy=np.ascontiguousarray(
            full_spectral.negative_delta, dtype=np.float64
        ),
        positive_energy=np.ascontiguousarray(
            full_spectral.positive_delta, dtype=np.float64
        ),
        structural_gain=np.ascontiguousarray(
            full_spectral.additive_gain, dtype=np.float64
        ),
    )
    shadow = counting_measure_retraction_shadow(
        full_chart,
        admitted_chart,
        full_spectral_tape,
        completion.final_spectral_tape,
        full_score.residual_scale,
        admitted_score.residual_scale,
    )
    completion_statistics = dict(completion.statistics)
    del full_score, admitted_score, full_spectral_tape, full_chart, completion
    parent = hamiltonian_shadow_parent(
        path,
        admitted_chart,
        anti_closure_parent,
        columns.palm_additive,
        columns.closure_increment_additive,
        shadow.additive,
        degree,
        target,
        sigma,
    )
    del anti_closure_parent
    phase = phase_protected_configuration_endpoint(
        path,
        admitted_chart,
        parent.messages,
        parent.quadratic_coefficient,
        columns,
        degree,
        target,
        sigma,
        degree_reliability=degree_reliability,
    )
    spherical = spherical_equal_information_endpoint(
        path,
        admitted_chart,
        parent.messages,
        parent.quadratic_coefficient,
        columns,
        degree,
        target,
        sigma,
        degree_reliability=degree_reliability,
    )
    shadow_diagnostics = dict(shadow.diagnostics)
    parent_diagnostics = dict(parent.diagnostics)
    del shadow, parent, columns
    return _SelectedEndpoints(
        tape=admitted_chart,
        phase_protected=phase,
        spherical_reference=spherical,
        diagnostics={
            "component": "horizontal",
            "chart_state_count": int(admitted_chart.state_value.size),
            "hamiltonian": completion_statistics,
            "retraction": shadow_diagnostics,
            "parent": parent_diagnostics,
        },
    )


def _vertical_endpoints(
    recovered: RecoveredCheckpointState,
    source: CSRGraph,
    path: np.ndarray,
    flagged_core: PackedRelocationTape,
    flagged_completion: PackedRelocationTape,
    flagged_horizontal: PackedRelocationTape,
    degree_reliability: float,
    palm_seed: int,
) -> _SelectedEndpoints:
    """Compile only the selected vertical conditional law."""

    law = recovered.graph_law
    degree = source.degree_array.astype(np.float64)
    target = recovered.endpoint_target
    sigma = recovered.checkpoint.degree_sigma
    feature_cache = spectral_two_step_features(law)

    horizontal_completion_law = positive_hamiltonian_completion(
        law,
        path,
        flagged_core,
        flagged_horizontal,
        degree,
        target,
        sigma,
        feature_cache=feature_cache,
    )
    horizontal_parent_chart = combine_tapes(
        flagged_core, horizontal_completion_law.filtered_completion
    )
    horizontal_parent_score = efficient_spectral_score_messages(
        law, path, horizontal_parent_chart, degree, target, sigma,
        feature_cache=feature_cache,
        precompiled_spectral_columns=spectral_columns_from_compiled_tape(
            horizontal_completion_law.final_spectral_tape
        ),
        _calibration_only=True,
        _retain_spectral_columns=False,
    )
    full_chart = combine_tapes(flagged_core, flagged_completion)
    full_score = efficient_spectral_score_messages(
        law, path, full_chart, degree, target, sigma,
        feature_cache=feature_cache,
    )
    occupation, occupation_diagnostics = vertical_completion_occupation(
        law,
        path,
        flagged_core,
        flagged_completion,
        full_chart,
    )
    carrier_reliability, carrier_diagnostics = (
        orthogonal_degree_carrier_reliability(
            recovered.checkpoint.continuous_noisy_degree,
            recovered.checkpoint.basis,
            sigma,
        )
    )
    soft_parent = gaussian_soft_vertical_parent(
        path,
        full_chart,
        full_score.messages,
        full_score.quadratic_coefficient,
        horizontal_parent_score.quadratic_coefficient,
        occupation,
        degree,
        target,
        sigma,
        carrier_reliability=carrier_reliability,
        diagnostics_required=False,
    )
    columns = compile_vertical_palm_score_columns(
        source, full_chart, law.component_fiber, palm_seed
    )
    parent = vertical_anti_closure_parent(
        path,
        full_chart,
        soft_parent,
        columns.phase_additive,
        columns.closure_increment_additive,
        occupation,
        degree,
        target,
        sigma,
        diagnostics_required=False,
    )
    phase = phase_protected_configuration_endpoint(
        path,
        full_chart,
        parent.messages,
        parent.quadratic_coefficient,
        columns,
        degree,
        target,
        sigma,
        degree_reliability=degree_reliability,
    )
    spherical = spherical_equal_information_endpoint(
        path,
        full_chart,
        parent.messages,
        parent.quadratic_coefficient,
        columns,
        degree,
        target,
        sigma,
        degree_reliability=degree_reliability,
    )
    return _SelectedEndpoints(
        tape=full_chart,
        phase_protected=phase,
        spherical_reference=spherical,
        diagnostics={
            "component": "vertical",
            "chart_state_count": int(full_chart.state_value.size),
            "horizontal_hamiltonian": dict(horizontal_completion_law.statistics),
            "vertical_occupation": dict(occupation_diagnostics),
            "carrier": dict(carrier_diagnostics),
            "parent": dict(parent.diagnostics),
        },
    )


def synthesize_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    public_experiment_id: int,
    canonical_sort: bool = True,
) -> EndToEndResult:
    """Generate one final sparse graph from one Gaussian Edge-DP checkpoint.

    No privacy parameter is accepted here: the serialized checkpoint is the
    authoritative released transcript.  The experiment ID controls public
    counter-based streams only.
    """

    total_started = time.perf_counter()
    recovery_started = time.perf_counter()
    recovered = recover_checkpoint_state(
        Path(checkpoint_path), public_experiment_id=int(public_experiment_id)
    )
    recovery_seconds = time.perf_counter() - recovery_started
    law = recovered.graph_law
    experiment_id = int(public_experiment_id)

    source_started = time.perf_counter()
    tree, tree_diagnostics = build_source_order_tree(law)
    source_edges, source_diagnostics = generate_source_edges(
        law,
        tree,
        order_seed=_SOURCE_ORDER_SEED_PREFIX + experiment_id,
        mark_seed=_SOURCE_MARK_SEED_PREFIX + experiment_id,
    )
    source = CSRGraph.from_edges(source_edges, law.n_nodes)
    source_edge_count = int(source_edges.shape[0])
    del tree, source_edges
    source_seconds = time.perf_counter() - source_started

    compile_started = time.perf_counter()
    degree = source.degree_array.astype(np.float64)
    residual = recovered.endpoint_target - degree
    path, path_diagnostics = phase_fiber_path(
        residual, law.component_fiber, law.parent_fiber
    )
    core, core_diagnostics = build_unique_dyad_chart(law, source, path)
    descent, descent_diagnostics = build_descent_support_chart(
        law, source, recovered.endpoint_target, path
    )
    restricted_core, restriction_diagnostics = descent_cone(
        path, core, degree, recovered.endpoint_target
    )
    completion, completion_diagnostics = completion_from_core_descent(
        path,
        restricted_core,
        descent,
        degree,
        recovered.endpoint_target,
    )
    flagged_core, core_flag_diagnostics = reorder_flag_faces(
        source, restricted_core
    )
    flagged_completion, completion_flag_diagnostics = reorder_flag_faces(
        source, completion
    )
    # Horizontal completion is a union of complete path bundles, while
    # flag-face ordering is independent inside each bundle.  Projecting the
    # already flagged full completion is therefore fieldwise identical to
    # projecting first and compiling a third flag order.  The latter tape and
    # face/sort pass are not sufficient state for any downstream consumer.
    flagged_horizontal, horizontal_diagnostics = horizontal_completion(
        law.component_fiber, path, flagged_completion
    )
    horizontal_flag_diagnostics = {
        "flag_unit_count": int(
            flagged_horizontal.negative_energy.size
            + flagged_horizontal.positive_energy.size
        ),
        "whole_bundle_projection_after_full_flag_order": True,
        "duplicate_face_evaluation_count": 0,
        "duplicate_bundle_sort_count": 0,
        "complete_dyad_enumeration": False,
    }
    del core, descent, restricted_core, completion
    common_compile_seconds = time.perf_counter() - compile_started

    branch_started = time.perf_counter()
    decision = select_carrier_component(
        recovered.checkpoint.continuous_noisy_degree,
        recovered.checkpoint.degree_sigma,
        flow_seed_prefix=_FLOW_SEED_PREFIX,
        public_experiment_id=experiment_id,
    )
    degree_reliability = float(decision.reliability)
    reliability_keys = (
        "degree_channel_node_count",
        "degree_channel_observed_mle_variance",
        "degree_channel_noise_variance",
        "degree_channel_positive_part_latent_variance",
        "degree_channel_reliability",
        "positive_part_boundary_active",
        "variance_estimator_denominator",
        "reliability_is_normal_random_effects_eb",
    )
    reliability_diagnostics = {
        key: decision.diagnostics[key] for key in reliability_keys
    }
    palm_seed = _PALM_SEED_PREFIX + experiment_id
    if decision.selected_component == "horizontal":
        selected = _horizontal_endpoints(
            recovered,
            source,
            path,
            flagged_core,
            flagged_completion,
            flagged_horizontal,
            degree_reliability,
            palm_seed,
        )
    else:
        selected = _vertical_endpoints(
            recovered,
            source,
            path,
            flagged_core,
            flagged_completion,
            flagged_horizontal,
            degree_reliability,
            palm_seed,
        )
    del flagged_core, flagged_completion, flagged_horizontal
    transport = baseline_preserving_information_transport(
        path,
        selected.tape,
        selected.spherical_reference.messages,
        selected.spherical_reference.quadratic_coefficient,
        selected.phase_protected.messages,
        selected.phase_protected.quadratic_coefficient,
        degree,
        recovered.endpoint_target,
        recovered.checkpoint.degree_sigma,
        degree_reliability=decision.reliability,
    )
    branch_seconds = time.perf_counter() - branch_started

    draw_started = time.perf_counter()
    flow_state, flow_diagnostics = sample_backward(
        path,
        transport.messages,
        degree,
        recovered.endpoint_target,
        recovered.checkpoint.degree_sigma,
        quadratic_coefficient=transport.quadratic_coefficient,
        structural_temperature=1.0,
        sample_seed=_FLOW_SEED_PREFIX + experiment_id,
    )
    conditional = apply_relocation_state(source, selected.tape, flow_state)
    conditional_edges = np.ascontiguousarray(conditional.edge_array, dtype=np.int64)
    draw_seconds = time.perf_counter() - draw_started

    terminal_started = time.perf_counter()
    prepared = prepare_terminal_state(
        conditional_edges,
        law,
        public_experiment_id=experiment_id,
    )
    final_edges, terminal_diagnostics = apply_terminal(
        prepared, canonical_sort=bool(canonical_sort)
    )
    terminal_seconds = time.perf_counter() - terminal_started
    total_seconds = time.perf_counter() - total_started

    conditional_digest = hashlib.sha256(conditional_edges.tobytes()).hexdigest()
    final_digest = hashlib.sha256(final_edges.tobytes()).hexdigest()
    diagnostics = {
        "selected_component": decision.selected_component,
        "carrier_decision": dict(decision.diagnostics),
        "degree_reliability": dict(reliability_diagnostics),
        "source_tree": dict(tree_diagnostics),
        "source_order": dict(source_diagnostics),
        "path": dict(path_diagnostics),
        "core_chart": dict(core_diagnostics),
        "descent_chart": dict(descent_diagnostics),
        "descent_restriction": dict(restriction_diagnostics),
        "completion": dict(completion_diagnostics),
        "horizontal_restriction": dict(horizontal_diagnostics),
        "core_flag_order": dict(core_flag_diagnostics),
        "completion_flag_order": dict(completion_flag_diagnostics),
        "horizontal_flag_order": dict(horizontal_flag_diagnostics),
        "selected_branch": dict(selected.diagnostics),
        "transport": dict(transport.diagnostics),
        "flow": dict(flow_diagnostics),
        "terminal_preparation": dict(prepared.preparation_statistics),
        "terminal": dict(terminal_diagnostics),
        "source_edge_count": source_edge_count,
        "conditional_edge_count": int(conditional_edges.shape[0]),
        "final_edge_count": int(final_edges.shape[0]),
        "conditional_edge_sha256": conditional_digest,
        "final_edge_sha256": final_digest,
        "total_seconds": float(total_seconds),
        "checkpoint_load_count": int(
            recovered.audit["checkpoint_load_materializations"]
        ),
        "private_graph_access_after_checkpoint": False,
        "complete_dyad_enumerated": False,
        "complete_wedge_enumerated": False,
        "probability_law_changed": False,
        "public_random_stream_changed": False,
    }
    timings = StageTimings(
        checkpoint_release_seconds=0.0,
        law_recovery_seconds=float(recovery_seconds),
        source_order_seconds=float(source_seconds),
        degree_chain_compile_seconds=float(common_compile_seconds),
        degree_chain_draw_seconds=float(draw_seconds),
        phase_compile_seconds=float(branch_seconds),
        terminal_seconds=float(terminal_seconds),
    )
    return EndToEndResult(
        edge_array=np.ascontiguousarray(final_edges, dtype=np.int64),
        checkpoint_path=recovered.checkpoint_path,
        recovered_state_hash=str(law.state_hash),
        timings=timings,
        diagnostics=diagnostics,
    )




__all__ = ("synthesize_from_checkpoint",)
