from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np
import route_s_clean.p2_recovery as p2


def recover_checkpoint_in_memory(
    checkpoint: GaussianCheckpoint, public_experiment_id: int
) -> p2.RecoveredCheckpointState:
    """Exact ``recover_checkpoint_state`` arithmetic without a file materialization."""

    started = time.perf_counter()
    coherent = p2._recover_coherent_gaussian_base(checkpoint)
    shell = p2._compile_shell_posterior_state(
        coherent.raw_operator, float(checkpoint.operator_noise_std)
    )
    temperature, temperature_diagnostics = p2.derive_source_temperature(
        checkpoint, coherent.target_edges, coherent.endpoint_target, shell
    )
    temperature_seconds = time.perf_counter() - started

    recovery_started = time.perf_counter()
    recovered, operator_diagnostics = p2._recover_operator_shell_state(
        checkpoint, coherent, shell
    )
    endpoint_target = coherent.endpoint_target
    base_law, base_diagnostics = p2._freeze_base_graph_law(
        checkpoint, recovered, endpoint_target, temperature
    )
    graph_law, functional_diagnostics = p2._freeze_commutant_functional(
        base_law, checkpoint, recovered, int(public_experiment_id)
    )
    carrier, carrier_diagnostics = p2.recover_rank_coupled_degree_carrier(
        checkpoint, endpoint_target, graph_law.sociability, graph_law.target_edges
    )
    carrier = np.asarray(carrier, dtype=np.float64)
    graph_law = replace(
        graph_law,
        target_endpoint_mass=carrier,
        noisy_degree=carrier,
    )
    digest = hashlib.sha256()
    digest.update(graph_law.state_hash.encode("ascii"))
    digest.update(np.ascontiguousarray(carrier).tobytes())
    graph_law = replace(graph_law, state_hash=digest.hexdigest())
    recovery_seconds = time.perf_counter() - recovery_started
    return p2.RecoveredCheckpointState(
        checkpoint_path=Path("<in-memory-private-transcript>"),
        checkpoint=checkpoint,
        temperature=float(temperature),
        temperature_diagnostics=dict(temperature_diagnostics),
        graph_law=graph_law,
        endpoint_target=carrier,
        degree_carrier_target=carrier,
        recovery_diagnostics={
            **dict(operator_diagnostics),
            **dict(base_diagnostics),
            **dict(functional_diagnostics),
            **dict(carrier_diagnostics),
            "private_graph_access_after_release": False,
            "checkpoint_serialization_count": 0,
        },
        public_experiment_id=int(public_experiment_id),
        stage_seconds={
            "checkpoint_load_seconds": 0.0,
            "temperature_formula_seconds": float(temperature_seconds),
            "p2_recovery_seconds": float(recovery_seconds),
        },
        audit={
            "checkpoint_load_requests": 0,
            "checkpoint_load_materializations": 0,
            "p2_recovery_requests": 1,
            "p2_recovery_materializations": 1,
        },
    )
