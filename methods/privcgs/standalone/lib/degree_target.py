from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np
from common import canonical


def degree_target_from_carrier_degree(
        degree, edge_count, release, recovered, released, stack, cycle02):
    """Historical released degree law without requiring a carrier edge array."""
    import target_histogram
    namespace = vars(target_histogram)
    histogram = np.asarray(released.integer_histogram, dtype=np.int64)
    probability = np.asarray(released.probability, dtype=np.float64)
    expected = len(recovered.endpoint_target) * probability
    aligned, correction = namespace["_adjacent_mass_correction"](
        histogram, expected, 2 * int(edge_count))
    aligned, repair = namespace["_graphicalize_by_balancing"](aligned)
    support = np.arange(len(aligned), dtype=np.int64)
    multiset = np.repeat(support, aligned)
    carrier_degree = np.asarray(degree, dtype=np.float64)
    residual = np.asarray(recovered.endpoint_target, dtype=float) - carrier_degree
    centered = residual - residual.mean()
    observed = float(np.var(centered, ddof=1 if len(centered) > 1 else 0))
    noise = float(recovered.graph_law.degree_noise_std) ** 2
    latent = max(observed - noise, 0.0)
    shrinkage = latent / (latent + noise) if latent else 0.0

    import structural_order as source
    rank_order = source.structural_order(
        carrier_degree.astype(np.int64), recovered.graph_law,
        source.source_rank_direct(recovered.graph_law))
    rank = np.empty(len(multiset), dtype=np.int64)
    rank[rank_order] = np.arange(len(multiset), dtype=np.int64)
    nodes = np.arange(len(multiset), dtype=np.int64)
    order = np.lexsort((nodes, rank, carrier_degree + shrinkage * residual))
    target = np.empty_like(multiset)
    target[order] = multiset
    return np.ascontiguousarray(target), {
        "mass_correction": correction,
        "graphicality_repair": repair,
        "posterior_shrinkage": shrinkage,
        "degree_noise_variance": noise,
        "observed_residual_variance": observed,
        "carrier_degree_from_flow_state": True,
        "carrier_graph_materialized": False,
    }


def selected_deltas(tape, flow):
    positive = tape.positive_rank <= np.maximum(flow[tape.positive_bundle], 0)
    negative = tape.negative_rank <= np.maximum(-flow[tape.negative_bundle], 0)
    remove = np.concatenate((tape.negative_old_edge[negative],
                             tape.positive_old_edge[positive]), axis=0)
    add = np.concatenate((tape.negative_new_edge[negative],
                          tape.positive_new_edge[positive]), axis=0)
    return canonical(remove), canonical(add)
