"""Adapt the frozen Gaussian Ladder experiment to the public baseline CLI."""

from fractions import Fraction
import math
from pathlib import Path
from random import SystemRandom
import subprocess
from time import perf_counter

import numpy as np

import cl_core
from fast_ladder import build
from gaussian_ladder import _Ladder, _select


def run(graph, n, epsilon, seed, *, large, compiler, delta=1e-5):
    """Return synthetic edges, the public budget ledger, and benchmark diagnostics."""
    if not 0 <= n <= np.iinfo(np.int32).max:
        raise ValueError("The public node count must fit the native nonnegative int32 domain.")
    edges = np.loadtxt(graph, dtype=np.int64, ndmin=2)
    if not edges.size:
        edges = edges.reshape(0, 2)
    if edges.shape[1] != 2 or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("Expected canonical edge pairs on the public node domain.")

    if n >= 3:
        native = Path(__file__).resolve().parent
        library = native / "ladder_native.so"
        if not library.is_file():
            subprocess.run([
                compiler, "-std=c++17", "-O3", "-shared", "-fPIC",
                str(native / "ladder_native.cpp"), "-o", str(library),
            ], check=True)

    started = perf_counter()
    if n >= 3:
        ladder, _ = build(edges, n)
    else:
        ladder = _Ladder(n, [])
        if len(edges) and (n != 2 or len(edges) != 1 or tuple(edges[0]) != (0, 1)):
            raise ValueError("Expected a simple graph with canonical edge pairs.")
    degree = np.bincount(edges.ravel(), minlength=n)
    preparation_seconds = perf_counter() - started
    del edges

    started = perf_counter()
    rho = cl_core.rho_from_epsilon(epsilon, delta)
    rho_degree = rho_triangle = rho / 2
    sigma = 1 / math.sqrt(rho_degree)
    weights = np.clip(degree + np.random.default_rng().normal(0, sigma, n), 0, n - 1)
    upper = math.comb(n, 3)
    rounds = upper.bit_length()
    target = _select(ladder, upper, Fraction(rho_triangle), SystemRandom())
    release_seconds = perf_counter() - started
    del degree, ladder

    if large:
        import networkx as nx
        from networkx.generators import degree_seq
        from cl_boundary import build as build_sampler
        from compact_cl import generate as synthesize

        nx.expected_degree_graph = build_sampler(Path(degree_seq.__file__))
        max_seconds = math.inf
    else:
        synthesize = cl_core.generate
        max_seconds = 3600
    synthetic, diagnostics = synthesize(
        weights, target, seed, max_seconds=max_seconds, proposals_per_edge=100,
    )
    if large:
        diagnostics["max_repair_seconds"] = None
    assert diagnostics["edge_count_preserved"] and diagnostics["triangle_ledger_exact"]
    diagnostics.update(
        ladder_preparation_seconds=preparation_seconds,
        release_seconds=release_seconds,
    )
    ledger = {
        "epsilon": epsilon,
        "delta": delta,
        "rho_total": rho,
        "rho_degree": rho_degree,
        "rho_triangle": rho_triangle,
        "degree_l2_sensitivity": math.sqrt(2),
        "degree_sigma": sigma,
        "comparison_round_limit": rounds,
        "comparison_l2_sensitivity": 2,
        "comparison_noise_variance": 2 * rounds / rho_triangle if rounds else None,
        "triangle_target": target,
        "triangle_mechanism": "Gaussian Ladder",
        "fresh_hidden_noise": True,
        "public_postprocessing_seed": seed,
    }
    return synthetic, ledger, diagnostics
