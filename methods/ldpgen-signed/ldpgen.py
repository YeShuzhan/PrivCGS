"""Author-lineage LDPGen + signed-total fixed-M postprocessing (independent).

Clustering/mapping/intra/inter/OR follow the PrivGraph-provided author package.
Signed continuous second reports are saved BEFORE the author's integer/clipping
view. Only that global noisy total controls M; no true edge-count target.
"""
import math
import random
import warnings
import numpy as np
from sklearn.cluster import KMeans
from common import private_rng, target_size, edges_of
from privacy import Ledger, gaussian


def author_intra(dd, rng):
    # Original generate_intra_edge, with int64 products (no int32 overflow).
    d = np.maximum(np.asarray(dd, dtype=np.int64), 0)
    n = len(d)
    total = int(d.sum())
    if total == 0:
        return np.zeros((n, n), dtype=np.uint8)
    pair_product = d[:, None] * d[None, :]
    directed = pair_product > rng.integers(0, total, size=(n, n))
    upper = np.triu(directed, 1)
    return (upper | upper.T).astype(np.uint8)


def author_reconstruct(counts, labels, rng):
    n, k = counts.shape
    out = np.zeros((n, n), dtype=np.uint8)
    for i in range(k):
        loc = np.flatnonzero(labels == i)
        out[np.ix_(loc, loc)] = author_intra(counts[loc, i], rng)
        for j in range(k):
            if i == j:
                continue
            loc2 = np.flatnonzero(labels == j)
            if not len(loc2):
                continue  # Empty block has no candidate pairs; no RNG draw.
            drawn = rng.integers(0, len(loc2), size=(len(loc), len(loc2)))
            out[np.ix_(loc, loc2)] = counts[loc, j, None] > drawn
    upper = np.triu(out | out.T, 1)
    return (upper | upper.T).astype(np.uint8)


def reconcile_total(adj, target, rng):
    """Minimal repair: uniformly remove/add only the noisy count difference.

    Selection is random without replacement. A deliberate fixed-M variant of
    the author's Bernoulli/OR reconstruction, not an equivalent reproduction.
    """
    n = len(adj)
    row, col = np.triu_indices(n, 1)
    chosen = adj[row, col].astype(bool)
    difference = int(target) - int(chosen.sum())
    if difference:
        candidates = np.flatnonzero(~chosen if difference > 0 else chosen)
        changed = rng.choice(candidates, size=abs(difference), replace=False)
        chosen[changed] = difference > 0
    result = np.zeros_like(adj)
    result[row, col] = chosen
    result |= result.T
    return result


def generate(adj, epsilon, delta=1e-5, seed=42, k=10):
    n = len(adj)
    if not 2 <= k <= n:
        raise ValueError("public author k must satisfy 2<=k<=N; no silent shrink")
    ledger = Ledger(epsilon, delta)
    noise = private_rng()
    public = np.random.default_rng(seed)
    half = set(random.Random(seed).sample(list(range(n)), n//2))
    labels0 = np.array([0 if i in half else 1 for i in range(n)])
    reports0 = np.column_stack([adj[:, labels0 == j].sum(1) for j in range(2)]).astype(np.int64)
    sigma, cost = gaussian(math.sqrt(2), ledger.limit / 2)
    ledger.reserve("round_1_node_group_counts", cost)
    # Quantize the RELEASE, not noise before addition: the former is provable
    # Gaussian postprocessing. Author's trunc(noise)+true_count is not silently
    # treated as that same mechanism (trunc is not translation equivariant).
    noisy0 = np.trunc(reports0 + noise.normal(0, sigma, reports0.shape)).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        km0 = KMeans(n_clusters=k, max_iter=100000, n_init=10, random_state=int(seed)).fit(noisy0)
        labels1 = km0.predict(noisy0)
    reports1 = np.column_stack([adj[:, labels1 == j].sum(1) for j in range(k)]).astype(np.int64)
    ledger.reserve("round_2_node_group_counts", cost)
    raw_noise = noise.normal(0, sigma, reports1.shape)
    signed = reports1 + raw_noise
    # NEW: total precedes every coordinatewise clipping and integer rounding.
    target = target_size(signed, n)
    # Preserve the author's clustering/reconstruction view as a separate copy.
    clipped = np.maximum(np.trunc(signed).astype(np.int64), 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        km1 = KMeans(n_clusters=k, max_iter=100000, n_init=10, random_state=int(seed)).fit(clipped)
        labels2 = km1.predict(clipped)
    mapped = np.zeros((n, k), dtype=np.int64)
    for i in range(k):
        loc1 = np.flatnonzero(labels1 == i)
        if not len(loc1):
            continue  # DP postprocessing empty cluster; avoids original 0/0.
        weights = np.array([np.count_nonzero(labels2[loc1] == j)/len(loc1) for j in range(k)])
        # Intentionally retain author's orientation, not silent W transpose fix.
        mapped[:, i] = np.sum(clipped * weights, axis=1).astype(np.int64)
    initial = author_reconstruct(mapped, labels2, public)
    result = reconcile_total(initial, target, public)
    certificate = ledger.report()
    certificate.update({"variant": "LDPGen_author_signed_total_gaussian_v1", "k": k,
        "l2_sensitivity_each_round": math.sqrt(2), "sigma_each_round": sigma,
        "target_edges": target, "count_source": "sum continuous signed round-2 reports /2 before clipping",
        "count_extra_privacy_cost": 0, "author_mapping_orientation_retained": True,
        "fixed_total_repair": "uniform random add/remove difference, without replacement",
        "privacy_scope": "undirected edge add/remove, fixed public IDs; two-round joint transcript zCDP",
        "generation_raw_graph_access": False, "noise_seed_exposed": False})
    return edges_of(result), certificate
