"""Public-domain validation and trusted benchmark I/O, separate from releases."""
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import numpy as np


def private_rng():
    # Never record its seed/state. Public benchmark seed is a separate stream.
    return np.random.default_rng(secrets.randbits(128))


def read_graph(path, n):
    if not isinstance(n, int) or n < 2:
        raise ValueError("explicit public node count >=2 required")
    edges = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("two integer endpoints per line required")
            u, v = map(int, parts)
            if not 0 <= u < n or not 0 <= v < n or u == v:
                raise ValueError("edge outside public simple-graph domain")
            edges.append((min(u, v), max(u, v)))
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    return adj


def target_size(signed_counts, n):
    """Undirected M: add signed *continuous* reports, divide by two, then clip."""
    total = math.fsum(float(x) for x in np.asarray(signed_counts).ravel()) / 2
    if not math.isfinite(total):
        raise ValueError("nonfinite released counts")
    return int(math.floor(min(max(total, 0), n * (n - 1) / 2) + .5))


def edges_of(adj):
    return np.column_stack(np.where(np.triu(adj, 1) > 0)).astype(np.int64)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def save_edges(path, edges):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    # Preserve the existing benchmark's Windows CRLF serialization on Linux.
    with Path(path).open("w", encoding="ascii", newline="") as stream:
        for u, v in edges:
            stream.write("%d %d\r\n" % (u, v))


def resource_guard(n, method, minimum_ram_gib=12, minimum_disk_gib=64):
    import psutil
    import shutil
    available = psutil.virtual_memory().available
    free = shutil.disk_usage(Path(__file__).resolve().parent).free
    # Public-N conservative admission guard, not a graph/model approximation.
    # Author replicated per-example parameters/gradients dominate DPG memory.
    estimate = max(512 * 2**20, n*n*(256 if method in ("DPGVAE", "DPGGAN") else 128)
        + (n * 105 * 128 * 128 * 8 if method in ("DPGVAE", "DPGGAN") else 0))
    if available - estimate < minimum_ram_gib * 2**30 or free < minimum_disk_gib * 2**30:
        raise RuntimeError("RESOURCE_DEFERRED: public-N memory estimate/reserves; no graph reduction")
    return {"available_bytes": available, "free_disk_bytes": free, "estimated_bytes": estimate}


LIMITATION = "Ideal Gaussian/accounting validation only; not complete end-to-end privacy certification. Only synthetic graph is a release; input hashes, timings, metrics and diagnostic reports are trusted benchmark audit."
