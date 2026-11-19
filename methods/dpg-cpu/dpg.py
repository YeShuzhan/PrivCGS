"""Pinned author's model, with independent full-query Gaussian zCDP trainer.

Imports only model/config, not the author's benchmark logger or evaluation.
Read-only source snapshot is shipped in vendor/dpg_author.
"""
import ast
import importlib
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import torch


def load_author():
    source = Path(__file__).resolve().parent / "vendor/dpg_author"
    sys.path.insert(0, str(source))
    # Author functional.py imports a removed private Torch module but does not
    # use it. AST verifies that before providing the unused compatibility name.
    parsed = ast.parse((source/"src/DPGGAN/functional.py").read_text(encoding="utf-8"))
    if any(isinstance(x, ast.Name) and x.id == "_functions" for x in ast.walk(parsed)):
        raise RuntimeError("author compatibility precondition changed")
    if not hasattr(torch.nn, "_functions"):
        torch.nn._functions = types.ModuleType("torch.nn._functions")
    from src.DPGGAN.model import DPGGAN
    from src import config
    return DPGGAN, config


def spectral_dataset(adj, maximum_features=128):
    """Exact author's generate_feature operations; SciPy keyword adapter only."""
    n = len(adj)
    features = min(maximum_features, int(.65*n))
    if features < 1:
        raise ValueError("public graph size insufficient for author spectral features")
    matrix = sp.csr_matrix(np.asarray(adj, dtype=np.float64))
    with_loops = matrix + sp.eye(n)
    row_sum = np.asarray(with_loops.sum(1))
    inverse = sp.diags(np.power(row_sum, -.5).flatten())
    normalized = with_loops.dot(inverse).transpose().dot(inverse).toarray()
    _, values = la.eigh(normalized, subset_by_index=(n-features, n-1))
    adjacency = with_loops.tocsr()
    lists = {i: set(adjacency.getrow(i).indices.tolist()) for i in range(n)}
    return SimpleNamespace(adj=adjacency, features=values, adj_list=lists, actual_feature_dim=features)


def generate(adj, model_name, epsilon, delta=1e-5, seed=42, steps=100,
        author_profile="dblp2", edge_fraction=.1):
    if model_name not in ("DPGVAE", "DPGGAN"):
        raise ValueError("unknown author model")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Model, configs = load_author()
    # Public profile chooses author's existing hyperparameters, not a data fit.
    if author_profile not in ("dblp2", "IMDB_MULTI"):
        raise ValueError("unknown public author profile")
    config_name = "imdb" if author_profile == "IMDB_MULTI" else author_profile
    model_config = getattr(configs, model_name+"_"+config_name+"_Adam")()
    dataset = spectral_dataset(adj)
    n = len(adj)
    # Required for the author's discriminator's raw spectral feature input.
    model_config.dec2_dim = dataset.actual_feature_dim
    args = SimpleNamespace(discriminator_ratio=.1 if model_name == "DPGGAN" else 0,
        kl_ratio=1.0, std=.75, model_name=model_name, batch_size=n, num_samples=n)
    model = Model(args, model_config, dataset.features, dataset.adj_list)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=.005)
    from dpg_mechanism import run
    edges, certificate, times = run(model, optimizer, dataset, args, model_config,
        epsilon, delta, steps=steps, edge_fraction=edge_fraction, public_generation_seed=seed)
    certificate.update({"variant": model_name+"_author_architecture_full_query_gaussian_v2",
        "author_commit": "cbc28d38346921bf06147ed7a63494132f3c207d",
        "author_profile": author_profile, "effective_config": vars(model_config),
        "learning_rate": .005, "feature_dim_public": dataset.actual_feature_dim,
        "decoder_dim_adapter": "dec2_dim=public spectral feature width for discriminator compatibility",
        "optimizer": "Adam", "features": "author structural spectral features; trusted training only",
        "adaptations": ["whole-query joint L2 clipping and composed Gaussian rho schedule",
            "fixed public training steps", "separate noisy edge count, rho fraction "+str(edge_fraction),
            "generation uses graph-independent prior and DP decoder only; no raw re-encoding"]})
    return edges, certificate, times
