"""Independent privacy-repaired author-model mechanism, not an equivalent port.

Trusted computation may access the graph. Public output generation accepts ONLY
DP-updated decoder weights, a noisy edge count and the public vertex universe.
Do not expose the raw encoder/features, gradients, losses or evaluation records.
"""
import math
import secrets
import time
import numpy as np
import torch
import torch.nn.functional as F
from privacy import Ledger, gaussian

DECODER_NAMES = ("dec1.weight", "dec2.weight", "mapping1.weight", "mapping2.weight")


def private_generator():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(secrets.randbits(63))
    return generator  # Seed/state must never be stored in public output.


def clipped_query(parameters, clip_radius):
    """Return the jointly clipped mean vector, using a bounded zero fallback.

    Non-finite raw gradients are mapped to zero inside the query, not to a
    data-dependent public failure or early stopping event.
    """
    grads = [None if p.grad is None else p.grad.detach() for p in parameters]
    finite = all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    norm2 = math.fsum(float(torch.sum(g.to(torch.float64).square()))
        for g in grads if g is not None) if finite else math.inf
    norm = math.sqrt(norm2)
    # A small conservative margin prevents float32 storage overshooting C.
    scale = min(1.0, clip_radius * (1 - 1e-8) / max(norm, 1e-300)) if math.isfinite(norm) else 0.0
    for p, g in zip(parameters, grads):
        if g is None or scale == 0:
            yield torch.zeros_like(p, dtype=torch.float64)
        else:
            yield g.to(torch.float64) * scale


def generate_from_release(decoder, n_nodes, latent_dim, prior_std, target_edges, public_seed):
    """Graph-independent inference matching the paper's prior -> decoder route.

    The author's main.forward(for_test=True) is deliberately NOT called because
    it encodes private input features again. Matrix operations below are the
    author's decoder and mapped inner product, with unchanged order.
    """
    assert set(decoder) == set(DECODER_NAMES)
    rng = torch.Generator().manual_seed(int(public_seed))
    with torch.no_grad():
        latent = torch.randn((n_nodes, latent_dim), generator=rng) * prior_std
        hidden = torch.mm(latent, decoder["dec1.weight"].t())
        hidden = F.relu(F.normalize(hidden, dim=-1, p=2))
        hidden = torch.mm(hidden, decoder["dec2.weight"].t())
        hidden = F.relu(F.normalize(hidden, dim=-1, p=2))
        left = torch.mm(hidden, decoder["mapping1.weight"].t())
        right = torch.mm(hidden, decoder["mapping2.weight"].t())
        scores = torch.mm(F.normalize(left, dim=-1, p=2), F.normalize(right, dim=-1, p=2).t())
        values = np.nan_to_num(scores.numpy(), nan=0.0, posinf=1.0, neginf=-1.0)
    row, col = np.triu_indices(n_nodes, k=1)
    count = min(max(int(target_edges), 0), len(row))
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    # The author selects the upper-triangular largest values; a public stable tie
    # rule and zero-count handling remove private-data-dependent failure cases.
    selected = np.argsort(values[row, col], kind="stable")[-count:]
    edges = np.column_stack((row[selected], col[selected])).astype(np.int64)
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]


def run(model, optimizer, dataset, args, model_args, epsilon, delta=1e-5,
        steps=100, edge_fraction=0.1, public_generation_seed=42):
    if not isinstance(steps, int) or steps < 1 or not 0 < edge_fraction < 1:
        raise ValueError("invalid public schedule")
    ledger = Ledger(float(epsilon), float(delta))
    rho_count = ledger.limit * edge_fraction
    rho_each = ledger.limit * (1 - edge_fraction) / steps
    trainable = [p for p in model.parameters() if p.requires_grad]
    noise = private_generator()
    n = int(dataset.adj.shape[0])
    # Self loops are the author's training convention, not released edges.
    dense = torch.FloatTensor(np.asarray(dataset.adj.todense()))
    pos_weight = float(n * n - dataset.adj.sum()) / float(dataset.adj.sum())
    nodes = np.arange(n)
    initial_C = float(model_args.grad_norm_max)
    schedule = []
    begin = time.perf_counter()
    cpu_begin = time.process_time()
    for step in range(steps):
        C = initial_C * float(model_args.C_decay) ** step
        sigma, cost = gaussian(2 * C, rho_each)
        ledger.reserve("training_step_%03d" % step, cost)  # BEFORE raw query and update.
        optimizer.zero_grad(set_to_none=True)
        _, logvar, reconstruction, gan_pred, gan_label = model.forward(nodes, dense, for_test=False)
        loss = model.loss_function(reconstruction, dense, logvar, pos_weight, gan_pred, gan_label)
        if bool(torch.isfinite(loss)):
            loss.backward()
        # All trainable parameters form ONE clipped query. In particular, an
        # edge can change all spectral features and all example gradients.
        for parameter, mean in zip(trainable, clipped_query(trainable, C)):
            perturbation = torch.randn(mean.shape, dtype=torch.float64, generator=noise) * sigma
            parameter.grad = (mean + perturbation).to(parameter.dtype)
        optimizer.step()
        schedule.append({"step": step, "clip_radius": C, "l2_sensitivity": 2 * C,
            "gaussian_sigma": sigma, "rho_cost": cost})
    training_seconds = time.perf_counter() - begin
    training_cpu_seconds = time.process_time() - cpu_begin
    sigma_count, cost_count = gaussian(1.0, rho_count)
    ledger.reserve("edge_count", cost_count)
    exact_edge_count = float((dataset.adj.sum() - dataset.adj.diagonal().sum()) / 2)
    released_edge_count = exact_edge_count + float(torch.randn((), dtype=torch.float64, generator=noise)) * sigma_count
    target = int(round(min(max(released_edge_count, 0), n * (n - 1) / 2)))
    named = dict(model.named_parameters())
    # Only the parameters used by author's test-time decoder are extracted.
    decoder = {name: named[name].detach()[0].clone() for name in DECODER_NAMES}
    generation_start = time.perf_counter()
    edges = generate_from_release(decoder, n, model.layer2_dim, args.std, target, public_generation_seed)
    generation_seconds = time.perf_counter() - generation_start
    certificate = ledger.report()
    certificate.update({"adjacency": "one undirected unweighted edge addition/removal, fixed public labeled vertex set",
        "training_steps": steps, "edge_fraction": edge_fraction, "training_fraction": 1 - edge_fraction,
        "gradient_sensitivity_proof": "both complete clipped gradient vectors have norm <=C, hence distance <=2C",
        "edge_count_sensitivity": 1, "edge_count_sigma": sigma_count, "target_edges_after_DP_postprocessing": target,
        "generation_has_raw_graph_access": False, "private_noise_seed_exposed": False,
        "noise_implementation": "secret OS-seeded Torch float64 Gaussian PRNG; ideal-Gaussian theorem, no finite-precision/CSPRNG certification",
        "scope": "graph output ONLY; raw-data metrics, timing, RSS, hashes and logs are trusted benchmark audit, not DP releases",
        "schedule": schedule})
    times = {"training_seconds": training_seconds, "training_cpu_seconds": training_cpu_seconds,
        "decoder_generation_seconds": generation_seconds,
        "mechanism_internal_seconds": time.perf_counter() - begin}
    return edges, certificate, times
