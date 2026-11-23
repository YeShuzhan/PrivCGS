"""Run a verified baseline bundle through its generation functions."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.sched_setaffinity(0, [args.cpu])
    privdpr_cuda = args.method == "PrivDPR" and args.profile == "privdpr-cuda"
    if privdpr_cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if not torch.cuda.is_available():
            parser.error("The privdpr-cuda profile requires a CUDA-enabled PyTorch build and a visible NVIDIA GPU")
        torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import common

    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.method == "Tricycle" and args.profile.startswith("tricycle-gaussian-ladder"):
        import ladder_runner

        synthetic, ledger, diagnostics = ladder_runner.run(
            args.graph, args.n, args.epsilon, args.seed,
            large=args.profile == "tricycle-gaussian-ladder-large", compiler=args.compiler,
        )
    elif args.method == "LDPGen":
        import ldpgen
        synthetic, ledger = ldpgen.generate(common.read_graph(args.graph, args.n), args.epsilon,
                                             seed=args.seed, k=10)
        diagnostics = {}
    elif args.method in ("DPGVAE", "DPGGAN"):
        import dpg
        synthetic, ledger, diagnostics = dpg.generate(common.read_graph(args.graph, args.n), args.method,
                                                        args.epsilon, seed=args.seed, steps=100,
                                                        author_profile="dblp2", edge_fraction=.1)
    elif args.method == "PrivDPR":
        from privdpr_zcdp import generate

        device = "cuda:0" if privdpr_cuda else "cpu"
        synthetic, ledger, diagnostics = generate(common.read_graph(args.graph, args.n),
                                                  args.epsilon, seed=args.seed, device=device, threads=1)
        if privdpr_cuda and (not ledger["within_budget"] or not ledger["private_noise_is_OS_seeded"]):
            raise RuntimeError("PrivDPR CUDA accounting checks failed")
    else:
        parser.error("No direct adapter for this method")
    array = np.asarray(synthetic, dtype=np.int64).reshape((-1, 2))
    common.save_edges(args.output / "synthetic.edgelist", array)
    for name, value in (("privacy_accounting", ledger), ("diagnostics", diagnostics)):
        (args.output / f"{name}.json").write_text(json.dumps(value, default=str, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps({
        "method": args.method, "profile": args.profile, "n_nodes": args.n, "n_edges": len(array),
        "epsilon": args.epsilon, "seed": args.seed, "wrapper_wall_seconds": perf_counter() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
