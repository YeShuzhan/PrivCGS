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
    parser.add_argument("--large-graph", action="store_true")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.sched_setaffinity(0, [args.cpu])
    neural_profile = {"DPGVAE": "dpg-cuda", "DPGGAN": "dpg-cuda", "PrivDPR": "privdpr-cuda"}.get(args.method)
    if neural_profile:
        if args.profile != neural_profile:
            parser.error(f"{args.method} uses the {neural_profile} profile")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if not torch.cuda.is_available():
            parser.error(f"The {neural_profile} profile requires CUDA-enabled PyTorch and a visible NVIDIA GPU")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import common

    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.method == "Tricycle" and args.profile == "tricycle-gaussian-ladder":
        import ladder_runner

        synthetic, ledger, diagnostics = ladder_runner.run(
            args.graph, args.n, args.epsilon, args.seed,
            large=args.large_graph, compiler=args.compiler,
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

        synthetic, ledger, diagnostics = generate(common.read_graph(args.graph, args.n),
                                                  args.epsilon, seed=args.seed, device="cuda:0", threads=1)
        if not ledger["within_budget"] or not ledger["private_noise_is_OS_seeded"]:
            raise RuntimeError("PrivDPR CUDA accounting checks failed")
    else:
        parser.error("No direct adapter for this method")
    if neural_profile:
        torch.cuda.synchronize()
    array = np.asarray(synthetic, dtype=np.int64).reshape((-1, 2))
    common.save_edges(args.output / "synthetic.edgelist", array)
    for name, value in (("privacy_accounting", ledger), ("diagnostics", diagnostics)):
        (args.output / f"{name}.json").write_text(json.dumps(value, default=str, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps({
        "method": args.method, "profile": args.profile, "n_nodes": args.n, "n_edges": len(array),
        "epsilon": args.epsilon, "seed": args.seed, "device": "cuda:0" if neural_profile else "cpu",
        "wrapper_wall_seconds": perf_counter() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
