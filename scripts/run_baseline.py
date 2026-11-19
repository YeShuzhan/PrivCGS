"""Run a verified baseline bundle through its generation functions."""
import argparse
import json
import math
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
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.sched_setaffinity(0, [args.cpu])
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import common

    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.method == "Tricycle":
        import cl_core
        if args.profile == "large-cl-boundary":
            import cl_boundary
            import networkx as nx
            import networkx.generators.degree_seq as degree_seq
            nx.expected_degree_graph = cl_boundary.build(Path(degree_seq.__file__))
        edges = np.loadtxt(args.graph, dtype=np.int64, ndmin=2)
        degree = np.bincount(edges.ravel(), minlength=args.n)
        weights, target, ledger = cl_core.release(degree, cl_core.triangle_count(edges, args.n),
                                                   args.epsilon, args.seed)
        seconds = math.inf if args.profile == "large-cl-boundary" else 3600
        synthetic, diagnostics = cl_core.generate(weights, target, args.seed, max_seconds=seconds)
        if not math.isfinite(seconds):
            diagnostics["max_repair_seconds"] = None
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
                                                  args.epsilon, seed=args.seed, threads=1)
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
