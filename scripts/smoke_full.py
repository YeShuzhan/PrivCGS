"""Compile PrivCGS and verify an end-to-end Full run on a public test graph."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import ROOT, prepare, profile_for, read, route, sha, validated_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--compiler")
    args = parser.parse_args()
    n_nodes = 128
    edges = sorted({tuple(sorted((u, (u + offset) % n_nodes)))
                    for u in range(n_nodes) for offset in (1, 2, 7, 16)})
    with TemporaryDirectory(prefix="privcgs-smoke-") as directory:
        root = Path(directory)
        raw = root / "public_fixture.txt"
        np.savetxt(raw, edges, fmt="%d")
        ds = {"n_nodes": n_nodes, "n_edges": len(edges)}
        prepare(SimpleNamespace(raw=raw, dataset="Polblogs", nodes=None, delimiter=None,
                                output=root / "inputs", format="lf"), dict(ds, input_sha256=[]))
        graph = root / "inputs/graph.txt"
        ds["input_sha256"] = [sha(graph)]
        entry = profile_for("PrivCGS", "Polblogs", None, read(ROOT / "configs/implementations.json"))
        files = validated_files(entry, root / "external", True)
        output = root / "internal"
        route(SimpleNamespace(dataset="Polblogs", graph=graph, output=output,
                              cpu=str(args.cpu) if args.cpu is not None else None,
                              dry_run=False, epsilon=5.0, seed=42, compiler=args.compiler),
              ds, entry, files, dict(os.environ, PYTHONHASHSEED="42"))
        release = read(output / "release/release_manifest.json")
        report = read(output / "generation/generation_report.json")
        assert abs(release["rho_degree"] + release["rho_operator"] + release["rho_closure"]
                   - release["rho_total"]) < 1e-12
        assert report["n"] == n_nodes and report["m"] > 0
        assert report["status"] == "GENERATED_AWAITING_EVALUATION"
        assert report["new_private_queries"] == 0 and report["H0_H_G_same_degree_sequence"]
        edge_file = output / "generation/edges.edgelist"
        generated = np.loadtxt(edge_file, dtype=np.int64, ndmin=2)
        assert generated.shape == (report["m"], 2)
        assert np.all(generated[:, 0] >= 0) and np.all(generated[:, 1] < n_nodes)
        assert np.all(generated[:, 0] < generated[:, 1])
        assert len(np.unique(generated, axis=0)) == len(generated)
        target = np.load(output / "generation/target_degree.npy", allow_pickle=False)
        assert np.array_equal(np.bincount(generated.ravel(), minlength=n_nodes), target)
        assert sha(edge_file) == report["graph_sha256"]
        exported = root / "exported.edgelist"
        subprocess.run([sys.executable, str(ROOT / "run.py"), "export-graph", "--method", "PrivCGS",
                        "--run-dir", str(output), "--output", str(exported)], check=True)
        assert sha(exported) == sha(edge_file)
        assert ((output / "release/private_provenance.json").stat().st_mode & 0o077) == 0
        print(json.dumps({"status": "PASS", "vertices": n_nodes, "generated_edges": len(generated),
                          "graph_sha256": sha(exported)}))


if __name__ == "__main__":
    main()
