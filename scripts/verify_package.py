"""Verify source bytes and plan each wrapper without evaluating a graph."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    manifest = read(ROOT / "SOURCE_SHA256.json")
    for name, expected in manifest.items():
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), name
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
        if path.suffix == ".py":
            ast.parse(data, filename=name)
    entries = read(ROOT / "configs/implementations.json")
    methods = {"PrivCGS", "LDPGen", "TmF", "DER", "PrivHRG", "PrivGraph", "Tricycle",
               "DPGVAE", "DPGGAN", "PrivDPR"}
    assert len(entries) == len(methods) and {r["method"] for r in entries} == methods
    for method, profile in (("DPGVAE", "dpg-cuda"), ("DPGGAN", "dpg-cuda"), ("PrivDPR", "privdpr-cuda")):
        assert [r["profile"] for r in entries if r["method"] == method] == [profile]
    for entry in entries:
        routes = entry.get("dataset_routes", [entry])
        assigned = [dataset for route in routes for dataset in route["datasets"]]
        assert len(assigned) == len(set(assigned)) and set(assigned) == set(entry["datasets"])
        for route in routes:
            for binding in route["files"].values():
                if binding["bundled"]:
                    assert manifest[binding["path"]] == binding["sha256"], binding["path"]
    assert all(Path(name).parts[0] not in {"results", "data", "runs", "external"} for name in manifest)
    assert len(list((ROOT / "methods/privcgs/standalone").rglob("*.cpp"))) > 10
    runner = [sys.executable, str(ROOT / "run.py")]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    large_graphs = {"Flickr-G", "YouTube", "WikiTalk"}
    plans = 0
    for entry in entries:
        method = entry["method"]
        if args.source_root:
            subprocess.run([*runner, "sources", "--method", method,
                            "--source-root", str(args.source_root), "--verify"], env=env, check=True)
        for dataset in entry["datasets"]:
            cmd = [*runner, "run", "--method", method, "--dataset", dataset, "--epsilon", "1", "--seed", "42",
                   "--graph", "unused-input.txt", "--output", "unused-output", "--dry-run"]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            plan = json.loads(result.stdout)
            assert isinstance(plan, list), (method, dataset)
            if method == "Tricycle":
                assert ("--large-graph" in plan) == (dataset in large_graphs)
            elif method == "TmF":
                assert ("--mode" in plan) == (dataset in large_graphs)
                if dataset in large_graphs:
                    assert plan[plan.index("--mode") + 1] == "sparse"
            elif method == "DER":
                assert plan[plan.index("--der-counts") + 1] == "numeric_compat_v2"
            plans += 1
    print(f"Verified {len(manifest)} files and {plans} launch plans across {len(methods)} methods.")


if __name__ == "__main__":
    main()
