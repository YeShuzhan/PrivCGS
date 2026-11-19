"""Relocate the frozen release and generation stages for a single Full run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-nodes", type=int, required=True)
    parser.add_argument("--n-edges", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error("C++ compiler not found. Use --compiler to select it.")
    src = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = out / "configuration.json"
    fixed = {
        "dataset": args.dataset, "epsilon": args.epsilon, "delta": 1e-5, "seed": args.seed,
        "variant": "full", "budget": [.3, .3, .4], "r": 32, "alpha": .05, "zeta": .05,
    }
    cfg.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf-8")
    shared = ["--dataset", args.dataset, "--seed", str(args.seed), "--cpu", args.cpu,
              "--compiler", compiler]
    python = os.environ.get("PRIVCGS_PYTHON", sys.executable)
    commands = {
        "compile": [python, str(src / "linux_entry.py"), "compile"],
        "release": [python, str(src / "linux_entry.py"), "release", *shared,
                    "--graph", str(args.graph.resolve()), "--n-nodes", str(args.n_nodes),
                    "--n-edges", str(args.n_edges), "--epsilon", str(args.epsilon),
                    "--r", str(fixed["r"]), "--fractions", *map(str, fixed["budget"]),
                    "--output", str(out / "release")],
        "generation": [python, str(src / "study_entry.py"), str(cfg), *shared, "--verify-imports",
                       "--release-manifest", str(out / "release/release_manifest.json"),
                       "--output", str(out / "generation")],
    }
    env = dict(os.environ, CXX=compiler, PYTHONHASHSEED=str(args.seed),
               PRIVCGS_THREADS=str(len(args.cpu.split(","))))
    for phase, command in commands.items():
        try:
            subprocess.run(command, env=env, check=True)
        finally:
            if phase == "compile":
                for name in ("COMPILE.stdout", "COMPILE.stderr", "COMPILE.PARENT_NATIVE.json"):
                    origin = src / "equivalent" / name
                    if origin.is_file():
                        shutil.copyfile(origin, out / name)
    print(out / "generation/edges.edgelist")


if __name__ == "__main__":
    main()
