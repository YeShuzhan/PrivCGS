"""Run the release and generation stages for a single PrivCGS Full run."""
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
    parser.add_argument("--n-nodes", type=int)
    parser.add_argument("--n-edges", type=int)
    parser.add_argument("--nodes", type=Path)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--mode", choices=("benchmark", "private"), default="benchmark")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--public-seed", type=int)
    parser.add_argument("--synthesis-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    if args.mode == "private":
        if args.nodes is None or any(seed is None or not 0 <= seed < 2**32
                                     for seed in (args.public_seed, args.synthesis_seed)):
            parser.error("Private mode requires --nodes, --public-seed and --synthesis-seed.")
        if any(value is not None for value in (args.seed, args.n_nodes, args.n_edges)):
            parser.error("Pass benchmark counts and --seed in benchmark mode only.")
        seed = args.synthesis_seed
        release_args = ["--public-seed", str(args.public_seed)]
    else:
        if args.seed is None or args.n_nodes is None or args.n_edges is None:
            parser.error("Benchmark mode requires --seed, --n-nodes and --n-edges.")
        if args.public_seed is not None or args.synthesis_seed is not None:
            parser.error("Pass separate seeds in private mode only.")
        seed = args.seed
        release_args = ["--seed", str(seed), "--n-nodes", str(args.n_nodes), "--n-edges", str(args.n_edges)]
    if args.nodes is not None:
        release_args.extend(["--nodes", str(args.nodes.resolve())])
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error("C++ compiler not found. Use --compiler to select it.")
    src = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = out / "configuration.json"
    fixed = {
        "dataset": args.dataset, "epsilon": args.epsilon, "delta": 1e-5, "seed": seed,
        "randomness_mode": args.mode,
        "variant": "full", "budget": [.3, .3, .4], "r": 32, "alpha": .05, "zeta": .05,
    }
    if args.mode == "private":
        fixed.update(public_seed=args.public_seed, synthesis_seed=seed)
    cfg.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf-8")
    shared = ["--dataset", args.dataset, "--cpu", args.cpu, "--compiler", compiler]
    python = os.environ.get("PRIVCGS_PYTHON", sys.executable)
    commands = {
        "compile": [python, str(src / "linux_entry.py"), "compile"],
        "release": [python, str(src / "linux_entry.py"), "release", *shared,
                    "--graph", str(args.graph.resolve()), "--mode", args.mode,
                    *release_args, "--epsilon", str(args.epsilon),
                    "--r", str(fixed["r"]), "--fractions", *map(str, fixed["budget"]),
                    "--output", str(out / "release")],
        "generation": [python, str(src / "full_entry.py"), str(cfg), *shared, "--seed", str(seed), "--verify-imports",
                       "--release-manifest", str(out / "release/release_manifest.json"),
                       "--output", str(out / "generation")],
    }
    env = dict(os.environ, CXX=compiler, PYTHONHASHSEED=str(seed),
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
