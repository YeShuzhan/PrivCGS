"""Portable command line for the frozen PrivCGS reproduction package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(args, ds):
    # numpys sorting handles multi-million-edge inputs without Python tuples per edge.
    import numpy as np
    opener = __import__("gzip").open if args.raw.suffix == ".gz" else open
    with opener(args.raw, "rt", encoding="utf-8") as stream:
        edges = np.loadtxt(stream, dtype=np.int64, ndmin=2, usecols=(0, 1),
                           delimiter=args.delimiter)
    vertices = np.unique(edges) if args.nodes is None else np.loadtxt(
        args.nodes, dtype=np.int64, ndmin=1)
    if len(vertices) != len(np.unique(vertices)):
        raise ValueError("The public vertex list contains duplicates")
    # Respect an explicit public node order.
    order = np.argsort(vertices)
    indices = np.searchsorted(vertices[order], edges)
    if np.any(indices >= len(vertices)) or not np.array_equal(vertices[order][indices], edges):
        raise ValueError("Edge endpoint lies outside the public vertex domain")
    canonical = np.sort(order[indices], axis=1)
    canonical = np.unique(canonical[canonical[:, 0] != canonical[:, 1]], axis=0)
    if (len(vertices), len(canonical)) != (ds["n_nodes"], ds["n_edges"]):
        raise ValueError(f"Graph size differs from the manifest: {len(vertices)} vertices, {len(canonical)} edges")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "graph.txt").open("w", encoding="ascii", newline="") as stream:
        np.savetxt(stream, canonical, fmt="%d", delimiter="\t" if args.format == "tsv" else " ",
                   newline="\r\n" if args.format == "crlf" else "\n")
    np.savetxt(args.output / "nodes.txt", vertices, fmt="%d", newline="\n")
    digest = sha(args.output / "graph.txt")
    save(args.output / "manifest.json", {"dataset": args.dataset, "sha256": digest,
         "known_frozen_bytes": digest in ds["input_sha256"], "n_nodes": len(vertices),
         "n_edges": len(canonical), "raw_sha256": sha(args.raw)})
    print("Canonical graph SHA-256:", digest)


def profile_for(method, dataset, explicit, entries):
    candidates = [entry for entry in entries if entry["method"] == method]
    if explicit:
        return next((entry for entry in candidates if entry["profile"] == explicit), None)
    return next((entry for entry in candidates if dataset in entry["datasets"]), None)


def validated_files(entry, external, check):
    files = []
    for name, record in entry["files"].items():
        root = ROOT if record["bundled"] else external
        path = root / record["path"]
        if check and (not path.is_file() or sha(path) != record["sha256"]):
            raise ValueError(f"Missing or different {path}. See 'python run.py sources --method {entry['method']}'")
        files.append((path, name))
    return files


def route(args, ds, entry, files, env):
    graph, output = args.graph.resolve(), args.output.resolve()
    available = sorted(os.sched_getaffinity(0))
    cpu_count = 1 if entry["kind"] != "privcgs" else 4
    cpus = args.cpu or ",".join(map(str, available[:cpu_count]))
    requested = [int(cpu) for cpu in cpus.split(",")]
    if not set(requested).issubset(available) or entry["kind"] != "privcgs" and len(requested) != 1:
        raise ValueError("Choose CPUs from the current affinity set. Baseline routes take one CPU.")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not args.dry_run:
        actual = sha(graph)
        if actual not in ds["input_sha256"] and not args.allow_unverified_input:
            raise ValueError("Graph SHA differs from the available frozen records. Check docs/data.md, or explicitly use --allow-unverified-input for a new benchmark.")
        output.parent.mkdir(parents=True, exist_ok=True)
    if entry["kind"] == "privcgs":
        cmd = [sys.executable, str(ROOT / "scripts/run_privcgs.py"), "--graph", str(graph),
               "--dataset", entry["dataset_names"].get(args.dataset, args.dataset),
               "--n-nodes", str(ds["n_nodes"]), "--n-edges", str(ds["n_edges"]),
               "--epsilon", str(args.epsilon), "--seed", str(args.seed), "--output", str(output),
               "--cpu", cpus, "--source", "<isolated-source-root>"]
        if args.compiler:
            cmd.extend(["--compiler", args.compiler])
        if args.dry_run:
            print(json.dumps(cmd, indent=2))
        else:
            with TemporaryDirectory(prefix="privcgs-native-", dir=output.parent) as name:
                bind(files, Path(name))
                cmd[cmd.index("<isolated-source-root>")] = name
                subprocess.run(cmd, env=env, check=True)
        return
    if entry["kind"] in ("direct", "direct-port"):
        cmd = [sys.executable, str(ROOT / "scripts/run_baseline.py"), "--method", args.method,
               "--graph", str(graph), "--output", str(output), "--n", str(ds["n_nodes"]),
               "--epsilon", str(args.epsilon), "--seed", str(args.seed), "--cpu", cpus,
               "--profile", entry["profile"], "--source", "<isolated-source-root>"]
        if args.compiler and entry["profile"].startswith("tricycle-gaussian-ladder"):
            cmd.extend(["--compiler", args.compiler])
        if args.dry_run:
            print(json.dumps(cmd, indent=2))
        else:
            with TemporaryDirectory(prefix="privcgs-source-", dir=output.parent) as name:
                bound = Path(name)
                bind(files, bound)
                cmd[cmd.index("<isolated-source-root>")] = name
                subprocess.run(cmd, env=env, check=True)
        return
    # The legacy runner requires artifacts under its source folder. Bind the
    # exact files into a per-run directory and keep the relocated config there.
    tmp_path = output.parent / "<isolated-source-root>"
    bound = None
    try:
        if not args.dry_run:
            bound = TemporaryDirectory(prefix="privcgs-source-", dir=output.parent)
            tmp_path = Path(bound.name)
            bind(files, tmp_path)
        sparse = entry["kind"] == "tmf-sparse"
        out = tmp_path / ("frozen_legacy" if sparse else "") / "artifacts"
        cell = f"{args.dataset}__{args.method}__e{args.epsilon:g}__seed{args.seed}"
        cmd = [sys.executable, str(tmp_path / entry["entry"]), "--input", str(graph),
               "--output", str(out), "--epsilon", str(args.epsilon),
               "--seed", str(args.seed), "--cpu", cpus]
        if sparse:
            cmd += ["--mode", "sparse", "--cell", cell]
        else:
            cmd += ["--method", args.method, "--n-nodes", str(ds["n_nodes"]),
                    "--dataset", entry["dataset_names"].get(args.dataset, args.dataset)]
            if args.method == "DER":
                cmd += ["--der-counts", entry["der_counts"]]
        if args.dry_run:
            print(json.dumps(cmd, indent=2))
            return
        if sparse:
            save(tmp_path / "CONFIG.json", {"epsilon_values": [args.epsilon], "seed_values": [args.seed]})
            save(tmp_path / "CATALOG.json", [{"cell": cell, "input": str(graph), "input_sha256": sha(graph),
                 "n_nodes": ds["n_nodes"], "dataset": args.dataset, "method": "TmF",
                 "epsilon": args.epsilon, "seed": args.seed}])
            save(tmp_path / "FROZEN_LEGACY_ORIGINS.json", {
                str(Path(name).relative_to("frozen_legacy")): {"sha256": entry["files"][name]["sha256"]}
                for _, name in files if Path(name).is_relative_to("frozen_legacy")
            })
        try:
            subprocess.run(cmd, env=env, check=True)
        finally:
            trace = out.with_name(out.name + ".trace.json")
            if trace.exists():
                out.mkdir(parents=True, exist_ok=True)
                shutil.move(str(trace), str(out / "trace.json"))
            if out.exists():
                out.rename(output)
    finally:
        if bound:
            bound.cleanup()


def bind(files, root):
    for path, name in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def main():
    datasets = read(ROOT / "configs/datasets.json")
    entries = read(ROOT / "configs/implementations.json")
    protocol = read(ROOT / "configs/protocol.json")
    methods = sorted({entry["method"] for entry in entries})
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", required=True, choices=datasets)
    prep.add_argument("--raw", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--nodes", type=Path, help="Optional explicit public node order, one ID per line")
    prep.add_argument("--delimiter", help="Input delimiter; the default is whitespace")
    prep.add_argument("--format", choices=("lf", "crlf", "tsv"), default="lf",
                      help="Canonical output serialization")
    run = commands.add_parser("run")
    run.add_argument("--dataset", required=True, choices=datasets)
    run.add_argument("--method", required=True, choices=methods)
    run.add_argument("--epsilon", required=True, type=float, choices=protocol["epsilon"])
    run.add_argument("--seed", type=int, default=42, choices=protocol["seeds"])
    run.add_argument("--graph", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--source-root", default=Path("external"), type=Path)
    run.add_argument("--profile")
    run.add_argument("--cpu", help="A CPU ID, or a comma-separated list for PrivCGS")
    run.add_argument("--compiler")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--allow-unverified-input", action="store_true")
    sources = commands.add_parser("sources")
    sources.add_argument("--method", required=True, choices=methods)
    sources.add_argument("--profile")
    sources.add_argument("--verify", action="store_true")
    sources.add_argument("--source-root", default=Path("external"), type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args, datasets[args.dataset])
        elif args.command == "sources":
            selected = [entry for entry in entries if entry["method"] == args.method
                        and (args.profile is None or entry["profile"] == args.profile)]
            if not selected:
                raise ValueError("Unknown source profile")
            if args.verify:
                for entry in selected:
                    count = len(validated_files(entry, args.source_root.resolve(), True))
                    print(f"{entry['profile']}: verified {count} files")
            else:
                print(json.dumps(selected, indent=2))
        else:
            entry = profile_for(args.method, args.dataset, args.profile, entries)
            if entry is None:
                parser.error("No default route for this dataset and method. See docs/sources.md.")
            env = dict(os.environ, PYTHONHASHSEED=str(args.seed), PYTHONDONTWRITEBYTECODE="1")
            files = validated_files(entry, args.source_root.resolve(), not args.dry_run)
            route(args, datasets[args.dataset], entry, files, env)
    except ValueError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
