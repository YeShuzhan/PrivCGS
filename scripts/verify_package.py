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
    assert {r["method"] for r in entries} == methods
    assert all(Path(name).parts[0] not in {"results", "data", "runs", "external"} for name in manifest)
    assert len(list((ROOT / "methods/privcgs/standalone").rglob("*.cpp"))) > 10
    for entry in entries:
        runner = [sys.executable, str(ROOT / "run.py")]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        if args.source_root:
            subprocess.run([*runner, "sources", "--method", entry["method"], "--profile", entry["profile"],
                            "--source-root", str(args.source_root), "--verify"], env=env, check=True)
        cmd = [*runner, "run", "--method", entry["method"], "--dataset", entry["datasets"][0],
               "--profile", entry["profile"], "--epsilon", "1", "--seed", "42",
               "--graph", "unused-input.txt", "--output", "unused-output", "--dry-run"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        assert isinstance(json.loads(result.stdout), list), entry["profile"]
    print(f"Verified {len(manifest)} files and {len(entries)} launch plans across {len(methods)} methods.")


if __name__ == "__main__":
    main()
