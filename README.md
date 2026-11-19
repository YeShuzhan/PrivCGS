# PrivCGS

Paper reproduction code for PrivCGS and nine graph synthesis baselines, with a shared command line for preparing graph datasets and running individual methods.

[Requirements](#1-requirements) · [Datasets](#2-datasets) · [Run PrivCGS](#3-run-privcgs) · [Baselines](#4-baselines) · [Verification](#5-verification)

## 1. Requirements

Use Linux and Python 3.9 or 3.10. PrivCGS builds its native components locally with a C++17 compiler that supports OpenMP, such as `g++`.

```bash
git clone https://github.com/YeShuzhan/PrivCGS.git
cd PrivCGS
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Substitute `python3.10` if using Python 3.10. Run the examples below from the repository root. DPGVAE, DPGGAN, and PrivDPR also require [the neural dependencies](requirements-neural.txt) and a PyTorch build compatible with your environment.

## 2. Datasets

Download an input graph from [the dataset manifest](configs/datasets.json) and prepare its edge list. The examples below use Facebook.

```bash
mkdir -p data/raw
curl -fL https://snap.stanford.edu/data/facebook_combined.txt.gz -o data/raw/facebook_combined.txt.gz
python run.py prepare --dataset Facebook --raw data/raw/facebook_combined.txt.gz \
  --format crlf --output data/Facebook
```

Pass the canonical edge list to `--graph`. Preparation removes loops, deduplicates undirected edges, and uses a fixed dense vertex order. See the [dataset preparation guide](docs/data.md) for input formats and [the protocol](configs/protocol.json) for budget and seed settings.

## 3. Run PrivCGS

The Full release and generation stages are bundled with the repository.

```bash
python run.py run --method PrivCGS --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-privcgs-e1-s42
```

The generated edge list is `runs/facebook-privcgs-e1-s42/generation/edges.edgelist`.

## 4. Baselines

| Source availability | Methods | What to do after cloning |
| --- | --- | --- |
| [Bundled](#bundled-baselines) | **LDPGen**, **Tricycle** | Install `requirements.txt` and prepare the input graph. |
| [Available from the original authors](#source-from-the-original-authors) | **DPGVAE**, **DPGGAN** | Install `requirements-neural.txt` and obtain the shared source files from the pinned upstream commit. |
| [Request from the PrivCGS maintainers](#source-from-the-privcgs-maintainers) | **TmF**, **DER**, **PrivHRG**, **PrivGraph**, **PrivDPR** | Request the exact frozen adaptation for the selected profile. PrivDPR also requires `requirements-neural.txt`. |

### Bundled baselines

LDPGen includes the signed-total implementation. Tricycle includes both the standard and large-graph profiles. For example, the following commands verify the bundled files and run LDPGen on the prepared Facebook graph.

```bash
python run.py sources --method LDPGen --verify
python run.py sources --method Tricycle --verify

python run.py run --method LDPGen --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-ldpgen-e1-s42
```

### Source from the original authors

DPGVAE and DPGGAN share one source bundle from the [authors' repository](https://github.com/haonan3/Secure-Network-Release-with-Link-Privacy). Extract its `src` directory from commit `cbc28d38346921bf06147ed7a63494132f3c207d` into the location expected by the runner.

```bash
python -m pip install -r requirements-neural.txt
git clone https://github.com/haonan3/Secure-Network-Release-with-Link-Privacy ../dpg-upstream
mkdir -p external/dpg-cpu/vendor/dpg_author
git -C ../dpg-upstream archive cbc28d38346921bf06147ed7a63494132f3c207d src \
  | tar -x -C external/dpg-cpu/vendor/dpg_author

python run.py sources --method DPGVAE --source-root external --verify
python run.py sources --method DPGGAN --source-root external --verify

python run.py run --method DPGVAE --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --source-root external \
  --output runs/facebook-dpgvae-e1-s42
```

### Source from the PrivCGS maintainers

[Contact the maintainers](https://github.com/YeShuzhan/PrivCGS/issues/new) with the method and profile you need. Use the `sources` command to list the exact file paths and SHA-256 hashes. Place the files you obtain under `external/`, preserving the paths printed by the command.

| Method | Profile to request |
| --- | --- |
| TmF | `tmf-dense` for the dense route or `tmf-sparse` for the large-graph route |
| DER | `der-author` |
| PrivHRG | `privhrg-gaussian` |
| PrivGraph | `privgraph-n20` |
| PrivDPR | `privdpr-cpu` |

For example, list and verify the N20 files, then run that profile on Facebook.

```bash
python run.py sources --method PrivGraph --profile privgraph-n20
python run.py sources --method PrivGraph --profile privgraph-n20 --source-root external --verify

python run.py run --method PrivGraph --profile privgraph-n20 --dataset Facebook \
  --epsilon 1 --seed 42 --graph data/Facebook/graph.txt --source-root external \
  --output runs/facebook-privgraph-e1-s42
```

Select the profile explicitly when verifying one TmF route, for example `python run.py sources --method TmF --profile tmf-sparse --source-root external --verify`. Baseline runs write `synthetic.edgelist` under the chosen output directory.

## 5. Verification

```bash
python scripts/verify_package.py
```

This checks the repository file manifest and launch plans without evaluating a graph. Append `--dry-run` to an individual run command to print its launch plan. Use `sources --verify` to check the SHA-256 hashes for an acquired baseline profile.

See the [source ledger](docs/sources.md) for provenance and acquisition details, the [baseline accounting notes](docs/baseline-accounting.md) for privacy scope, and the [release checklist](docs/release-checklist.md) for outstanding source checks.

## 6. Citation

Please cite the PrivCGS paper and the upstream methods listed in the [source ledger](docs/sources.md). Third-party rights remain with their respective owners.
