# PrivCGS

[Introduction](#1-introduction) · [Requirements](#2-requirements) · [Datasets](#3-datasets) · [Usage](#4-usage) · [Verification](#verification)

## 1 Introduction

This repository provides the graph generation code for the main experiments with PrivCGS Full and the nine baselines below. All baselines run in a centralized benchmark setting with a common target `(epsilon, delta)`. We use ρ-zCDP to calibrate the Gaussian components under that target so that differences in noise calibration are easier to control across methods. Where we obtained source implementations, the benchmark versions adapt that code. Additional ablation and sensitivity experiments are outside this repository's scope.

**Table 1. Baseline papers**

| Method | Paper | Venue | Year |
| --- | --- | --- | :-: |
| LDPGen | [Generating Synthetic Decentralized Social Graphs with Local Differential Privacy](https://doi.org/10.1145/3133956.3134086) | CCS | 2017 |
| TmF | [Differentially Private Publication of Social Graphs at Linear Cost](https://doi.org/10.1145/2808797.2809385) | ASONAM | 2015 |
| DER | [Correlated Network Data Publication via Differential Privacy](https://doi.org/10.1007/s00778-013-0344-8) | The VLDB Journal | 2014 |
| PrivHRG | [Differentially Private Network Data Release via Structural Inference](https://doi.org/10.1145/2623330.2623642) | KDD | 2014 |
| PrivGraph | [PrivGraph: Differentially Private Graph Data Publication by Exploiting Community Information](https://www.usenix.org/conference/usenixsecurity23/presentation/yuan-quan) | USENIX Security | 2023 |
| Tricycle | [Publishing Attributed Social Graphs with Formal Privacy Guarantees](https://doi.org/10.1145/2882903.2915215) | SIGMOD | 2016 |
| DPGVAE | [Secure Deep Graph Generation with Link Differential Privacy](https://www.ijcai.org/proceedings/2021/450) | IJCAI | 2021 |
| DPGGAN | [Secure Deep Graph Generation with Link Differential Privacy](https://www.ijcai.org/proceedings/2021/450) | IJCAI | 2021 |
| PrivDPR | [PrivDPR: Synthetic Graph Publishing with Deep PageRank under Differential Privacy](https://doi.org/10.1145/3690624.3709334) | KDD | 2025 |

Please cite the PrivCGS paper and the relevant baseline papers.

### Bundled source code

- [PrivCGS Full](methods/privcgs)
- [LDPGen](methods/ldpgen-signed)
- [Tricycle (Gaussian Ladder)](methods/tricycle-gaussian-ladder)
- [PrivDPR (CPU and CUDA)](methods/privdpr-cpu)

The remaining routes use pinned external files. [Source and profile details](docs/sources.md) describes how to obtain and verify them.

## 2 Requirements

Use Linux, Python 3.9 or 3.10, and a C++17 compiler with OpenMP support, such as `g++`.

```bash
git clone https://github.com/YeShuzhan/PrivCGS.git
cd PrivCGS
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the examples from the repository root. For the PrivDPR GPU example, install the [PyTorch 2.8.0 CUDA 12.6 build](https://pytorch.org/get-started/previous-versions/#v280) on a machine with a compatible NVIDIA driver.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

For PrivDPR on CPU, DPGVAE, or DPGGAN, use `python -m pip install -r requirements-neural.txt`.

## 3 Datasets

Download an input graph from [the dataset manifest](configs/datasets.json) and prepare its edge list. The examples below use Facebook.

```bash
mkdir -p data/raw
curl -fL https://snap.stanford.edu/data/facebook_combined.txt.gz -o data/raw/facebook_combined.txt.gz
python run.py prepare --dataset Facebook --raw data/raw/facebook_combined.txt.gz \
  --format crlf --output data/Facebook
```

Use the prepared `data/Facebook/graph.txt` as `--graph`. Other input formats are listed in the [dataset preparation guide](docs/data.md).

## 4 Usage

### Run PrivCGS

```bash
python run.py run --method PrivCGS --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-privcgs-e1-s42
```

The generated edge list is `runs/facebook-privcgs-e1-s42/generation/edges.edgelist`.

### Run a baseline (PrivDPR on GPU)

```bash
python run.py run --method PrivDPR --profile privdpr-cuda --dataset Facebook \
  --epsilon 1 --seed 42 --graph data/Facebook/graph.txt \
  --output runs/facebook-privdpr-cuda-e1-s42
```

The generated edge list is `runs/facebook-privdpr-cuda-e1-s42/synthetic.edgelist`. Use `--profile privdpr-cpu` to run on CPU. For another baseline, change `--method`, omit the PrivDPR profile, and use a new output directory.

### CPU resources and run outputs

The launcher defaults to one logical CPU for each method. Use the same `--cpu ID` and host when comparing CPU routes. PrivCGS Full also accepts a comma-separated CPU list for explicitly parallel runs. Record the device, thread count and timed stages when reporting runtimes. Report CUDA runs with their GPU configuration.

Treat the run directory as internal benchmark data. It contains experimental checkpoints, diagnostics and input provenance. To copy just the generated edge list into a separate location, use:

```bash
python run.py export-graph --method PrivCGS --run-dir runs/facebook-privcgs-e1-s42 \
  --output facebook-privcgs.edgelist
```

The export is a benchmark artifact. The fixed seeds make experiments repeatable where the method uses seeded randomness. Production releases on sensitive inputs require fresh, secret randomness for private measurements.

## Verification

```bash
python scripts/verify_package.py
python scripts/smoke_full.py
```

`verify_package.py` checks packaged files and launch plans. `smoke_full.py` compiles PrivCGS and checks a complete Full run on a small public test graph, including the degree invariant and standalone edge-list export. Add `--dry-run` to a run command to inspect its launch plan.

## Acknowledgments

We thank Quan Yuan, an author of PrivGraph, for sharing the implementations of all baselines evaluated in the PrivGraph paper.
