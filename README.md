# PrivCGS

[Introduction](#1-introduction) · [Requirements](#2-requirements) · [Datasets](#3-datasets) · [Usage](#4-usage) · [Verification](#verification)

## 1 Introduction

Use the same command line to prepare datasets and run PrivCGS or any of the nine graph synthesis baselines.

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

## 2 Requirements

Use Linux, Python 3.9 or 3.10, and a C++17 compiler with OpenMP support, such as `g++`.

```bash
git clone https://github.com/YeShuzhan/PrivCGS.git
cd PrivCGS
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the examples from the repository root. DPGVAE, DPGGAN, and PrivDPR also require `python -m pip install -r requirements-neural.txt` and a compatible PyTorch build.

### Source availability

| Source | Methods |
| --- | --- |
| Included in this repository | PrivCGS, LDPGen, Tricycle (Gaussian Ladder) |
| Available from the original authors | DPGVAE, DPGGAN |
| Request from the PrivCGS maintainers | TmF (`tmf-dense` or `tmf-sparse`), DER (`der-author`), PrivHRG (`privhrg-gaussian`), PrivGraph (`privgraph-n20`), and our PyTorch implementation of PrivDPR (`privdpr-cpu`) |

DPGVAE and DPGGAN share the same upstream source. Download it once.

```bash
git clone https://github.com/haonan3/Secure-Network-Release-with-Link-Privacy ../dpg-upstream
mkdir -p external/dpg-cpu/vendor/dpg_author
git -C ../dpg-upstream archive cbc28d38346921bf06147ed7a63494132f3c207d src \
  | tar -x -C external/dpg-cpu/vendor/dpg_author
```

For the other external bundles, [contact the maintainers](https://github.com/YeShuzhan/PrivCGS/issues/new) with the method and profile. The `sources` command lists the required file paths. Place the files at those paths under `external/`.

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

### Run a baseline

```bash
python run.py run --method Tricycle --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-tricycle-e1-s42
```

The generated edge list is `runs/facebook-tricycle-e1-s42/synthetic.edgelist`. To run another baseline, change `--method` and use a new `--output` directory. Use `--profile` to select a specific implementation.

## Verification

```bash
python scripts/verify_package.py
```

For an external bundle, list its files and verify their SHA-256 hashes. The example uses PrivDPR.

```bash
python run.py sources --method PrivDPR --profile privdpr-cpu
python run.py sources --method PrivDPR --profile privdpr-cpu --source-root external --verify
```

Add `--dry-run` to a run command to inspect its launch plan.

## Acknowledgments

We thank Quan Yuan, an author of PrivGraph, for sharing the LDPGen, TmF, and DER baseline code.
