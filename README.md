# PrivCGS

[Introduction](#1-introduction) · [Requirements](#2-requirements) · [Datasets](#3-datasets) · [Usage](#4-usage) · [Verification](#verification)

## 1 Introduction

This repository contains the graph generation code for the main experiments with PrivCGS Full and nine baselines. The baselines use a common centralized Gaussian benchmark setting. We calibrate the Gaussian components using ρ-zCDP under the same target `(epsilon, delta)` to control differences in noise calibration across methods. Where source code was available, we adapted it for this setting.

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
- [DPGVAE and DPGGAN (CUDA adapters)](methods/dpg-cuda)
- [PrivDPR (CUDA)](methods/privdpr-cuda)

[Source and profile details](docs/sources.md) lists the original repositories where available and explains how to obtain and verify external files.

## 2 Requirements

Use Linux, Python 3.9 or 3.10, and a C++17 compiler with OpenMP support, such as `g++`.

```bash
git clone https://github.com/YeShuzhan/PrivCGS.git
cd PrivCGS
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the examples from the repository root. DPGVAE, DPGGAN, and PrivDPR use the [PyTorch 2.8.0 CUDA 12.6 build](https://pytorch.org/get-started/previous-versions/#v280), which requires Linux x86-64 with glibc 2.28 or newer and a compatible NVIDIA GPU and driver.

```bash
python -m pip install -r requirements-neural.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

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

### Run PrivCGS in benchmark mode

```bash
python run.py run --method PrivCGS --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-privcgs-e1-s42
```

The generated edge list is `runs/facebook-privcgs-e1-s42/generation/edges.edgelist`.

### Run PrivCGS in private mode

Provide a public node file with one integer ID per line in fixed order, including isolates. The private edge list uses those IDs. Set the node file independently of the private edge list and keep it identical across neighboring graphs.

```bash
python run.py run --method PrivCGS --dataset Facebook --epsilon 1 --mode private \
  --nodes public-nodes.txt --public-seed 13 --synthesis-seed 42 \
  --graph private-edges.txt --output runs/private-privcgs
```

The public seed controls the Fourier coordinates and projection priorities. The synthesis seed controls post-processing. Private Gaussian draws use fresh operating-system randomness on every release. Export the synthetic edge list alone and interpret its zero-based vertex IDs in the supplied public order. The public node list specifies the full domain, including vertices absent from the exported edge list.

### Run a baseline (PrivDPR on GPU)

```bash
python run.py run --method PrivDPR --dataset Facebook \
  --epsilon 1 --seed 42 --graph data/Facebook/graph.txt \
  --output runs/facebook-privdpr-cuda-e1-s42
```

The generated edge list is `runs/facebook-privdpr-cuda-e1-s42/synthetic.edgelist`. To run another baseline, change `--method` and use a new output directory. Tricycle selects its Gaussian Ladder settings by dataset. TmF selects its storage automatically.

### CPU resources and run outputs

The launcher defaults to one logical CPU for host-side work. Use `--cpu ID` to select a core. PrivCGS Full also accepts a comma-separated CPU list for parallel runs. The neural baselines use the first visible GPU. Use `CUDA_VISIBLE_DEVICES` to select a GPU.

Treat the run directory as internal benchmark data. It contains experimental checkpoints, diagnostics and input provenance. To copy just the generated edge list into a separate location, use:

```bash
python run.py export-graph --method PrivCGS --run-dir runs/facebook-privcgs-e1-s42 \
  --output facebook-privcgs.edgelist
```

Use benchmark mode with the public experiment inputs. Its fixed seed also controls measurement noise for reproducibility. Use private mode for sensitive inputs. Keep the run directories and their diagnostics internal in both modes.

## Verification

```bash
python scripts/verify_package.py
python scripts/smoke_full.py
```

`verify_package.py` checks packaged files and launch plans. `smoke_full.py` compiles PrivCGS and checks a complete Full run on a small public test graph, including the degree invariant and standalone edge-list export. Add `--dry-run` to a run command to inspect its launch plan.

The [full paper](PrivCGS.pdf) reports results from frozen historical implementations. See the [source version check](docs/sources.md#privcgs-source-version-and-run-check) for the archived Full run, the earlier public package and the implementation changes in this release.

## License

The project-owned code listed in [LICENSE](LICENSE) is available under the MIT License. Third-party code and dependencies retain their respective licenses and notices. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Acknowledgments

We thank Quan Yuan, an author of PrivGraph, for sharing the implementations of all baselines evaluated in the PrivGraph paper.
