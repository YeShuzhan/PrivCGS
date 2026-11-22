# PrivCGS

[Introduction](#1-introduction) · [Requirements](#2-requirements) · [Datasets](#3-datasets) · [Usage](#4-usage)

## 1 Introduction

This repository provides a shared command line for PrivCGS and the nine graph synthesis baselines used in the paper. It prepares benchmark datasets and runs each method with the bundled or separately acquired source files listed below.

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

For matched settings, the utility comparison uses the same prepared graphs, evaluation metrics, and target `(epsilon, delta)`, with `delta = 1e-5`. We use zCDP to calibrate the Gaussian components and cite the original papers for the privacy guarantees of their mechanisms. See the [baseline accounting notes](docs/baseline-accounting.md) and [source ledger](docs/sources.md) for the implementation details.

Please cite the PrivCGS paper and the relevant baseline papers.

## 2 Requirements

Use Linux and Python 3.9 or 3.10. PrivCGS builds its native components locally with a C++17 compiler that supports OpenMP, such as `g++`.

```bash
git clone https://github.com/YeShuzhan/PrivCGS.git
cd PrivCGS
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Substitute `python3.10` if using Python 3.10. Run the examples below from the repository root. DPGVAE, DPGGAN, and PrivDPR also require [the neural dependencies](requirements-neural.txt) and a PyTorch build compatible with your environment.

## 3 Datasets

Download an input graph from [the dataset manifest](configs/datasets.json) and prepare its edge list. The examples below use Facebook.

```bash
mkdir -p data/raw
curl -fL https://snap.stanford.edu/data/facebook_combined.txt.gz -o data/raw/facebook_combined.txt.gz
python run.py prepare --dataset Facebook --raw data/raw/facebook_combined.txt.gz \
  --format crlf --output data/Facebook
```

Pass the canonical edge list to `--graph`. Preparation removes loops, deduplicates undirected edges, and uses a fixed dense vertex order. See the [dataset preparation guide](docs/data.md) for input formats and [the protocol](configs/protocol.json) for budget and seed settings.

## 4 Usage

### Run PrivCGS

The Full release and generation stages are bundled with the repository.

```bash
python run.py run --method PrivCGS --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-privcgs-e1-s42
```

The generated edge list is `runs/facebook-privcgs-e1-s42/generation/edges.edgelist`.

This command reproduces the public-data benchmark. In the bundled PrivCGS profile, `--seed` controls both the benchmark noise and synthesis. A release on sensitive data requires a separate private noise stream. Keep raw-input provenance and other internal run records in the trusted environment.

### Baselines

| Source availability | Methods | What to do after cloning |
| --- | --- | --- |
| [Bundled](#bundled-baselines) | **LDPGen**, **Tricycle** | Install `requirements.txt` and prepare the input graph. |
| [Available from the original authors](#source-from-the-original-authors) | **DPGVAE**, **DPGGAN** | Install `requirements-neural.txt` and obtain the shared source files from the pinned upstream commit. |
| [Request from the PrivCGS maintainers](#source-from-the-privcgs-maintainers) | **TmF**, **DER**, **PrivHRG**, **PrivGraph**, **PrivDPR** | Request the source bundle for the selected reproduction profile. PrivDPR also requires `requirements-neural.txt`. |

#### Bundled baselines

LDPGen includes the signed-total central Gaussian adaptation. Tricycle requires a C++17 compiler. By default, the runner selects the Gaussian Ladder profile by dataset and uses its compact implementation for Flickr-G, YouTube, and WikiTalk. The commands below verify the bundled profiles and run both methods on Facebook.

```bash
python run.py sources --method LDPGen --verify
python run.py sources --method Tricycle --profile tricycle-gaussian-ladder --verify
python run.py sources --method Tricycle --profile tricycle-gaussian-ladder-large --verify

python run.py run --method LDPGen --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-ldpgen-e1-s42

python run.py run --method Tricycle --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-tricycle-e1-s42
```

The Gaussian Ladder profiles draw fresh private noise on each run. Their public seed controls reconstruction. Tables 4 and 5 in the [bundled PDF](PrivCGS.pdf) report the earlier CL-TriCycLe experiments with direct Gaussian perturbation of the triangle count. Select those profiles explicitly to use the PDF's benchmark method.

| Tricycle benchmark | Polblogs, Facebook, HepTh, HepPh, Enron, Epinions, Gowalla | Flickr-G, YouTube, WikiTalk |
| --- | --- | --- |
| Current default, Gaussian Ladder | `tricycle-gaussian-ladder` | `tricycle-gaussian-ladder-large` |
| Bundled PDF, direct Gaussian count | `tricycle-reference` | `large-cl-boundary` |

Both large-graph profiles use the same numerical boundary fix for NetworkX's expected-degree sampler. `large-cl-boundary` constructs a NetworkX graph and restores the initial edge order for FIFO repair. `tricycle-gaussian-ladder-large` first collects the edges in a compact array and falls back to NetworkX reconstruction if repair is needed. Use the corresponding profile when reproducing runtime or memory measurements.

For example, use the direct-count profile on Facebook:

```bash
python run.py run --method Tricycle --profile tricycle-reference --dataset Facebook \
  --epsilon 1 --seed 42 --graph data/Facebook/graph.txt --output runs/facebook-tricycle-pdf-e1-s42
```

The following commands select the two profiles explicitly for a prepared Flickr-G graph.

```bash
python run.py run --method Tricycle --profile tricycle-gaussian-ladder-large --dataset Flickr-G \
  --epsilon 1 --seed 42 --graph data/Flickr-G/graph.txt --output runs/flickr-tricycle-ladder-e1-s42

python run.py run --method Tricycle --profile large-cl-boundary --dataset Flickr-G \
  --epsilon 1 --seed 42 --graph data/Flickr-G/graph.txt --output runs/flickr-tricycle-pdf-e1-s42
```

See the [Tricycle accounting notes](docs/baseline-accounting.md#tricycle) for the measurement and reconstruction paths.

#### Source from the original authors

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

#### Source from the PrivCGS maintainers

[Contact the maintainers](https://github.com/YeShuzhan/PrivCGS/issues/new) with the method and profile you need. The `sources` command lists the expected paths and SHA-256 hashes for that profile. Place the files under `external/` using those paths. The [release checklist](docs/release-checklist.md) records the remaining checks against the original experiment runs.

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

### Verification

```bash
python scripts/verify_package.py
```

This checks the repository file manifest and launch plans without evaluating a graph. Append `--dry-run` to an individual run command to print its launch plan. Use `sources --verify` to check that an acquired bundle matches the listed profile hashes.

See the [source ledger](docs/sources.md) for provenance and acquisition details, the [baseline accounting notes](docs/baseline-accounting.md) for privacy scope, and the [release checklist](docs/release-checklist.md) for outstanding source checks.

Third-party rights remain with their respective owners.
