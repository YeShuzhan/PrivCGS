# PrivCGS

Paper reproduction code for PrivCGS and nine graph synthesis baselines. This repository provides commands for graph preparation and single-run generation.

## Install

Use Linux, Python 3.9 or 3.10, and a C++17 compiler with OpenMP. Python packages can be installed in an isolated environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The three neural methods require the additional packages in `requirements-neural.txt`. Match the PyTorch build to the selected execution route. The [source ledger](docs/sources.md) records author sources, local adaptations and files that must be obtained separately. Source verification occurs before a command is launched.

## Prepare a dataset

Download an original graph from the source in [the dataset manifest](configs/datasets.json). Extract its edge list with the corresponding source's documented format. Then prepare and verify it against the paper's vertex counts, edge counts and available input hashes.

```bash
mkdir -p data/raw
curl -L https://snap.stanford.edu/data/facebook_combined.txt.gz -o data/raw/facebook_combined.txt.gz
python run.py prepare --dataset Facebook --raw data/raw/facebook_combined.txt.gz \
  --format crlf --output data/Facebook
```

For an already prepared graph, use `--graph` with the canonical edge list. Graphs are simple, undirected and use a dense public vertex index. Canonicalization removes loops, deduplicates undirected edges, and uses a fixed node order. Dataset-specific extraction and ordering matter, especially for an ID-aware method. See [dataset preparation](docs/data.md). Public benchmark settings are recorded in [the protocol](configs/protocol.json).

## Run a method

```bash
python run.py run --method PrivCGS --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-privcgs-e1-s42

python run.py run --method LDPGen --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --output runs/facebook-ldpgen-e1-s42

python run.py run --method PrivGraph --dataset Facebook --epsilon 1 --seed 42 \
  --graph data/Facebook/graph.txt --source-root external --output runs/facebook-privgraph-e1-s42
```

Append `--dry-run` to inspect an execution plan. `python run.py sources --method PrivGraph` lists required files and hashes. An approved bundle can be supplied under `--source-root` with the exact relative paths from that command. The default PrivCGS command runs the frozen Full release and generation stages. Its final edge list is `generation/edges.edgelist` under the requested output directory. Baseline commands write `synthetic.edgelist`. The [accounting notes](docs/baseline-accounting.md) document the privacy scope, including fixed benchmark noise and local diagnostics.

The [release checklist](docs/release-checklist.md) records items requiring author review before publication.

`python scripts/verify_package.py` checks the source manifest and all launch plans without evaluating a graph. Pass `--source-root external` to also verify the external bytes.

## Cite

Please cite the PrivCGS paper and the relevant upstream methods listed in [the source ledger](docs/sources.md). The paper's final bibliographic entry can be inserted when its citation is public. Third-party rights remain with their respective owners.
