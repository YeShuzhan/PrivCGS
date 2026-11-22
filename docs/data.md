# Dataset preparation

[Dataset manifest](../configs/datasets.json) records the benchmarks, sizes, source pages and known digests. Download the raw data from the publishers under their terms. Raw graphs and ID lists are external resources. The different releases of a named network may use different endpoints or node orders.

Extract the two integer endpoint columns from the stated release. Strip archive headers and convert GML or Matrix Market as specified by the source. Keep node IDs and, when using a separately recorded public vertex list, its order. The `prepare` command checks vertex and edge counts, maps to 0-based IDs, drops self-loops, normalizes endpoints and removes duplicate undirected edges. Its `--nodes` option preserves an explicitly supplied public node order and keeps isolates. The default order is ascending observed numeric ID. For Matrix Market inputs, remove the dimension line as well as the comment header. For a GML source, use its node IDs and both endpoints from the edge records.

The study uses the 513,969-node Flickr representation, distinct from the archived larger Flickr variant. Week-long or signed data releases with similar names can differ in their vertex list or edges. Reproducibility depends on the input SHA-256 and node order as well as graph sizes. If canonical bytes differ in delimiter, order or actual IDs, recover the matching frozen representation from the original data source or the replication authors. Use `--allow-unverified-input` only for a new benchmark instance where the discrepancy is intentional.

The experiment values are six target epsilons and five public generation seeds in [the protocol](../configs/protocol.json). Evaluation uses the original graph independently of generation.
