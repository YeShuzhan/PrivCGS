"""Integer-ID equivalent of dpgraph.data normalization without Python edge tuples."""
import gc,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read_arrays(path, n_expected=None, m_expected=None, *, nodes=None):
    if nodes is None:
        if n_expected is None or n_expected < 1:
            raise ValueError("Specify an ordered public node file or a positive dense-domain size.")
        public = np.arange(n_expected, dtype=np.int64)
    else:
        vertex_column = np.loadtxt(nodes, dtype=np.int64, comments=('#', '%'), ndmin=2)
        if vertex_column.shape[1] != 1 or not len(vertex_column):
            raise ValueError("The public node file must contain one integer ID per line.")
        public = vertex_column[:, 0]
        if n_expected is not None and len(public) != n_expected:
            raise ValueError("The public node file does not match the benchmark domain size.")
    order = np.argsort(public)
    ordered = public[order]
    if np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("The public node file contains duplicate IDs.")
    edges = np.loadtxt(path, dtype=np.int64, comments=('#', '%'), usecols=(0, 1), ndmin=2)
    indices = np.searchsorted(ordered, edges)
    if np.any(indices >= len(public)) or not np.array_equal(ordered[indices], edges):
        raise ValueError("An edge endpoint lies outside the public node domain.")
    edges = order[indices]
    del indices
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges.sort(axis=1)
    n = len(public)
    keys = np.unique(edges[:, 0] * np.int64(n) + edges[:, 1])
    del edges
    if m_expected is not None and len(keys) != m_expected:
        raise ValueError("The canonical edge count does not match the benchmark graph.")
    canonical = np.column_stack((keys // n, keys % n))
    return canonical, public
def load_csr(path,n,m,CSRGraph):
 edges,public=read_arrays(path,n,m)
 graph=CSRGraph.from_edges(edges,n);del edges,public;gc.collect()
 result=SimpleNamespace(csr_offsets=graph.csr_offsets,csr_neighbors=graph.csr_neighbors)
 assert len(result.csr_offsets)==n+1 and len(result.csr_neighbors)==2*m
 del graph;gc.collect();return result
