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
def read_arrays(path,n_expected,m_expected):
 # All selected source files have integer public IDs. Other formats fail closed.
 edges=np.loadtxt(path,dtype=np.int64,comments=('#','%'),usecols=(0,1),ndmin=2)
 edges=edges[edges[:,0]!=edges[:,1]]
 public=np.unique(edges)
 assert len(public)==n_expected,(len(public),n_expected)
 edges=np.searchsorted(public,edges)
 edges.sort(axis=1)
 keys=edges[:,0]*np.int64(n_expected)+edges[:,1];del edges
 keys=np.unique(keys);assert len(keys)==m_expected,(len(keys),m_expected)
 canonical=np.column_stack((keys//n_expected,keys%n_expected));del keys;gc.collect()
 return canonical,public
def load_csr(path,n,m,CSRGraph):
 edges,public=read_arrays(path,n,m)
 graph=CSRGraph.from_edges(edges,n);del edges,public;gc.collect()
 result=SimpleNamespace(csr_offsets=graph.csr_offsets,csr_neighbors=graph.csr_neighbors)
 assert len(result.csr_offsets)==n+1 and len(result.csr_neighbors)==2*m
 del graph;gc.collect();return result
