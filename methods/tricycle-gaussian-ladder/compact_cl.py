"""Replay the pinned expected-degree generator with a compact edge sink."""
from array import array
from pathlib import Path
import time
import numpy as np
import networkx as nx
import networkx.generators.degree_seq as degree_seq
from cl_boundary import build
from cl_core import triangle_count


def generate(weights,target,seed,**kw):
    start=time.perf_counter()
    sampler=build(Path(degree_seq.__file__))
    body=sampler.__wrapped__
    class Sink:
        def __init__(self):self.data=array('q')
        def add_edge(self,u,v):self.data.extend((min(u,v),max(u,v)))
    original_nx=body.__globals__['nx']
    class Proxy:
        @staticmethod
        def empty_graph(n):return Sink()
    body.__globals__['nx']=Proxy
    try:sink=sampler(weights.tolist(),seed=seed,selfloops=False)
    finally:body.__globals__['nx']=original_nx
    edges=np.frombuffer(sink.data,dtype=np.int64).reshape(-1,2)
    keys=edges[:,0]*len(weights)+edges[:,1]
    keys.sort()
    assert len(keys)<2 or np.all(keys[1:]>keys[:-1])
    result=np.column_stack((keys//len(weights),keys%len(weights)))
    del keys,edges,sink
    tri=triangle_count(result,len(weights))
    if tri<target and len(result):
        # Preserve the frozen repair including FIFO and RNG semantics.
        # Release inputs were not re-drawn.
        import cl_core
        del result
        return cl_core.generate(weights,target,seed,**kw)
    finish=time.perf_counter()-start
    m=len(result)
    return result,dict(n=len(weights),m=m,initial_triangles=tri,final_triangles=tri,target=target,
        initial_above_target=tri>target,target_reached=tri>=target,target_exact=tri==target,
        stop_reason='TARGET_REACHED_OR_EXCEEDED' if tri>=target else 'EMPTY_SUPPORT',
        proposals=0,accepted=0,positive_moves=0,rejected=0,skipped=0,cl_seconds=finish,
        postprocess_seconds=0.,total_generation_seconds=finish,max_proposals=max(10000,kw['proposals_per_edge']*max(m,1)),
        max_repair_seconds=kw['max_seconds'],edge_count_preserved=True,triangle_ledger_exact=True,
        degree_exact_not_promised=True,forced_connectivity=False,minimum_degree=0,
        initial_edge_age_policy='Frozen CL with an array edge sink, repair unnecessary')
