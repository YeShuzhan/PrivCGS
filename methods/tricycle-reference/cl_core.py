"""ID-aware Gaussian degree + Gaussian triangle release; CL and TriCycLe postprocessing."""
import math,time
from collections import deque
import numpy as np
from scipy import sparse
import networkx as nx

def rho_from_epsilon(epsilon,delta):
    l=math.log(1/delta)
    return (epsilon/(math.sqrt(l+epsilon)+math.sqrt(l)))**2

def triangle_count(edges,n):
    if len(edges)==0:return 0
    e=np.sort(np.asarray(edges,dtype=np.int64),axis=1)
    u=sparse.csr_matrix((np.ones(len(e),dtype=np.int64),(e[:,0],e[:,1])),shape=(n,n))
    total=0
    for start in range(0,n,128):
        block=u[start:start+128]
        total+=int((block@u).multiply(block).sum())
    return total

def release(degree,triangles,epsilon,seed,delta=1e-5):
    n=len(degree);rho=rho_from_epsilon(epsilon,delta);rd=rt=rho/2
    streams=np.random.SeedSequence([seed,16092026]).spawn(2)
    dr,tr=[np.random.default_rng(x) for x in streams]
    sd=1/math.sqrt(rd);st=max(0,n-2)/math.sqrt(2*rt)
    y=np.asarray(degree,dtype=float)+dr.normal(0,sd,n)
    weights=np.clip(y,0,max(0,n-1))
    raw_t=float(triangles+tr.normal(0,st))
    target=int(round(min(max(raw_t,0),math.comb(n,3) if n>=3 else 0)))
    ledger=dict(epsilon=epsilon,delta=delta,rho_total=rho,rho_degree=rd,rho_triangle=rt,
        degree_L2_sensitivity=math.sqrt(2),triangle_L2_sensitivity=max(0,n-2),
        degree_sigma=sd,triangle_sigma=st,triangle_noisy=raw_t,triangle_target=target,
        degree_policy='ID-aware continuous noisy degree clipped to [0,n-1]; no sort-based identity reassignment',
        accountant='Gaussian zCDP additive composition; epsilon=rho+2*sqrt(rho*log(1/delta))')
    assert abs(rho+2*math.sqrt(rho*math.log(1/delta))-epsilon)<1e-10
    return weights,target,ledger

def generate(weights,target,seed,progress=None,max_seconds=3600,proposals_per_edge=100):
    n=len(weights);start=time.perf_counter()
    graph=nx.expected_degree_graph(weights.tolist(),seed=seed,selfloops=False)
    # NetworkX preserves input node IDs and uses min(w_i*w_j/sum(w),1).
    # expected_degree_graph inserts edges in decreasing-weight rank order,
    # whereas Graph.edges() iterates by public ID. Restore birth order for FIFO.
    order=sorted(range(n),key=lambda u:float(weights[u]),reverse=True)
    rank=np.empty(n,dtype=np.int64);rank[order]=np.arange(n)
    births=sorted(graph.edges(),key=lambda e:(min(rank[e[0]],rank[e[1]]),max(rank[e[0]],rank[e[1]])))
    initial_edges=np.asarray(births,dtype=np.int64).reshape(-1,2)
    initial=triangle_count(initial_edges,n);tri=initial;m=len(initial_edges)
    cl_seconds=time.perf_counter()-start
    neighbours=[list(graph.neighbors(i)) for i in range(n)]
    positions=[{v:k for k,v in enumerate(row)} for row in neighbours]
    oldest=deque(map(tuple,initial_edges.tolist()));del graph
    rng=np.random.default_rng(np.random.SeedSequence([seed,271828]))
    cdf=np.cumsum(weights);mass=float(cdf[-1]) if n else 0
    attempts=accepted=net_positive=rejected=skipped=0
    limit=max(10000,proposals_per_edge*max(m,1));last=time.perf_counter();repair_start=last
    def add(u,v):
        positions[u][v]=len(neighbours[u]);neighbours[u].append(v)
        positions[v][u]=len(neighbours[v]);neighbours[v].append(u)
    def remove(u,v):
        for a,b in ((u,v),(v,u)):
            k=positions[a].pop(b);lastv=neighbours[a].pop()
            if k<len(neighbours[a]):neighbours[a][k]=lastv;positions[a][lastv]=k
    while tri<target and m and mass>0 and attempts<limit:
        if attempts%4096==0 and time.perf_counter()-repair_start>=max_seconds:break
        attempts+=1
        u=min(n-1,int(np.searchsorted(cdf,rng.random()*mass,side='right')))
        if not neighbours[u]:skipped+=1;continue
        k=neighbours[u][int(rng.integers(len(neighbours[u])))]
        v=neighbours[k][int(rng.integers(len(neighbours[k])))]
        if v==u or v in positions[u]:skipped+=1;continue
        a,b=oldest.popleft();lost=len(positions[a].keys()&positions[b].keys());remove(a,b)
        gained=len(positions[u].keys()&positions[v].keys())
        if gained>=lost:
            add(u,v);oldest.append((u,v));tri+=gained-lost;accepted+=1;net_positive+=int(gained>lost)
        else:
            add(a,b);oldest.append((a,b));rejected+=1
        if progress and time.perf_counter()-last>=30:
            progress(dict(attempts=attempts,accepted=accepted,triangles=tri,target=target));last=time.perf_counter()
    edges=np.asarray(sorted((min(a,b),max(a,b)) for a,b in oldest),dtype=np.int64).reshape(-1,2)
    exact=triangle_count(edges,n);assert exact==tri and len(edges)==m
    assert len(np.unique(edges,axis=0))==m and np.all(edges[:,0]<edges[:,1])
    reason='TARGET_REACHED_OR_EXCEEDED' if tri>=target else 'EMPTY_SUPPORT' if not m or mass<=0 else 'RESOURCE_LIMIT'
    return edges,dict(n=n,m=m,initial_triangles=initial,final_triangles=tri,target=target,
        initial_above_target=initial>target,target_reached=tri>=target,target_exact=tri==target,
        stop_reason=reason,proposals=attempts,accepted=accepted,positive_moves=net_positive,rejected=rejected,skipped=skipped,
        cl_seconds=cl_seconds,postprocess_seconds=time.perf_counter()-repair_start,
        total_generation_seconds=time.perf_counter()-start,max_proposals=limit,max_repair_seconds=max_seconds,
        edge_count_preserved=True,triangle_ledger_exact=True,degree_exact_not_promised=True,
        forced_connectivity=False,minimum_degree=0,initial_edge_age_policy='CL sampler insertion order reconstructed from stable decreasing weight rank')
