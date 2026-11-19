import numpy as np
def assignment(multiset,degree,recovered):
 carrier_degree=np.asarray(degree,dtype=np.float64)
 residual=np.asarray(recovered.endpoint_target,dtype=float)-carrier_degree
 centered=residual-residual.mean();observed=float(np.var(centered,ddof=1 if len(centered)>1 else 0))
 noise=float(recovered.graph_law.degree_noise_std)**2;latent=max(observed-noise,0.0);shrinkage=latent/(latent+noise) if latent else 0.0
 import structural_order as source
 rank_order=source.structural_order(carrier_degree.astype(np.int64),recovered.graph_law,source.source_rank_direct(recovered.graph_law))
 rank=np.empty(len(multiset),dtype=np.int64);rank[rank_order]=np.arange(len(multiset),dtype=np.int64)
 nodes=np.arange(len(multiset),dtype=np.int64);order=np.lexsort((nodes,rank,carrier_degree+shrinkage*residual))
 target=np.empty_like(multiset);target[order]=multiset
 return np.ascontiguousarray(target),dict(posterior_shrinkage=shrinkage,degree_noise_variance=noise,observed_residual_variance=observed,carrier_degree_from_flow_state=True,carrier_graph_materialized=False)

