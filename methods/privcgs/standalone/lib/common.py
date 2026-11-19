from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def canonical(a):
    a=np.sort(np.asarray(a,dtype=np.int64).reshape(-1,2),axis=1)
    return np.ascontiguousarray(a[np.lexsort((a[:,1],a[:,0]))])


def numeric_release_hash(release):
    c=release.checkpoint;t=release.transcript
    h=hashlib.sha256()
    for key in ('basis','continuous_noisy_degree','raw_noisy_operator_vector'):
        a=np.ascontiguousarray(getattr(c,key),dtype='<f8')
        h.update(key.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    scalar={k:float(getattr(c,k)) for k in ('continuous_noisy_edge_mass','mass_coordinate_weight',
        'degree_sigma','operator_noise_std','rho_total','rho_degree','rho_structural')}
    scalar.update({k:getattr(t,k) for k in ('noisy_closure_ratio','closure_noise_std',
        'projection_degree_cap','denominator_stabilizer','closure_l2_sensitivity',
        'retained_wedge_proxy_fraction','rho_total','rho_degree','rho_operator','rho_closure','public_projection_seed')})
    for k in ('noisy_projected_wedge_count','noisy_projected_triangle_incidence','closure_count_noise_std'):
        scalar[k]=float(release.diagnostics[k])
    h.update(json.dumps(scalar,sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()


def sanitized_release(release):
    from dataclasses import replace
    # All retained diagnostics are functions of released arrays, public parameters or timings.
    prohibited={'projected_edge_count_evaluation_for_release','discarded_edge_count_evaluation_for_release'}
    diag={k:v for k,v in release.diagnostics.items() if k not in prohibited}
    td={k:v for k,v in release.transcript.diagnostics.items() if k not in prohibited}
    return replace(release,diagnostics=diag,transcript=replace(release.transcript,diagnostics=td))
