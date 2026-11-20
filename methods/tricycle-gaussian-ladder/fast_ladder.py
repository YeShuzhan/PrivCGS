"""Exact accelerated Ladder profile, with the unchanged Gaussian selector."""
import ctypes
import hashlib
from pathlib import Path
import numpy as np

from gaussian_ladder import _Ladder


def profile(edges, n):
    if n < 3:
        return _Ladder(n, [])
    edges = np.ascontiguousarray(edges, dtype=np.int32)
    best = np.full(n-1, -1, dtype=np.int32)
    evidence = np.zeros(3, dtype=np.int64)
    method = ctypes.CDLL(str(Path(__file__).with_name('ladder_native.so'))).profile
    a32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    a64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    method.argtypes = [ctypes.c_int32, ctypes.c_int64, a32, a32, a64]
    method.restype = ctypes.c_int
    status = method(n, len(edges), edges, best, evidence)
    if status:
        raise ValueError(('canonical edge or profile failure', status))
    cap = n-2
    useful = np.flatnonzero(best >= 0)
    suffix = np.maximum.accumulate(best[::-1])[::-1]
    lower = -1
    prefix = [0]
    for t in range(2*cap+1):
        if 0 < t <= len(best) and best[t-1] >= 0:
            lower = max(lower, 2*int(best[t-1])+t-1)
        upper = int(suffix[t]) if t < len(best) else -1
        width = min(cap,max(t+upper if upper>=0 else 0,(t+lower)//2 if lower>=0 else 0))
        if width == cap:
            break
        prefix.append(prefix[-1]+width)
    else:
        raise RuntimeError('ladder did not reach its public sensitivity cap')
    return _Ladder.__new__(_Ladder), prefix, evidence, len(useful)


def build(edges, n):
    ladder,prefix,evidence,buckets = profile(edges,n)
    ladder.cap=n-2
    ladder.triangles=int(evidence[0])
    ladder.prefix=prefix
    return ladder, {'intersection_count':int(evidence[1]),'retained_witnesses':int(evidence[2]),'b_buckets':buckets}


def from_record(data):
    ladder=_Ladder.__new__(_Ladder)
    ladder.cap=data['n']-2
    ladder.triangles=data['triangles']
    ladder.prefix=data['prefix']
    return ladder


def proof():
    # L_ij(t) before clipping = min(a+t, floor((c+t)/2)),
    # where c=du+dv-2*A_uv=2*a+b. Increasing either a or c
    # cannot decrease any rung. A single witness with both at least as
    # large covers a skipped pair for every t; using d_v and du+dv
    # gives valid upper bounds before the exact intersection.
    return 'coordinatewise domination in (a,2a+b)'
