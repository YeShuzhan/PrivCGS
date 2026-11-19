"""Exact ordered-bucket realization and matching; same MCR1 graph decisions."""
import ctypes,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
_LIB=None
def library():
    global _LIB
    if _LIB is not None:return _LIB
    from route_s_clean._native_build import build_shared_library
    target=build_shared_library('mcr-ordered',HERE/'mcr_ordered.cpp',compiler_flags=(
        '-O3','-DNDEBUG','-std=c++17','-shared','-static-libgcc','-static-libstdc++'))
    lib=ctypes.CDLL(str(target));a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.partial_section.argtypes=[ctypes.c_int,a,ctypes.c_int64,a,ctypes.c_int64,a,ctypes.c_int64,a,a,a,a,a]
    lib.partial_section.restype=ctypes.c_int64
    lib.matching_tail.argtypes=[ctypes.c_int,a,ctypes.c_int64,a,a];lib.matching_tail.restype=ctypes.c_int64
    _LIB=lib;return lib
def array(x):return np.ascontiguousarray(x,dtype=np.int64)
def partial_virtual_section(target,source_edges,removals,additions):
    d=array(target);source=array(source_edges);rem=array(removals);add=array(additions);n=len(d)
    out=np.empty((int(d.sum())//2,2),dtype=np.int64);residual=np.empty(n,dtype=np.int64);closed=np.zeros(n,dtype=np.int64);laid=np.zeros(1,dtype=np.int64)
    count=library().partial_section(n,d,len(source),source,len(rem),rem,len(add),add,out,residual,closed,laid)
    if count<0:raise RuntimeError('native partial degree realization failed')
    active=set(map(int,np.flatnonzero(residual==1)))
    return out[:count].copy(),residual,active,closed.astype(bool),dict(laid_off_before_matching_boundary=int(laid[0]),partial_edge_count=count,matching_tail_edge_count=len(active)//2,maximum_residual_degree=int(residual.max(initial=0)))
def finish_virtual_matching(active,virtual):
    if not active:return np.empty((0,2),dtype=np.int64)
    n=1+max(active)
    flags=np.zeros(n,dtype=np.int64);flags[list(active)]=1
    # Original matching ignores edges outside the active set; filter before copy.
    edges=array([e for e in virtual if e[0] in active and e[1] in active]).reshape(-1,2)
    out=np.empty((len(active)//2,2),dtype=np.int64)
    count=library().matching_tail(n,flags,len(edges),edges,out)
    if count<0:raise RuntimeError('native matching failed')
    return out[:count]
