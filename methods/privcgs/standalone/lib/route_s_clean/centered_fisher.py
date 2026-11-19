"""Stable fallback for invalid raw-moment Fisher matrices, on the same law."""
import ctypes
from pathlib import Path
import numpy as np
from ._native_build import build_shared_library
_LIB=None
EVENTS=[]

def psd_valid(cov):
    if not np.all(np.isfinite(cov)): return False
    diag=np.diag(cov)
    if np.any(diag<0): return False
    zero=diag==0
    if np.any(cov[zero]!=0): return False
    active=~zero
    if not np.any(active): return True
    scale=np.sqrt(diag[active]);corr=cov[np.ix_(active,active)]/scale[:,None]/scale[None,:]
    return bool(np.linalg.eigvalsh(corr)[0]>=-1e-10)

def centered(path,messages,degree,target,sigma,coefficient,channels):
    global _LIB
    if _LIB is None:
        file=Path(__file__).parent/'_native_src/degree_chain/centered_fisher.cpp'
        lib=ctypes.CDLL(str(build_shared_library('centered-fisher-v1',file,compiler_flags=('-O3','-DNDEBUG','-std=c++17','-shared','-static-libgcc','-static-libstdc++'))))
        i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        lib.centered_fisher.argtypes=[ctypes.c_int64,ctypes.c_int64,i,i,f,f,f,f,f,ctypes.c_double,ctypes.c_double,f,f]
        lib.centered_fisher.restype=ctypes.c_int;_LIB=lib
    order=np.asarray(path,dtype=np.int64);k=len(channels);out=np.empty((k+1,k+1));diag=np.empty(3)
    code=_LIB.centered_fisher(len(messages.state_offsets)-1,k,
        np.ascontiguousarray(messages.state_offsets,dtype=np.int64),np.ascontiguousarray(messages.state_value,dtype=np.int64),
        np.ascontiguousarray(messages.forward_log_message),np.ascontiguousarray(messages.structural_gain),
        np.ascontiguousarray(channels,dtype=np.float64),np.ascontiguousarray(np.asarray(degree)[order],dtype=np.float64),
        np.ascontiguousarray(np.asarray(target)[order],dtype=np.float64),1/(2*float(sigma)**2),coefficient,out,diag)
    err=abs(diag[0]-messages.log_normalizer)
    tolerance=256*np.finfo(float).eps*max(1,abs(messages.log_normalizer),float(np.max(np.abs(messages.forward_log_message))))
    event=dict(status=int(code),normalizer_error=float(err),maximum_forward_error=float(diag[1]),tolerance=float(tolerance),long_double_bytes=int(diag[2]),covariance=out.tolist())
    EVENTS.append(event)
    if code or err>tolerance or diag[1]>tolerance or not psd_valid(out):
        raise ArithmeticError('centered Fisher validation failed: '+str(event))
    return out
