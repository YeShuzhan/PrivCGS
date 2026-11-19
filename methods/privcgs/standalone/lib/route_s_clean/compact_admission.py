import ctypes
from pathlib import Path
import numpy as np
from ._native_build import build_shared_library
_LIB=None
def select(n,gain,ol,orr,nl,nr,capacity):
    global _LIB
    if _LIB is None:
        p=Path(__file__).parent/'_native_src/degree_chain/compact_admission.cpp'
        lib=ctypes.CDLL(str(build_shared_library('compact-admission-v1',p,compiler_flags=('-O3','-DNDEBUG','-std=c++17','-shared','-static-libgcc','-static-libstdc++'))))
        i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        lib.compact_admission.argtypes=[ctypes.c_int64,ctypes.c_int64,f,i,i,i,i,ctypes.c_int64,i,i]
        lib.compact_admission.restype=ctypes.c_int;_LIB=lib
    output=np.empty(min(capacity,len(gain)),dtype=np.int64);info=np.zeros(2,dtype=np.int64)
    code=_LIB.compact_admission(n,len(gain),gain,ol,orr,nl,nr,len(output),output,info)
    if code: raise RuntimeError('compact admission status='+str(code))
    return output[:info[0]].copy(),int(info[1])
