"""Equivalent native W4/JR; unchanged Python reports and exact-risk certificates."""
from pathlib import Path
import ctypes as C, time, os
import numpy as np
HERE=Path(__file__).resolve().parent
_LIB=None
_DLL_DIRECTORY=None
_CURRENT=None
_TRACE=None
def native_path():return Path(_LIB._name) if _LIB is not None else HERE/'repair_native_final.so'
_I=C.c_longlong
_IP=C.POINTER(_I)
_PROGRESS=C.CFUNCTYPE(None,_I,_I,_I,_IP)
class Stats(C.Structure):
    _fields_=[(v,_I) for v in ['attempted','accepted','gain','open_wedges','scanned','nonpositive','overshoot']]+[('kinds',_I*5)]+[(v,_I) for v in ['valid','reject_hard','reject_risk','cross','current']]+[('seconds',C.c_double)]
def library():
    global _LIB,_DLL_DIRECTORY
    if _LIB:return _LIB
    if hasattr(os,'add_dll_directory'):_DLL_DIRECTORY=os.add_dll_directory(str(HERE))
    lib=C.CDLL(str(native_path()))
    lib.eq_error.restype=C.c_char_p
    lib.eq_create.argtypes=[_IP,_I,_I,C.c_int];lib.eq_create.restype=C.c_void_p
    lib.eq_free.argtypes=[C.c_void_p]
    lib.eq_bind.argtypes=[C.c_void_p,_IP,_I];lib.eq_bind.restype=C.c_int
    lib.eq_triangles.argtypes=[C.c_void_p];lib.eq_triangles.restype=_I
    lib.eq_common.argtypes=[C.c_void_p,_I,_I];lib.eq_common.restype=_I
    lib.eq_sweep.argtypes=[C.c_void_p,C.c_int,_I,_I,_I,_I,C.c_int,_I,_I,_I,C.c_char_p,C.c_char_p,C.c_char_p,C.POINTER(Stats),_PROGRESS,C.c_char_p];lib.eq_sweep.restype=C.c_int
    lib.eq_ledger_count.argtypes=[C.c_void_p];lib.eq_ledger_count.restype=_I
    lib.eq_ledger_data.argtypes=[C.c_void_p];lib.eq_ledger_data.restype=_IP
    _LIB=lib;return lib
def checked(v):
    if v<0:raise RuntimeError(library().eq_error().decode())
    return v
class Graph:
    def __init__(self,edges,n):
        self.edges=np.asarray(edges);assert self.edges.dtype==np.int64 and self.edges.flags.c_contiguous
        self.n=int(n);self.lib=library();self.handle=self.lib.eq_create(self.edges.ctypes.data_as(_IP),n,len(edges),int(os.environ.get('OMP_NUM_THREADS','8')))
        if not self.handle:raise RuntimeError(self.lib.eq_error().decode())
    def bind(self,edges):
        edges=np.asarray(edges);assert edges.dtype==np.int64 and edges.flags.c_contiguous
        if edges.ctypes.data!=self.edges.ctypes.data:checked(self.lib.eq_bind(self.handle,edges.ctypes.data_as(_IP),len(edges)));self.edges=edges
    def __del__(self):
        if getattr(self,'handle',None):self.lib.eq_free(self.handle);self.handle=None
def adjacency_from_edges(edges,n):
    global _CURRENT
    if _CURRENT is not None and _CURRENT.n==n and len(_CURRENT.edges)==len(edges):
        _CURRENT.bind(edges);return _CURRENT
    _CURRENT=Graph(edges,n);return _CURRENT
def count_triangles(edges,adjacency):
    adjacency.bind(edges);return checked(adjacency.lib.eq_triangles(adjacency.handle))
def common_count(adjacency,a,b):return checked(adjacency.lib.eq_common(adjacency.handle,int(a),int(b)))
def set_trace(path):
    global _TRACE
    _TRACE=Path(path) if path else None
    if _TRACE:_TRACE.parent.mkdir(parents=True,exist_ok=True)
def reset_graph():
    global _CURRENT
    _CURRENT=None
def sweep(direction,dataset,edges,adjacency,target,current,lo,hi,sweep_index,risk_policy=None):
    import w4_repair as repair
    began=time.perf_counter();adjacency.bind(edges);stats=Stats();callback_error=[]
    if risk_policy is not None and risk_policy.audit_callback is not None:
        raise RuntimeError('Audit callbacks require explicit trace replay adapter; production callback is None')
    @_PROGRESS
    def progress(position,accepted,tri,kinds):
        try:
            repair.write_json(repair.OUT/'progress.json',dict(dataset=dataset,phase='low_degree_sweep',sweep=sweep_index,pivots_completed=position,accepted=accepted,triangles=tri,accepted_degree_pairs={k:int(kinds[i]) for i,k in enumerate(['1_1','1_2','2_2','one_low','other'])}))
        except BaseException as e:callback_error.append(e)
    p=risk_policy
    bl,bh,star=(p.band_lo,p.band_hi,p.target) if p else (lo,hi,0)
    slopes=(str(p.a).encode(),str(p.b).encode(),str(p.den).encode()) if p else (b'1',b'1',b'1')
    trace=os.fsencode(str(_TRACE)) if _TRACE else b''
    checked(adjacency.lib.eq_sweep(adjacency.handle,direction,current,lo,hi,sweep_index,int(p is not None),bl,bh,star,*slopes,C.byref(stats),progress,trace))
    if callback_error:raise callback_error[0]
    if p:
        p.candidate_scans+=stats.attempted;p.valid_candidates+=stats.valid;p.accepted_moves+=stats.accepted
        p.rejected_hard_interval+=stats.reject_hard;p.rejected_risk+=stats.reject_risk;p.cross_center_moves+=stats.cross
        if stats.accepted:p.last_triangle=int(stats.current)
    assert np.array_equal(np.bincount(edges.ravel(),minlength=len(target)),target)
    if direction>0:
        kinds={k:int(stats.kinds[i]) for i,k in enumerate(['1_1','1_2','2_2','one_low','other'])}
        log=dict(sweep=sweep_index,open_wedges=int(stats.open_wedges),candidate_pairs_scanned=int(stats.scanned),attempted_donor_pairs=int(stats.attempted),accepted=int(stats.accepted),triangle_gain=int(stats.gain),accepted_degree_pairs=kinds,guaranteed_new_size2_components=kinds['1_1'],rejected_nonpositive=int(stats.nonpositive),rejected_overshoot=int(stats.overshoot),interval_hit=bool(lo<=stats.current<=hi))
    else:
        n=adjacency.lib.eq_ledger_count(adjacency.handle);buf=adjacency.lib.eq_ledger_data(adjacency.handle)
        ledger=np.ctypeslib.as_array(buf,shape=(n*9,)).reshape(n,9) if n else np.empty((0,9),np.int64)
        repair.OUT.mkdir(parents=True,exist_ok=True);np.save(repair.OUT/f'decrease_ledger_{sweep_index:03d}.npy',ledger)
        log=dict(direction='decrease',sweep=sweep_index,accepted=int(stats.accepted),triangle_gain=int(stats.gain),attempted_donor_pairs=int(stats.attempted),rejected_nonnegative=int(stats.nonpositive),rejected_overshoot=int(stats.overshoot),interval_hit=bool(lo<=stats.current<=hi))
    log['sweep_seconds']=time.perf_counter()-began
    return int(stats.current),log
def increase_sweep(*a,**k):return sweep(1,*a,**k)
def decrease_sweep(*a,**k):return sweep(-1,*a,**k)
def install():
    import w4_repair as repair
    for name in ['adjacency_from_edges','count_triangles','common_count','increase_sweep','decrease_sweep']:setattr(repair,name,globals()[name])
