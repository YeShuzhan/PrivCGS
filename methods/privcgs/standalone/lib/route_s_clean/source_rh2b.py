"""RH2B-LS2 pair salvage, preserving the primary carrier and score."""
from pathlib import Path
import ctypes as ct
import hashlib,json,os,time
import numpy as np
from ._native_build import build_shared_library,native_source

_LIB=None
def load_rh2b_library():
    global _LIB
    if _LIB is None:
        src=native_source('source_order','rh2b.cpp')
        deps=sorted(src.parent.glob('*.hpp'))+sorted(src.parent.glob('*.inc'))
        dll=build_shared_library('rh2b_ls2_v1',src,dependencies=deps,compiler_flags=['-O3','-std=c++17','-shared','-static-libgcc','-static-libstdc++','-ffp-contract=off'])
        _LIB=ct.CDLL(str(dll));f=_LIB.rh2b_generate
        ptr=ct.c_void_p
        f.argtypes=[ct.c_int]*4+[ct.c_int64]+[ct.c_double]*2+[ptr]*5+[ct.c_double]+[ptr]*5+[ct.c_uint64]*2+[ptr]*5+[ct.c_char_p]*2
        f.restype=ct.c_int
        _LIB.rh2b_error.restype=ct.c_char_p
        _LIB.rh2b_diagnostic_io_seconds.restype=ct.c_double
    return _LIB

def build_source_order_tree(law):
    return None,dict(method='RH2B-LS2',tree_built=False,frontier_records=0,n_nodes=int(law.n_nodes),rank=int(law.rank))

def _reference_checks(law,masses,trace):
    """Independent dense rows only for the two real small graphs; no synthetic graphs."""
    n=int(law.n_nodes);s=np.asarray(law.sociability);a=np.asarray(law.star);x=np.asarray(law.interaction);w=np.asarray(law.interaction_diagonal)
    target=np.asarray(law.target_endpoint_mass);var=float(law.degree_noise_std)**2
    rows=np.unique(np.r_[np.arange(min(8,n)),np.linspace(0,n-1,min(24,n),dtype=int)])
    errors=[]
    for u in rows:
        carrier=s[u]*s*(np.log(2)+1e-6+(np.abs(x)*np.abs(x[u])*np.abs(w)).sum(axis=1))+s[u]*a+s*a[u]+law.dust
        carrier[u]=0
        expected=carrier.sum();errors.append(abs(masses[u]-expected)/max(expected,1e-300))
        np.testing.assert_allclose(masses[u],expected,rtol=2e-12,atol=1e-12)
    d=np.zeros(n,dtype=np.int64);seen=set();norm_errors=[];score_errors=[];occupied=np.zeros(n)
    for t in trace:
        u,v1,v2,v=map(int,t[:4]);assert t[6]==d[u] and t[7]==d[v1] and t[8]==d[v2]
        branch=int(t[12])
        with np.errstate(divide='ignore'):
            logW=np.log(masses)+(target-d)/var
        logW[d==n-1]=-np.inf
        mx=logW.max();logZ=mx+np.log(np.exp(logW-mx).sum())
        if branch!=2:
            c=np.divide(occupied,masses,out=np.zeros(n),where=masses>0)
            phi=float(np.exp(logW-logZ)@(c*c))
            np.testing.assert_allclose([t[13],t[16]],[c[u],phi],rtol=1e-9,atol=1e-12)
        if branch==1:
            assert all(tuple(sorted((u,int(x)))) in seen for x in t[14:16])
        if branch==2:
            assert tuple(sorted((u,v))) not in seen
            seen.add(tuple(sorted((u,v))));d[u]+=1;d[v]+=1
            continue
        np.testing.assert_allclose([t[9],t[10]],[logW[u],logZ],rtol=2e-12,atol=2e-11)
        norm_errors.append(abs(t[10]-logZ));scores=[]
        for j,partner in enumerate((v1,v2)):
            assert partner!=u
            if tuple(sorted((u,partner))) in seen:
                assert np.isneginf(t[4+j]);scores.append(-np.inf);continue
            b=np.sum(w*x[u]*x[partner]);intensity=s[u]*s[partner]*(np.logaddexp(0,b)+1e-6)+s[u]*a[partner]+s[partner]*a[u]+law.dust
            k=1 if law.component_fiber[u]==law.component_fiber[partner] and np.dot(x[u],x[partner])>=0 else (0 if a[u]+a[partner]>=2 else (3 if s[u]+s[partner]<=2 else 2))
            p,q=int(law.parent_fiber[u]),int(law.parent_fiber[partner])
            eta=np.log(max(np.expm1(np.clip(intensity,1e-12,20)),1e-12))+law.marked_shift[p]+law.marked_shift[q]+law.parent_flux[p,q]+law.component_shift[k]
            score=eta/law.temperature+(target[partner]-d[partner])/var
            np.testing.assert_allclose(t[4+j],score,rtol=2e-12,atol=2e-10)
            score_errors.append(abs(t[4+j]-score));scores.append(score)
        winner=v1 if t[4]>t[5] or (t[4]==t[5] and v1<v2) else v2
        assert v==winner and tuple(sorted((u,v))) not in seen
        massuv=s[u]*s[v]*(np.log(2)+1e-6+np.sum(np.abs(w)*np.abs(x[u])*np.abs(x[v])))+s[u]*a[v]+s[v]*a[u]+law.dust
        occupied[u]+=massuv;occupied[v]+=massuv
        seen.add(tuple(sorted((u,v))));d[u]+=1;d[v]+=1
    return dict(status='PASS',real_nodes=n,dense_reference_rows=len(rows),births_checked=len(trace),max_mass_relative_error=max(errors,default=0),max_log_normalization_error=max(norm_errors,default=0),max_score_error=max(score_errors,default=0))

def generate_source_edges(law,tree,*,order_seed,mark_seed):
    began=time.perf_counter();n=int(law.n_nodes);m=int(law.target_edges)
    lib=load_rh2b_library();outdir=Path(os.environ['SOURCE_TRACE_LOG']).parent
    arrays=[np.ascontiguousarray(getattr(law,name),dtype=dtype) for name,dtype in [
        ('target_endpoint_mass',np.float64),('sociability',np.float64),('star',np.float64),('interaction',np.float64),('interaction_diagonal',np.float64),
        ('parent_fiber',np.int32),('component_fiber',np.int32),('marked_shift',np.float64),('component_shift',np.float64),('parent_flux',np.float64)]]
    edges=np.empty((m,2),dtype=np.int64);di=np.zeros(20,dtype=np.int64);re=np.zeros(14);mass=np.empty(n);trace=np.full((min(m,128),18),np.nan)
    p=lambda a:ct.c_void_p(a.ctypes.data)
    code=lib.rh2b_generate(n,int(law.rank),len(law.marked_shift),len(law.component_shift),m,float(law.degree_noise_std),float(law.temperature),
        *[p(a) for a in arrays[:5]],float(law.dust),*[p(a) for a in arrays[5:]],int(order_seed),int(mark_seed),
        p(edges),p(di),p(re),p(mass),p(trace),str(outdir/'RH2B_progress.jsonl').encode(),str(outdir/'native_stop.request').encode())
    count=int(di[0]);trace=trace[:min(count,128)]
    diag=dict(method='RH2B-LS2-v1',n_nodes=n,changes_S_distribution=True,admitted=count,target_edges=m,two_choice_rounds=int(di[1]),partner_draws=2*int(di[1]),invalid_candidates=int(di[2]),eta_calls=int(di[3]),same_partner_pair_count=int(di[4]),max_rounds_one_birth=int(di[5]),overshoot_node_count=int(di[6]),edge_hash_bytes=int(di[7]),carrier_CDF_bytes=int(di[8]),source_tree_bytes=int(di[9]),native_algorithm_seconds=float(re[0]),carrier_setup_seconds=float(re[1]),terminal_degree_target_l1=float(re[2]),native_diagnostic_io_seconds=float(re[3]),frontier_records=0,full_dyad_array_created=False,new_private_queries=0,status='COMPLETE' if code==0 else 'FAILED',error=lib.rh2b_error().decode() if code else None)
    diag.update(rescue_edge_count=int(di[10]),uniform_missing_completion_count=int(di[11]),carrier_exhausted_rescue_count=int(di[12]),saturated_source_count=int(di[13]),legal_index_bytes=int(di[14]),legal_index_activations=int(di[15]),legal_rank_queries=int(di[16]),uniform_integer_bit_rejections=int(di[17]),normal_births=int(di[18]),telemetry_range_roundoff_count=int(di[19]),mean_selected_source_c=float(re[4]),mean_selected_source_c_squared=float(re[5]),mean_global_rescue_pressure=float(re[6]),max_selected_source_c=float(re[7]),selected_c_quantile_intervals={str(q):[float(re[8+i]),min(1.,float(re[8+i])+1/65536)] for i,q in enumerate((.95,.99,.999))},realized_rescue_fraction=float(re[11]),pressure_state_bytes=int(re[12]),telemetry_max_range_roundoff=float(re[13]),q0_retry_count=0,rescue_partner_draws=2*int(di[10]),global_completion_partner_draws=int(di[11]),pressure_normalization='Mean Phi over normal births; global-zero completion recorded separately',quantile_method='65536-bin rank interval; not falsely reported as exact quantiles')
    if not code:
        assert di[1]==di[18] and di[18]+di[11]==m and di[3]<=2*m
    from execution_accounting import charge
    with charge('diagnostic_io'):
        np.save(outdir/'RH2B_first_births.npy',trace)
        if n<=5000: np.save(outdir/'S_birth_order.npy',edges[:count])
        if code:np.save(outdir/'failed_S_prefix.npy',edges[:count])
        (outdir/'S_RH2B.json').write_text(json.dumps(diag,indent=2,allow_nan=False))
        (outdir/'S_RH2B_LS2.json').write_text(json.dumps(diag,indent=2,allow_nan=False))
    if code:raise RuntimeError(diag['error'])
    if n<=5000:
        with charge('mechanism_validation'):
            check=_reference_checks(law,mass,trace)
            (outdir/'RH2B_reference_checks.json').write_text(json.dumps(check,indent=2))
    # Canonical S is the unchanged downstream interface. Unique edges were enforced natively.
    order=np.lexsort((edges[:,1],edges[:,0]));edges=np.ascontiguousarray(edges[order]);del order
    assert len(edges)==m and np.all(edges[:,0]<edges[:,1]) and (not m or (edges.min()>=0 and edges.max()<n))
    assert m<2 or np.all(np.any(edges[1:]!=edges[:-1],axis=1))
    diag.update(edge_sha256=hashlib.sha256(edges.tobytes()).hexdigest(),source_order_seconds=time.perf_counter()-began,exact_stop_certificate=True)
    with charge('diagnostic_io'):
        # Freeze complete S before entering unchanged Q; a later Q/W4 failure must not lose S.
        np.save(outdir/'S_edges.npy',edges)
        (outdir/'S_RH2B.json').write_text(json.dumps(diag,indent=2,allow_nan=False))
        (outdir/'S_RH2B_LS2.json').write_text(json.dumps(diag,indent=2,allow_nan=False))
    return edges,diag
