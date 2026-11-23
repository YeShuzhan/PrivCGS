"""Independent MCR1 -> exact completion -> bidirectional W4 -> Cstar."""
from pathlib import Path
import argparse,hashlib,json,os,pickle,sys,time
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'modules'),str(ROOT/'standalone/lib')]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]=os.environ.get('PRIVCGS_THREADS','8')
import numpy as np
import psutil
from execution import Observer, configure, external_imports, sha, write
import execution
execution.ROOT=ROOT
from execution_accounting import diagnostic_writes
from reference import original_spectrum
from closure_base import original_wedge_interval
STUDY=json.loads(Path(os.environ['PRIVCGS_STUDY_CONFIG']).read_text())
VARIANT='PrivCGS'
from tmmr import degree_target as new_target, MethodBoundary
ACTIVE_OBSERVER=None

def diagnostic_json(v):
    if isinstance(v,dict):return {k:diagnostic_json(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [diagnostic_json(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
    return v

def main():
    global ACTIVE_OBSERVER
    ap=argparse.ArgumentParser()
    ap.add_argument('--release-manifest',required=True,type=Path)
    ap.add_argument('--dataset',required=True);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--cpu',default='');ap.add_argument('--verify-imports',action='store_true')
    ap.add_argument('--compiler',help='Compiler executable, alternatively set CXX')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    configure(args.cpu,args.compiler)
    forbidden=[k for k in os.environ if (k.startswith('S_') and k.endswith('_SHADOW')) or k in ('S_PREFIX_CANDIDATES','SOURCE_TRACE_LIMIT_VISITS','S_ACCEPT_RANKS','S_STATE_DUMP')]
    if forbidden:raise RuntimeError('Remove research-only environment variables: '+str(forbidden))
    os.environ.update(S_QUEUE_DIR=str(out/'scratch'),S_QUEUE_BUFFER_RECORDS=str(1<<20),S_QUEUE_READ_RECORDS='2048',SOURCE_TRACE_LOG=str(out/'S_progress.jsonl'),S_QUEUE_STATS=str(out/'S_queue.json'),S_SUCCESSOR_STATS=str(out/'S_competition.json'),S_CERT_STATS=str(out/'S_certificates.json'))
    from runtime import installed,recover_checkpoint_in_memory,capture_source_and_flow
    from degree_target import selected_deltas
    from mcr_fast import partial_virtual_section,finish_virtual_matching
    from closure_interval import closure_interval
    from common import canonical,sha_array
    from closure_center import validate_graph,minimax_target
    from joint_risk import refine
    import w4_repair as repair
    input_started=time.perf_counter()
    manifest_path=args.release_manifest.resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    checkpoint_path=Path(manifest['path'])
    if not checkpoint_path.is_absolute():checkpoint_path=manifest_path.parent/checkpoint_path
    assert sha(checkpoint_path)==manifest['sha256']
    with open(checkpoint_path,'rb') as f:release=pickle.load(f)
    input_seconds=time.perf_counter()-input_started
    observer=ACTIVE_OBSERVER=Observer(out)
    timing={};stage=None;stage_excluded={};stage_start_excluded={}
    def mark(name):
        nonlocal stage,stage_start_excluded
        if stage:
            timing[stage]=time.perf_counter()-mark.started
            end=observer.excluded();stage_excluded[stage]={k:end[k]-stage_start_excluded[k] for k in end}
        stage=name;mark.started=time.perf_counter()
        stage_start_excluded=observer.excluded()
        observer.current=name;observer.stage_started=mark.started
    setup_started=time.perf_counter()
    with installed(out/'numerical',int(os.environ['PRIVCGS_THREADS'])),diagnostic_writes(out),observer.pipeline_stages():
        setup_seconds=time.perf_counter()-setup_started
        excluded_before=observer.excluded()
        algorithm_start=time.perf_counter()
        mark('recovery');rec=recover_checkpoint_in_memory(release.checkpoint,args.seed)
        mark('spectrum');spectrum,logp0=original_spectrum(release.checkpoint,skip_integerization=True)
        mark('S_Q');carrier,tape,flow,details=capture_source_and_flow(rec,args.seed)
        mark('target');s=carrier.edge_array;n=len(rec.endpoint_target)
        rem,add=selected_deltas(tape,flow)
        keys=np.union1d(np.setdiff1d(s[:,0]*n+s[:,1],rem[:,0]*n+rem[:,1],assume_unique=True),add[:,0]*n+add[:,1])
        qdeg=np.bincount(np.r_[keys//n,keys%n],minlength=n)
        degree_wedge_state=original_wedge_interval(release.checkpoint.continuous_noisy_degree,float(release.checkpoint.degree_sigma),STUDY['alpha']/3)
        np.save(out/'S_edges.npy',s);np.save(out/'Q_keys.npy',keys);np.save(out/'p0.npy',spectrum.probability);np.save(out/'logp0.npy',logp0)
        write(out/'degree_W_state.json',diagnostic_json(dict(gL=degree_wedge_state[0],gU=degree_wedge_state[1],inversion=degree_wedge_state[2],alpha=STUDY['alpha']/3,release_sha256=manifest['sha256'])))
        target,tr=new_target(qdeg,len(s),release,rec,spectrum,log_probability=logp0,degree_wedge_state=degree_wedge_state,seed=args.seed,output=out,progress=lambda v:write(out/'target_progress.json',v))
        np.save(out/'target_degree.npy',target)
        mark('partial_HH_tail')
        prefix,residual,active,closed,boundary=partial_virtual_section(target,s,rem,add)
        tail=finish_virtual_matching(active,set(zip(map(int,keys//n),map(int,keys%n))))
        edges=canonical(np.concatenate((prefix,np.asarray(tail,dtype=np.int64).reshape(-1,2))))
        np.save(out/'H0_edges.npy',edges)
        H0_sha256=sha_array(edges)
        validate_graph(edges,target)
        mark('closure_interval');W=int(np.sum(target*(target-1)//2));lo,hi,contract=closure_interval(release,W,alpha=STUDY['alpha'],degree_wedge_state=degree_wedge_state)
        mark('W4');repair.OUT=out
        adjacency=repair.adjacency_from_edges(edges,n);tri=repair.count_triangles(edges,adjacency);initial=tri;sweeps=[]
        stage_lo,stage_hi=lo,hi
        while not stage_lo<=tri<=stage_hi:
            tri,log=repair.degree_only_sweep(args.dataset,edges,adjacency,target,tri,stage_lo,stage_hi,len(sweeps));sweeps.append(log)
            assert repair.count_triangles(edges,adjacency)==tri
            validate_graph(edges,target)
            if not log['accepted']:break
        # Preserve the old W4 save/load ordering, without disk round-tripping.
        edges=repair.canonical(edges);w4=edges.copy();w4_tri=tri
        mark('JR_RB2');selection=None;center_sweeps=[]
        if lo<=tri<=hi:
            validate_graph(edges,target)
            selection=minimax_target(*contract['original_C_interval'],W,lo,hi)
            adjacency=repair.adjacency_from_edges(edges,n)
            assert repair.count_triangles(edges,adjacency)==tri
        tri,rb2,center_sweeps=refine(args.dataset,edges,adjacency,target,tri,W,lo,hi,contract,repair)
        assert repair.count_triangles(edges,adjacency)==tri
        mark('final_canonicalization');edges=repair.canonical(edges);validate_graph(edges,target)
        timing[stage]=time.perf_counter()-mark.started
        excluded_after=observer.excluded()
        stage_excluded[stage]={k:excluded_after[k]-stage_start_excluded[k] for k in excluded_after}
        excluded_seconds={k:excluded_after[k]-excluded_before[k] for k in excluded_after}
        algorithm_wall_seconds=time.perf_counter()-algorithm_start
        algorithm_seconds=algorithm_wall_seconds-sum(excluded_seconds.values())
        algorithm_peak=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024
    # Final serialization and validation reports are outside algorithm time.
    observer.current='final_save';observer.stage_started=time.perf_counter()
    save_started=time.perf_counter()
    np.save(out/'S_edges.npy',s);np.save(out/'Q_keys.npy',keys);np.save(out/'target_degree.npy',target)
    np.save(out/'H_edges.npy',w4);np.save(out/'G_edges.npy',edges)
    np.savetxt(out/'edges.edgelist',edges,fmt='%d')
    domain=dict(n_nodes=n,internal_ids='0..n-1',isolated_nodes_included=True,ordering='Frozen checkpoint order',release_manifest=str(manifest_path))
    if manifest.get('public_node_ids_path'):
        import shutil
        mapping=Path(manifest['public_node_ids_path'])
        if not mapping.is_absolute():mapping=manifest_path.parent/mapping
        assert sha(mapping)==manifest['public_node_ids_sha256']
        shutil.copyfile(mapping,out/'public_node_ids.npy')
        domain.update(public_node_ids_path='public_node_ids.npy',public_node_ids_sha256=manifest['public_node_ids_sha256'])
    write(out/'node_domain.json',domain)
    final_save_seconds=time.perf_counter()-save_started
    loaded={k:str(v.__file__) for k,v in list(sys.modules.items()) if getattr(v,'__file__',None)}
    leaks=external_imports()
    report=dict(status='GENERATED_AWAITING_EVALUATION',version=VARIANT,q_tolerance_policy={'fallback_at_iterations':270400,'fallback_max_log_marginal_error':1e-6,'successful_path_unchanged':True,'classification':'user-authorized numerical tolerance adjustment'},dataset=args.dataset,seed=args.seed,n=n,m=len(edges),
        release_sha256=manifest['sha256'],S_sha256=sha_array(s),target_sha256=sha_array(target),graph_sha256=sha(out/'edges.edgelist'),
        W=W,initial_triangles=initial,W4_triangles=w4_tri,final_triangles=tri,closure_interval=[int(lo),int(hi)],closure_contract=contract,
        closure_hit=bool(lo<=tri<=hi),Cstar_selection=selection,target_report=tr,Cstar_RB2=rb2,JR_RB2=rb2,
        closure_interval_hit=rb2['closure_interval_hit'],cstar_risk_band_hit=rb2['cstar_risk_band_hit'],cstar_exact_hit=rb2['cstar_exact_hit'],
        timings_wall=timing,stage_excluded_seconds=stage_excluded,timings={k:timing[k]-sum(stage_excluded[k].values()) for k in timing},nested_S_Q_stages=observer.stages,
        input_seconds=input_seconds,setup_seconds=setup_seconds,algorithm_wall_seconds=algorithm_wall_seconds,excluded_seconds=excluded_seconds,algorithm_seconds=algorithm_seconds,final_save_seconds=final_save_seconds,peak_RSS_MiB=algorithm_peak,
        timing_scope='Checkpoint-to-final-canonical-G; P1 measured separately. Excludes input, native setup/cache, measured diagnostic I/O, final save and evaluation. Required queue I/O included. Nested stages must not be added to parent stages.',
        peak_RSS_scope='Process peak through algorithm end, including input/setup',resource_monitor_thread_CPU_seconds=observer.monitor_cpu_seconds,
        variant=VARIANT,degree_W_state_sha256=sha(out/'degree_W_state.json'),new_private_queries=0,original_graph_accessed=False,external_project_imports=leaks,W4_sweeps=sweeps,Cstar_sweeps=center_sweeps)
    report.update(study_config=STUDY,coverage_alpha=STUDY['alpha'],calibration_zeta=STUDY['zeta'],
        stage_array_sha256={'H0':H0_sha256,'H':sha_array(w4),'G':sha_array(edges)},
        H0_H_G_same_degree_sequence=True,upstream_reused=False,
        closure_candidate_interval=[int(stage_lo),int(stage_hi)])
    write(out/'generation_report.json',report);write(out/'imports.json',loaded)
    write(out/'MCR1_stage_diagnostics.json',diagnostic_json(details.diagnostics))
    if args.verify_imports:assert not leaks,leaks
    observer.close('COMPLETE')
    print(json.dumps({k:report[k] for k in ('status','n','m','algorithm_seconds','peak_RSS_MiB','graph_sha256','external_project_imports')}),flush=True)
if __name__=='__main__':
    try:main()
    except MethodBoundary as exc:
        report=getattr(exc,'report',getattr(exc,'certificate',{}));write(ACTIVE_OBSERVER.directory/'method_boundary.json',diagnostic_json(report))
        report['release_sha256']=json.loads((ACTIVE_OBSERVER.directory/'degree_W_state.json').read_text(encoding='utf-8'))['release_sha256'];report['variant']=VARIANT;write(ACTIVE_OBSERVER.directory/'method_boundary.json',diagnostic_json(report));ACTIVE_OBSERVER.close('METHOD_BOUNDARY');sys.exit(3)
    except BaseException as exc:
        if ACTIVE_OBSERVER:
            write(ACTIVE_OBSERVER.directory/'failure.json',dict(type=type(exc).__name__,error=str(exc),stage=ACTIVE_OBSERVER.current))
            ACTIVE_OBSERVER.close('FAILED')
        raise
