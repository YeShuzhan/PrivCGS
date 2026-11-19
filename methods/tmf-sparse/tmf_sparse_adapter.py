"""Run the byte-preserved author-lineage TmF using reference or exact sparse storage."""
import argparse, hashlib, importlib.util, json, os, pickle, random, resource, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['reference','sparse'],required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epsilon',type=float,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--cpu',type=int,default=15);ap.add_argument('--cell',required=True);ap.add_argument('--validation',action='store_true');a=ap.parse_args()
    cfg=json.loads((ROOT/'CONFIG.json').read_bytes());rows=json.loads((ROOT/'CATALOG.json').read_bytes())
    row=next(x for x in cfg['validation_rows'].values() if x['cell']==a.cell) if a.validation else next(x for x in rows if x['cell']==a.cell)
    assert row['cell']==a.cell and row['method']=='TmF' and a.input.resolve()==Path(row['input']).resolve()
    assert sha(a.input)==row['input_sha256'] and a.epsilon==row['epsilon'] and a.seed==row['seed']
    assert a.epsilon in cfg['epsilon_values'] and a.seed in cfg['seed_values']
    assert a.mode=='sparse' or a.validation and row['dataset']=='Polbooks' and row['n_nodes']==105
    assert os.environ['PYTHONHASHSEED']==str(a.seed)
    import psutil
    psutil.Process().cpu_affinity([a.cpu]);psutil.Process().nice(19)
    import numpy as np
    import community
    from exact_sparse_storage import install,digest
    legacy=ROOT/'frozen_legacy'
    for n,v in json.loads((ROOT/'FROZEN_LEGACY_ORIGINS.json').read_bytes()).items():assert sha(legacy/n)==v['sha256'],n
    trace={'test_only':a.validation,'dataset':row['dataset'],'cell':row['cell'],'mode':a.mode,'epsilon':a.epsilon,'seed':a.seed,'matrix_allocations':[],'random_calls':[],'Louvain':[],'outer_dead_read_elided':0,'actual_TmF_input_reads':0,'input_sha256':sha(a.input)}
    trace_path=a.output.parent/(a.output.name+'.trace.json');trace_path.parent.mkdir(parents=True,exist_ok=True)
    install_info=install(legacy,a.mode,trace);trace['install']=install_info
    original_partition=community.best_partition
    def partition(*args,**kwargs):
        before=digest(np.random.get_state());py_before=digest(random.getstate());value=original_partition(*args,**kwargs)
        trace['Louvain'].append(dict(before_numpy=before,after_numpy=digest(np.random.get_state()),before_python=py_before,after_python=digest(random.getstate()),partition_hash=digest(sorted(value.items()))));return value
    community.best_partition=partition
    randint=np.random.randint
    def traced_randint(*args,**kwargs):
        value=randint(*args,**kwargs);trace['random_calls'].append(dict(name='randint',args=repr(args),kwargs=repr(kwargs),output_hash=digest(value)));return value
    np.random.randint=traced_randint
    sys.path.insert(0,str(legacy));spec=importlib.util.spec_from_file_location('frozen_tmf_generation_entry',legacy/'generate_cell_linux_v2.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry);original_load=entry.load
    def scoped_load(name,path):
        m=original_load(name,path)
        if name=='zip_TmF':
            get_mat=m.get_mat;noise=m.laplace
            def scoped_input(p):
                if inspect_frame_is_outer():trace['outer_dead_read_elided']+=1;return None,None
                trace['actual_TmF_input_reads']+=1;return get_mat(p)
            def inspect_frame_is_outer():
                import inspect
                return inspect.currentframe().f_back.f_back.f_code is entry.main.__code__
            def traced_noise(*args,**kwargs):
                value=noise(*args,**kwargs);trace['random_calls'].append(dict(name='normal_alias',args=repr(args),kwargs=repr(kwargs),output_hash=digest(value)));return value
            m.get_mat=scoped_input;m.laplace=traced_noise
        return m
    entry.load=scoped_load
    sys.argv=[str(legacy/'generate_cell_linux_v2.py'),'--method','TmF','--input',str(a.input),'--output',str(a.output),'--epsilon',str(a.epsilon),'--seed',str(a.seed),'--cpu',str(a.cpu),'--dataset',row['dataset'],'--n-nodes',str(row['n_nodes'])]
    began=time.time()
    try:
        entry.main()
        assert trace['outer_dead_read_elided']==trace['actual_TmF_input_reads']==1
        assert len(trace['matrix_allocations'])==2
        trace['status']='SUCCESS';trace['graph_sha256']=sha(a.output/'synthetic.edgelist');trace['accounting_sha256']=sha(a.output/'privacy_accounting.json');trace['generation_adapter_audit_sha256']=sha(a.output/'generation_adapter_audit.json')
    except BaseException:
        trace['status']='ERROR';trace['traceback']=traceback.format_exc();raise
    finally:
        trace['numpy_random_final_state_hash']=digest(np.random.get_state());trace['python_random_final_state_hash']=digest(random.getstate());trace['elapsed_seconds']=time.time()-began;trace['process_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;write(trace_path,trace)

if __name__=='__main__':main()
