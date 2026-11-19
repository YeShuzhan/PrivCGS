"""Single-cell benchmark adapter for the five independently saved zCDP variants.

Calls the existing main function without its broken outer __main__ aggregation.
Generation and its RNG draws are preserved; native7 evaluation is disabled.
Public deterministic seeds and raw utility metrics are benchmark conventions.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, importlib.util, inspect, json, random, shutil, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent/'code'
sys.path.insert(0,str(BASE/'author_zip_zcdp'))
import zcdp_config as privacy
from native_metrics_22only import generation_only_function, GenerationCaptured

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['DER','LDPGen','TmF','PrivGraph','PrivHRG'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--epsilon',type=float)
    budget.add_argument('--rho',type=float)
    parser.add_argument('--delta',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cpu',type=int,default=4)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--n-nodes',type=int,required=True)
    parser.add_argument('--der-counts',choices=['author_uint64','int64_compat','numeric_compat_v2'],default='author_uint64')
    parser.add_argument('--require-certificate',action='store_true')
    args=parser.parse_args()
    if args.method!='DER' and args.der_counts!='author_uint64':parser.error('--der-counts is only applicable to DER')
    if args.require_certificate: privacy.require_end_to_end_certificate(args.method)
    epsilon=args.epsilon if args.epsilon is not None else privacy.rho_to_epsilon(args.rho,args.delta)
    privacy.configure(args.delta,args.rho)
    privacy.epsilon_delta_to_rho(epsilon,args.delta)
    if not 0<=args.seed<2**32-4: parser.error('seed must fit uint32 with successor streams')
    if os.environ.get('PYTHONHASHSEED')!=str(args.seed):
        parser.error('set PYTHONHASHSEED to --seed before starting Python')
    source=args.input.resolve(strict=True);out=args.output.resolve()
    if not out.resolve().is_relative_to(Path(__file__).resolve().parent):parser.error('artifacts must remain in the independent server experiment')
    import psutil
    process=psutil.Process();process.cpu_affinity([args.cpu]);process.nice(19)
    if psutil.virtual_memory().available<4*2**30:raise RuntimeError('less than4GiB available memory')
    import numpy as np
    from threadpoolctl import threadpool_limits
    import networkx as nx
    out.mkdir(parents=True,exist_ok=False)
    runtime=out/'runtime';(runtime/'data').mkdir(parents=True)
    shutil.copy2(source,runtime/'data/dataset.txt')
    root_hashes={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*.py') if '__pycache__' not in str(p)}
    write(out/'protocol.json',dict(method=args.method,epsilon_target=epsilon,rho_argument=args.rho,
        delta=args.delta,seed=args.seed,input=str(source),input_sha256=sha(source),
        cpu=[args.cpu],priority='IDLE',numerical_threads=1,source_hashes=root_hashes,
        author_algorithm_flow_preserved=True,end_to_end_zcdp_certified=False,
        benchmark_only=True,metrics_are_not_private_outputs=True,
        native7_evaluation=False,adapter_version='native7_skip_v1'))
    random.seed(args.seed);np.random.seed(args.seed)
    os.chdir(runtime)
    started=time.perf_counter();started_wall=time.time();captured=dict(native_metrics=None,native7_evaluation=False)
    sys.setrecursionlimit(200000)
    if args.method=='PrivHRG':
        module=load('hrg_zcdp',BASE/'privhrg_zcdp/mechanism.py')
        rows=np.atleast_2d(np.loadtxt(source,dtype=np.int64)).tolist()
        with threadpool_limits(limits=1): generated,diagnostics=module.generate_one(rows,epsilon,args.seed,args.delta)
        arr=np.asarray(generated,dtype=np.int64).reshape(-1,2)
        np.savetxt(out/'synthetic.edgelist',arr,fmt='%d',newline='\r\n')
        write(out/'privacy_accounting.json',diagnostics['budget'])
        write(out/'diagnostics_nonprivate.json',diagnostics)
        captured.update(nodes=diagnostics['n_nodes'],edges=len(arr),native_metrics=None)
        assert diagnostics['n_nodes']==args.n_nodes
        write(out/'generation.json',dict(status='GENERATED',method=args.method,dataset=args.dataset,epsilon=epsilon,seed=args.seed,n_nodes=args.n_nodes,n_edges=len(arr),graph_sha256=sha(out/'synthetic.edgelist'),algorithm_wall_seconds_including_yield=diagnostics['algorithm_internal_seconds'],timing_windows=[[started_wall,time.time()]],timing_scope='Inherited PrivHRG algorithm_internal_seconds; input/external saving/evaluation excluded; yield correction approximate window.',end_to_end_zcdp_certified=False,algorithm_flow_preserved=True))
    else:
        if args.method=='PrivGraph':
            folder=BASE/'privgraph_N20_zcdp';sys.path.insert(0,str(folder))
            module=load('privgraph_zcdp',folder/'main.py');function=module.main_func
            plan=module.calibrate(epsilon,args.delta)
            kwargs=dict(dataset_name='dataset',eps=[epsilon],N=20,exp_num=1,save_csv=False,delta=args.delta)
        else:
            folder=BASE/'author_zip_zcdp'
            if args.method=='DER' and args.der_counts=='int64_compat':
                folder=BASE/'author_zip_zcdp_der_int64_compat_v1';sys.path.insert(0,str(folder))
            elif args.method=='DER' and args.der_counts=='numeric_compat_v2':
                folder=BASE/'author_zip_zcdp_der_numeric_compat_v2';sys.path.insert(0,str(folder))
            module=load('zip_'+args.method,folder/('main_'+('LDPGen2' if args.method=='LDPGen' else args.method)+'_zcdp.py'))
            function=getattr(module,'main_func_'+args.method)
            kwargs=dict(root_path='./data/',dataset_name='dataset',eps=[epsilon],exp_num=1,save_csv=False)
            if args.method=='DER':kwargs['fixh']=6
            if args.method=='LDPGen':kwargs.update(k1=10,type1='LDPGen')
            # Public n is used only for the component calibration, not to alter loader behavior.
            mat,ids=module.get_mat(str(source))
            plan=privacy.der_plan(epsilon,len(mat)) if args.method=='DER' else (privacy.ldpgen_plan(epsilon) if args.method=='LDPGen' else privacy.tmf_plan(epsilon))
        function,transform_report=generation_only_function(function)
        write(out/'generation_adapter_audit.json',transform_report)
        write(out/'privacy_accounting.json',plan)
        der_timer={}
        if args.method=='DER':
            original_init=module.dnode.initialize
            def timed_initialize(self,*values,**keywords):
                der_timer['start']=time.time()
                return original_init(self,*values,**keywords)
            module.dnode.initialize=timed_initialize
        old_from=nx.from_numpy_array
        def observed_from(*values,**kwargs):
            matrix_finished=time.time()
            caller=inspect.currentframe().f_back
            if caller.f_code is function.__code__:
                local=caller.f_locals
                if 'mat2' in local and values[0] is local['mat2']:
                    a,b=np.where(np.triu(values[0],1)!=0)
                    np.savetxt(out/'synthetic.edgelist',np.column_stack((a,b)),fmt='%d',newline='\r\n')
                    assert len(values[0])==args.n_nodes
                    windows=[[float(local['t1']),matrix_finished]]
                    if args.method=='DER':
                        windows=[[float(local['t_r1']),float(local['t_r2'])],
                                 [float(der_timer['start']),matrix_finished]]
                    captured.update(nodes=len(values[0]),edges=len(a))
                    write(out/'generation.json',dict(status='GENERATED',method=args.method,dataset=args.dataset,
                        epsilon=args.epsilon,seed=args.seed,n_nodes=args.n_nodes,n_edges=len(a),
                        graph_sha256=sha(out/'synthetic.edgelist'),timing_windows=windows,
                        algorithm_wall_seconds_including_yield=sum(b-a for a,b in windows),
                        timing_scope='Native generation windows only; input, original metrics, output save, synthetic native metrics and frozen22 evaluation excluded; own yield time corrected by supervisor.',
                        end_to_end_zcdp_certified=False,algorithm_flow_preserved=True))
                    print(json.dumps({'stage':'GENERATED','method':args.method,'edges':len(a),'native7_evaluation':False}),flush=True)
                    raise GenerationCaptured()
            return old_from(*values,**kwargs)
        nx.from_numpy_array=observed_from
        try:
            with threadpool_limits(limits=1):function(**kwargs)
        except GenerationCaptured:
            assert (out/'generation.json').exists()
        finally:
            nx.from_numpy_array=old_from
    assert (out/'synthetic.edgelist').exists()
    assert all(sha(BASE/p)==h for p,h in root_hashes.items())
    arr=np.atleast_2d(np.loadtxt(out/'synthetic.edgelist',dtype=np.int64))
    if arr.size:
        assert np.all(arr[:,0]<arr[:,1]) and len(np.unique(arr,axis=0))==len(arr)
    result=dict(status='COMPLETE',method=args.method,epsilon_target=epsilon,delta=args.delta,
        seed=args.seed,wall_seconds=time.perf_counter()-started,
        peak_process_MiB=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024,
        graph_sha256=sha(out/'synthetic.edgelist'),sources_unchanged=True,
        algorithm_flow_preserved=True,end_to_end_zcdp_certified=False,**captured)
    write(out/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

