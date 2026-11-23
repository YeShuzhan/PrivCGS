"""The frozen three-channel Gaussian release; private input stops here."""
from pathlib import Path
import argparse,json,os,pickle,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'lib'))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'4')
import numpy as np
import psutil
from execution import configure, sha, write
from execution_accounting import TOTALS
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--graph',required=True,type=Path)
    ap.add_argument('--n-nodes',required=True,type=int);ap.add_argument('--n-edges',required=True,type=int)
    ap.add_argument('--dataset',required=True);ap.add_argument('--epsilon',required=True,type=float)
    ap.add_argument('--r',type=int,choices=(32,),default=32)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--fractions',nargs=3,type=float,default=[.3,.3,.4])
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--cpu',default='')
    ap.add_argument('--compiler',help='Compiler executable, alternatively set CXX')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert args.epsilon>0 and min(args.fractions)>0 and abs(sum(args.fractions)-1)<1e-12
    configure(args.cpu,args.compiler)
    from route_s_clean.graph_carrier import CSRGraph
    from csr_input_v5 import read_arrays,sha
    from release_design import spawn_release_streams
    from common import sanitized_release,numeric_release_hash
    import release_count40
    start=time.perf_counter()
    edges,public=read_arrays(args.graph,args.n_nodes,args.n_edges)
    csr=CSRGraph.from_edges(edges,args.n_nodes);del edges
    input_seconds=time.perf_counter()-start
    degree_rng,structure_rng,closure_rng,_=spawn_release_streams(args.seed)
    d,o,c=args.fractions;native_before=TOTALS['native_setup'];start=time.perf_counter()
    release=release_count40.release_count40_from_csr(
        csr.csr_offsets,csr.csr_neighbors,epsilon=args.epsilon,delta=1e-5,
        public_experiment_id=args.seed,degree_rho_fraction=d,operator_basis_rank=args.r,
        degree_private_rng=degree_rng,structural_private_rng=structure_rng,closure_private_rng=closure_rng,
        closure_rho_fraction_of_nondegree=c/(1-d),projection_candidate_caps=None)
    release=sanitized_release(release)
    wall_seconds=time.perf_counter()-start
    native_setup_seconds=TOTALS['native_setup']-native_before
    elapsed=wall_seconds-native_setup_seconds
    algorithm_peak=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024
    assert np.allclose([release.transcript.rho_degree,release.transcript.rho_operator,release.transcript.rho_closure],
                       np.asarray(args.fractions)*release.transcript.rho_total,rtol=1e-12,atol=1e-15)
    path=out/'release.pkl'
    with path.open('wb') as f:pickle.dump(release,f,protocol=5)
    np.save(out/'public_node_ids.npy',public)
    result=dict(dataset=args.dataset,epsilon=args.epsilon,seed=args.seed,n_nodes=args.n_nodes,
        fractions=args.fractions,delta=1e-5,path='release.pkl',sha256=sha(path),numeric_release_sha256=numeric_release_hash(release),
        rho_total=release.transcript.rho_total,rho_degree=release.transcript.rho_degree,rho_operator=release.transcript.rho_operator,
        rho_closure=release.transcript.rho_closure,release_algorithm_seconds=elapsed,release_wall_seconds=wall_seconds,native_setup_seconds=native_setup_seconds,
        peak_RSS_MiB=algorithm_peak,public_node_ids_path='public_node_ids.npy',public_node_ids_sha256=sha(out/'public_node_ids.npy'),
        projection_degree_cap=release.transcript.projection_degree_cap,count_sigma=release.diagnostics['closure_count_noise_std'],
        algorithm='MCR1-D40SCORE1 frozen count40 cap-selection score')
    result.update(operator_basis_rank=args.r,fixed_D=None,base_reuse=None)
    (out/'release_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write(out/'private_provenance.json',dict(private_input_path=str(args.graph.resolve()),original_edges=args.n_edges,graph_sha256=sha(args.graph),input_seconds=input_seconds,note='Local evaluation/provenance only; not a DP output. Fixed experimental noise seed is not a secure production release.'))
    print(json.dumps(dict(status='RELEASE_COMPLETE',numeric_release_sha256=result['numeric_release_sha256'],seconds=elapsed)),flush=True)
if __name__=='__main__':main()
