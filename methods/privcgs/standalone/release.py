"""The three-channel Gaussian release. Private input stops here."""
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
    ap.add_argument('--n-nodes',type=int);ap.add_argument('--n-edges',type=int)
    ap.add_argument('--nodes',type=Path,help='One public node ID per line, in public order')
    ap.add_argument('--dataset',required=True);ap.add_argument('--epsilon',required=True,type=float)
    ap.add_argument('--r',type=int,choices=(32,),default=32)
    ap.add_argument('--mode',choices=('benchmark','private'),default='benchmark')
    ap.add_argument('--seed',type=int);ap.add_argument('--public-seed',type=int)
    ap.add_argument('--fractions',nargs=3,type=float,default=[.3,.3,.4])
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--cpu',default='')
    ap.add_argument('--compiler',help='Compiler executable, alternatively set CXX')
    args=ap.parse_args()
    if args.mode=='private':
        if args.nodes is None or args.public_seed is None or not 0<=args.public_seed<2**32:
            ap.error('Private mode requires --nodes and an unsigned 32-bit --public-seed.')
        if any(value is not None for value in (args.seed,args.n_nodes,args.n_edges)):
            ap.error('Private mode takes its domain from --nodes and does not take benchmark counts or seeds.')
        public_seed=args.public_seed
    else:
        if args.n_nodes is None or args.n_edges is None or args.public_seed is not None:
            ap.error('Benchmark mode requires --n-nodes and --n-edges, and uses --seed.')
        args.seed=42 if args.seed is None else args.seed
        public_seed=args.seed
    os.umask(0o077)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert args.epsilon>0 and min(args.fractions)>0 and abs(sum(args.fractions)-1)<1e-12
    configure(args.cpu,args.compiler)
    from route_s_clean.graph_carrier import CSRGraph
    from csr_input_v5 import read_arrays,sha
    from release_design import PrivateGaussian, spawn_release_streams
    from common import sanitized_release,numeric_release_hash
    import release_count40
    start=time.perf_counter()
    edges,public=read_arrays(args.graph,args.n_nodes,args.n_edges,nodes=args.nodes)
    n_nodes=len(public)
    if n_nodes<3:
        raise ValueError('PrivCGS requires a public domain of at least three nodes.')
    csr=CSRGraph.from_edges(edges,n_nodes);del edges
    input_seconds=time.perf_counter()-start
    if args.mode=='private':
        private_rng=PrivateGaussian()
        degree_rng=structure_rng=closure_rng=private_rng
    else:
        degree_rng,structure_rng,closure_rng,_=spawn_release_streams(args.seed)
    d,o,c=args.fractions;native_before=TOTALS['native_setup'];start=time.perf_counter()
    release=release_count40.release_count40_from_csr(
        csr.csr_offsets,csr.csr_neighbors,epsilon=args.epsilon,delta=1e-5,
        public_experiment_id=public_seed,degree_rho_fraction=d,operator_basis_rank=args.r,
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
    result=dict(dataset=args.dataset,epsilon=args.epsilon,n_nodes=n_nodes,
        randomness_mode=args.mode,public_seed=public_seed,
        fractions=args.fractions,delta=1e-5,path='release.pkl',sha256=sha(path),numeric_release_sha256=numeric_release_hash(release),
        rho_total=release.transcript.rho_total,rho_degree=release.transcript.rho_degree,rho_operator=release.transcript.rho_operator,
        rho_closure=release.transcript.rho_closure,release_algorithm_seconds=elapsed,release_wall_seconds=wall_seconds,native_setup_seconds=native_setup_seconds,
        peak_RSS_MiB=algorithm_peak,public_node_ids_path='public_node_ids.npy',public_node_ids_sha256=sha(out/'public_node_ids.npy'),
        projection_degree_cap=release.transcript.projection_degree_cap,count_sigma=release.diagnostics['closure_count_noise_std'],
        algorithm='MCR1-D40SCORE1 frozen count40 cap-selection score')
    result.update(operator_basis_rank=args.r,fixed_D=None,base_reuse=None)
    if args.mode=='benchmark':
        result['seed']=args.seed
    (out/'release_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.mode=='benchmark':
        write(out/'private_provenance.json',dict(private_input_path=str(args.graph.resolve()),original_edges=args.n_edges,graph_sha256=sha(args.graph),input_seconds=input_seconds,note='Internal provenance for a seeded benchmark.'))
    print(json.dumps(dict(status='RELEASE_COMPLETE',numeric_release_sha256=result['numeric_release_sha256'],seconds=elapsed)),flush=True)
if __name__=='__main__':main()
