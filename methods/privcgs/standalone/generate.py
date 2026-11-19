"""Run the complete private-release -> MCR1 -> W4 -> Cstar chain."""
from pathlib import Path
import argparse, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--n-nodes',type=int,required=True)
    p.add_argument('--n-edges',type=int,required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--epsilon',type=float,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fractions',nargs=3,type=float,default=[.3,.3,.4])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',default='')
    p.add_argument('--compiler')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    common=['--dataset',args.dataset,'--seed',str(args.seed),'--cpu',args.cpu]
    if args.compiler:common+=['--compiler',args.compiler]
    env=dict(os.environ);env.setdefault('PYTHONHASHSEED','42')
    began=time.perf_counter()
    release_cmd=[sys.executable,str(ROOT/'release.py'),*common,'--graph',str(args.graph.resolve()),
                 '--n-nodes',str(args.n_nodes),'--n-edges',str(args.n_edges),'--epsilon',str(args.epsilon),
                 '--fractions',*map(str,args.fractions),'--output',str(out/'release')]
    generation_cmd=[sys.executable,str(ROOT/'run.py'),*common,
                    '--release-manifest',str(out/'release/release_manifest.json'),
                    '--output',str(out/'generation'),'--verify-imports']
    commands=[release_cmd,generation_cmd]
    (out/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    for command in commands:
        subprocess.run(command,env=env,check=True)
    release=json.loads((out/'release/release_manifest.json').read_text())
    generation=json.loads((out/'generation/generation_report.json').read_text())
    summary=dict(status='COMPLETE',dataset=args.dataset,epsilon=args.epsilon,seed=args.seed,
        n_nodes=args.n_nodes,graph_sha256=generation['graph_sha256'],
        algorithm_seconds=release['release_algorithm_seconds']+generation['algorithm_seconds'],
        algorithm_scope='P1 release + recovery/S/Q/target/completion/W4/Cstar; excludes input, measured setup/cache/diagnostic I/O, checkpoint/output serialization and evaluation',
        process_peak_RSS_MiB=max(release['peak_RSS_MiB'],generation['peak_RSS_MiB']),
        peak_scope='Maximum of two sequential process peaks (both include input/setup), not their sum',
        end_to_end_wall_seconds=time.perf_counter()-began,
        final_graph='generation/edges.edgelist',node_domain='generation/node_domain.json',
        privacy_fractions=release['fractions'],metrics='Run the independently frozen evaluator after graph freeze')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
