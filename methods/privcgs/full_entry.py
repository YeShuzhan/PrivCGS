"""Run the Full generation stage with the benchmark's calibration settings."""
import json,os,sys,runpy,faulthandler
from pathlib import Path
faulthandler.enable()
root=Path(__file__).resolve().parent
configuration=Path(sys.argv[1]).resolve();config=json.loads(configuration.read_bytes())
assert config['delta']==1e-5 and config['epsilon'] in [.5,1,1.5,2,2.5,5] and config['seed'] in [42,520,1314,114514,2024]
if config['variant'] != 'full' or config.get('upstream_cache'):
    raise ValueError('This entry requires a fresh PrivCGS Full configuration.')
threads=int(os.environ.get('PRIVCGS_THREADS','8'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]=str(threads)
os.environ['PRIVCGS_STUDY_CONFIG']=str(configuration)
sys.path[:0]=[str(root),str(root/'modules'),str(root/'standalone/lib'),str(root/'equivalent')]
import joint_risk
original_refine=joint_risk.refine

def refine(dataset,edges,adjacency,target,current,W,lo,hi,contract,repair,audit_callback=None):
    # The confidence level and joint-risk calibration slack have distinct roles.
    local=dict(contract,alpha_per_cell=config['zeta'])
    result,report,sweeps=original_refine(dataset,edges,adjacency,target,current,W,lo,hi,local,repair,audit_callback)
    if 'JR_triangle_certificate' in local:contract['JR_triangle_certificate']=local['JR_triangle_certificate']
    report.update(coverage_alpha=config['alpha'],optimization_zeta=config['zeta'])
    report.pop('alpha',None)
    return result,report,sweeps

joint_risk.refine=refine
import repair_fast
repair_fast.install()
target=root/'generation.py';sys.argv=[str(target)]+sys.argv[2:]
runpy.run_path(str(target),run_name='__main__')
