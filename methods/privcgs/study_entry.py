"""One isolated scientific configuration; controller observes its actual exit."""
import json,os,sys,runpy,faulthandler
from pathlib import Path
faulthandler.enable()
root=Path(__file__).resolve().parent
configuration=Path(sys.argv[1]).resolve();config=json.loads(configuration.read_bytes())
assert config['delta']==1e-5 and config['epsilon'] in [.5,1,1.5,2,2.5,5] and config['seed'] in [42,520,1314,114514,2024]
assert config['variant'] in ['full','a','b','c','d','e','f','g']
threads=int(os.environ.get('PRIVCGS_THREADS','8'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]=str(threads)
os.environ['PRIVCGS_STUDY_CONFIG']=str(configuration)
sys.path[:0]=[str(root),str(root/'modules'),str(root/'standalone/lib'),str(root/'equivalent')]
from study_hooks import install
install(config)
import repair_fast
repair_fast.install()
target=root/'generation.py';sys.argv=[str(target)]+sys.argv[2:]
runpy.run_path(str(target),run_name='__main__')
