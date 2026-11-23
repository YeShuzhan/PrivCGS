"""Independent Linux entry; frozen algorithms and candidate order retained."""
import os,sys,runpy
from pathlib import Path
root=Path(__file__).resolve().parent
phase=sys.argv[1];assert phase in ('release','compile')
threads=int(os.environ.get('PRIVCGS_THREADS','8'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]=str(threads)
sys.path[:0]=[str(root/'modules'),str(root/'standalone/lib'),str(root/'equivalent')]
if phase=='compile':
 import subprocess,json,time
 argv=[os.environ.get('CXX','g++'),'-O3','-std=c++17','-fPIC','-shared','-fopenmp','-fno-fast-math',str(root/'equivalent/repair_native.cpp'),'-o',str(root/'equivalent/repair_native_final.so')]
 with (root/'equivalent/COMPILE.stdout').open('xb') as out,(root/'equivalent/COMPILE.stderr').open('xb') as err:
  p=subprocess.Popen(argv,stdout=out,stderr=err);code=p.wait()
 receipt={'time':time.time(),'argv':argv,'actual_parent_native':code,'pid':p.pid}
 (root/'equivalent/COMPILE.PARENT_NATIVE.json').open('x').write(json.dumps(receipt,indent=2))
 print(json.dumps(receipt),flush=True);sys.exit(code)
target=root/'standalone/release.py'
sys.argv=[str(target)]+sys.argv[2:]
runpy.run_path(str(target),run_name='__main__')
