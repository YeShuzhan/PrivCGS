"""Original continuous solver and certified quantile reference; no refitting/floor."""
import inspect,math,time,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal,Context,localcontext,ROUND_FLOOR,ROUND_CEILING,ROUND_HALF_EVEN
import numpy as np

def original_spectrum(checkpoint,skip_integerization=True,endpoint_mass=None):
 import spectrum_solver as solver
 import gfl_mpe_rkss1_pprs1_v1 as native
 from runtime import accelerated_spectrum
 native.THREAD_COUNT=int(__import__('os').environ.get('PRIVCGS_THREADS','8'));native.reset_plan()
 if skip_integerization:
  # The pinned original Gaussian source prefix is executed unchanged through
  # represented-mean validation. The old LR/mass/EG tail is intentionally absent.
  src=inspect.getsource(solver.released_spectrum_in_memory_v41)
  assert src.count('    expected_count = float(node_count) * probability')==1
  prefix=src.split('    expected_count = float(node_count) * probability')[0]
  prefix+='    return probability,log_probability,dict(iterations=iterations,duality_gap_log=final_log_gap,represented_mean=represented_mean,original_integerization_executed=False)\n'
  if endpoint_mass is not None:
   prefix=prefix.replace('    endpoint_mass = 2 * edge_count', '    endpoint_mass = '+str(int(endpoint_mass)))
   assert '    endpoint_mass = '+str(int(endpoint_mass)) in prefix
  namespace=dict(solver.__dict__);exec(compile(prefix,str(Path(solver.__file__).resolve())+'::continuous-prefix','exec'),namespace)
  p,lp,diagnostics=namespace['released_spectrum_in_memory_v41'](checkpoint)
  return SimpleNamespace(probability=p,diagnostics=diagnostics),lp
 state={};saved={k:getattr(solver,k) for k in ('_maximum_entropy_log_initialization','_mean_constrained_log_m_step')}
 def wrap(fn):
  def call(*a,**kw):
   value=fn(*a,**kw);state['logp']=value[0].copy();return value
  return call
 for k,fn in saved.items():setattr(solver,k,wrap(fn))
 try:r=accelerated_spectrum(checkpoint,threads=2)
 finally:
  for k,fn in saved.items():setattr(solver,k,fn)
 lp=state['logp'];p=np.exp(lp);p/=float(np.sum(p));assert np.array_equal(p,r.probability)
 return r,lp

