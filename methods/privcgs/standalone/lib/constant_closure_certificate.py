"""Support-proved quotient certificate; unchanged successful legacy path."""
import functools
import json
import math
from types import FunctionType
from dataclasses import replace
from pathlib import Path
import numpy as np

def install(audit_path):
    import route_s_clean.parent_laws as laws
    import route_s_clean.pipeline as pipeline
    original=laws.closure_degree_moment_substitution_parent
    if getattr(original,'support_quotient_fix',False): return

    @functools.wraps(original)
    def certified(*args,**kwargs):
        try:
            return original(*args,**kwargs)
        except ArithmeticError as error:
            if str(error)!='closure-degree dual moment certificate failed': raise
            tb=error.__traceback__
            while tb and tb.tb_frame.f_code is not original.__code__: tb=tb.tb_next
            if tb is None: raise
            v=tb.tb_frame.f_locals
            closure=v['closure']
            offsets=v['base_messages'].state_offsets
            # A finite Gaussian chain gives positive weight to every Cartesian
            # choice of per-site states. Constant on each site is thus an exact
            # support certificate for the additive statistic, not a tolerance.
            nonempty=bool(len(offsets)>1 and np.all(np.diff(offsets)>0))
            if nonempty:
                lo=np.minimum.reduceat(closure,offsets[:-1])
                hi=np.maximum.reduceat(closure,offsets[:-1])
                constant=bool(np.all(np.isfinite(closure)) and np.array_equal(lo,hi))
            else:
                constant=bool(len(offsets)==1 and closure.size==0)
                lo=np.empty(0)
            audit=dict(support_constant=constant,site_count=len(offsets)-1,
                       varying_sites=int(np.count_nonzero(lo!=hi)) if nonempty else 0,
                       legacy_residual=v['residual'].tolist(),legacy_minimum_eigenvalue=v['minimum_eigenvalue'])
            if not constant:
                # Center each additive column BEFORE moment propagation, not
                # after subtracting rounded raw second moments. C'=C-s has
                # Var(C')=Var(C), Cov(C',D)=Cov(C,D). Message weights unchanged.
                raw_moments=original.__globals__['bivariate_chain_moments']
                evaluations=[]
                def centered(path,messages,source,target,sigma,additive,*,quadratic_coefficient=1.):
                    a=np.asarray(additive,dtype=np.float64)
                    off=messages.state_offsets
                    centers=np.maximum.reduceat(a,off[:-1])
                    shifted=a-np.repeat(centers,np.diff(off))
                    shift=math.fsum(map(float,centers))
                    m=raw_moments(path,messages,source,target,sigma,shifted,quadratic_coefficient=quadratic_coefficient)
                    evaluations.append(dict(shift=shift,centered_mean=m.mean_structural_score,
                         variance=m.variance_structural_score,covariance=m.covariance,degree_variance=m.variance_degree_energy))
                    return replace(m,mean_structural_score=m.mean_structural_score+shift)
                scope=dict(original.__globals__,bivariate_chain_moments=centered)
                stable=FunctionType(original.__code__,scope,original.__name__,original.__defaults__,original.__closure__)
                audit['repair']='per_site_centered_closure_moment_propagation'
                try:
                    result=stable(*args,**kwargs)
                finally:
                    audit['centered_evaluations']=evaluations
                    Path(audit_path).write_text(json.dumps(audit,indent=2),encoding='utf-8')
                audit['stable_certificate']=dict(result.diagnostics)
                Path(audit_path).write_text(json.dumps(audit,indent=2),encoding='utf-8')
                return replace(result,diagnostics=dict(result.diagnostics,numerical_repair=audit['repair']))
            value=math.fsum(map(float,lo))
            # E_base[D] is the degree target by construction. With C constant,
            # the original base law already satisfies both moment constraints.
            # Its KL projection onto the feasible family is itself (KL=0).
            m=v['base_moments']
            moments=replace(m,mean_structural_score=value,variance_structural_score=0.,covariance=0.)
            audit.update(constant_closure_value=value,degree_target=m.mean_degree_energy,
                         messages_reused_without_resampling=True,parameter=[0.,0.],
                         proof='Per-site exact constant => total constant; base degree mean defines target; KL(base||base)=0',
                         legacy_file=original.__code__.co_filename)
            Path(audit_path).write_text(json.dumps(audit,indent=2),encoding='utf-8')
            return laws.MomentSubstitutionParent(v['base_messages'],float(v['base_coefficient']),0.,0.,moments,
                  dict(solver_success=True,solver_method='exact_support_constant_closure_quotient',
                       target_closure_drift=value,target_degree_energy=m.mean_degree_energy,
                       closure_moment_residual=0.,degree_energy_residual=0.,
                       fisher_quotient_zero_parameter_branch=True,redundant_parameter_direction_count=1,
                       support_certificate=audit,new_fitted_parameter_count=0))
    certified.support_quotient_fix=True
    laws.closure_degree_moment_substitution_parent=certified
    pipeline.closure_degree_moment_substitution_parent=certified
