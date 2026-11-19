"""Same constrained M-step in a nonsingular log-distance coordinate."""
import functools,json,math
from pathlib import Path
import numpy as np

def install(stack,audit_path):
    module=stack.GFL.V41
    original=module._mean_constrained_log_m_step
    if getattr(original,'log_bracket_fix',False): return
    records=[]
    @functools.wraps(original)
    def solve(log_expected,support,target_mean):
        try: return original(log_expected,support,target_mean)
        except RuntimeError as error:
            if str(error) not in ('full-log M-step lower endpoint does not bracket the mean',
                                 'full-log M-step upper endpoint does not bracket the mean'): raise
            reason=str(error)
        lse=module._logsumexp
        q=np.asarray(log_expected,dtype=float)-lse(log_expected)
        x=np.asarray(support,dtype=float);active=np.isfinite(q)
        c=x-float(target_mean);pos=active&(c>0);neg=active&(c<0)
        xmin=float(x[active].min());xmax=float(x[active].max())
        left=-1/(xmax-target_mean);right=1/(target_mean-xmin)
        # lambda=(1-w)*left+w*right, w=sigmoid(t). Construct the
        # endpoint denominators from support differences (no 1+lambda*c
        # cancellation), then their convex combination entirely in logs.
        with np.errstate(divide='ignore',invalid='ignore'):
            dl=np.log((xmax-x)/(xmax-target_mean))
            du=np.log((x-xmin)/(target_mean-xmin))
        def denominator(t):
            return np.logaddexp(dl-np.logaddexp(0.,t),du-np.logaddexp(0.,-t))
        def equation(t):
            d=denominator(t)
            return float(lse(q[pos]+np.log(c[pos])-d[pos])-lse(q[neg]+np.log(-c[neg])-d[neg]))
        lo=-1.;hi=1.
        while equation(lo)<=0:
            lo*=2
            if not math.isfinite(lo): raise ArithmeticError('log bracket overflow')
        while equation(hi)>=0:
            hi*=2
            if not math.isfinite(hi): raise ArithmeticError('log bracket overflow')
        bracket=[lo,hi]
        for _ in range(128):
            mid=lo+(hi-lo)/2
            if mid==lo or mid==hi: break
            sign=equation(mid)
            if sign==0: lo=hi=mid;break
            if sign>0: lo=mid
            else: hi=mid
        t=lo+(hi-lo)/2
        result=np.full(q.size,-math.inf)
        result[active]=q[active]-denominator(t)[active]
        result-=lse(result)
        mean=float(np.dot(x,np.exp(result)))
        bound=4096*np.finfo(float).eps*(1+abs(target_mean))
        if abs(mean-target_mean)>bound:
            raise ArithmeticError(f'log bracket mean certificate failed: {mean} vs {target_mean}')
        w=float(np.exp(-np.logaddexp(0.,-t)))
        tilt=(1-w)*left+w*right
        records.append(dict(legacy_failure=reason,log_coordinate=t,expanded_log_bracket=bracket,
                  mean=mean,target_mean=target_mean,mean_error=mean-target_mean,mean_bound=bound,
                  equation_log_ratio=equation(t),tilt_float_for_diagnostics_only=tilt,
                  float_tilt_at_boundary=bool(tilt==left or tilt==right)))
        Path(audit_path).write_text(json.dumps(dict(repaired_calls=len(records),cases=records),indent=2),encoding='utf-8')
        return np.ascontiguousarray(result),float(tilt)
    solve.log_bracket_fix=True
    module._mean_constrained_log_m_step=solve
