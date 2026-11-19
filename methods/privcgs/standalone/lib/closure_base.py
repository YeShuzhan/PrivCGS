"""Simultaneous Gaussian count / degree-norm interval; no true graph inputs."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2, norm

def original_wedge_interval(noisy_degree, sigma, alpha=0.05/3):
    y=np.asarray(noisy_degree,dtype=np.float64); n=y.size
    x=float(np.dot(y-.5,y-.5)/(sigma*sigma))
    upper_physical=float(n*(n-1)*(n-2)//2)
    def inverse(p):
        f0=float(ncx2.cdf(x,n,0))
        if not math.isfinite(f0): raise ArithmeticError('noncentral chi-square cdf nonfinite')
        if f0<=p: return 0.
        upper=max(x,1.)
        for _ in range(80):
            if ncx2.cdf(x,n,upper)<p: break
            upper*=2
        else: raise ArithmeticError('noncentral chi-square bracket failed')
        return float(brentq(lambda lam:float(ncx2.cdf(x,n,lam))-p,0.,upper,xtol=1e-8,rtol=1e-12))
    ll=inverse(1-alpha/2);lu=inverse(alpha/2)
    lo=max(0.,(sigma*sigma*ll-n/4)/2)
    hi=min(upper_physical,max(0.,(sigma*sigma*lu-n/4)/2))
    if hi<lo: raise ArithmeticError('degree norm confidence interval empty')
    return lo,hi,{'chi_square_statistic':x,'df':int(n),'noncentrality_interval':[ll,lu]}

def closure_interval(release, synthetic_wedges, alpha=.05, degree_wedge_state=None):
    d=release.diagnostics; c=release.checkpoint
    sigma=float(d['closure_count_noise_std'])
    yH=float(d['noisy_projected_wedge_count']);yJ=float(d['noisy_projected_triangle_incidence'])
    z=float(norm.ppf(1-alpha/6))
    report={'alpha_per_cell':alpha,'coverage_method':'Bonferroni degree-norm nc-chi2 + two Gaussian count coordinates',
            'count_sigma':sigma,'count_z':z,'noisy_projected_wedges':yH,'noisy_projected_incidence':yJ,
            'synthetic_wedges':int(synthetic_wedges),'projection_cap':int(release.transcript.projection_degree_cap),
            'uses_stabilized_ratio_as_gaussian':False,'uses_original_truth':False}
    try:
        gL,gU,inv=original_wedge_interval(c.continuous_noisy_degree,float(c.degree_sigma),alpha/3) if degree_wedge_state is None else degree_wedge_state
        hL=max(0.,yH-z*sigma);hU=max(0.,yH+z*sigma)
        jL=max(0.,yJ-z*sigma);jU=min(yJ+z*sigma,hU)
        hL=max(hL,jL)
        if jU<jL or hL>hU or hL>gU: raise ArithmeticError('count/degree confidence box has no physical intersection')
        if gU==0: lower=upper=0.
        else:
            lower=max(0.,jL/gU)
            upper=min(1.,1.5+(min(jU,hL)-1.5*hL)/gU)
        if upper<lower: raise ArithmeticError('C interval empty')
        report.update({'original_wedge_interval':[gL,gU],'projected_wedge_interval':[hL,hU],
            'projected_incidence_interval':[jL,jU],'degree_norm_inversion':inv,'status':'identified_interval'})
    except (ArithmeticError,ValueError,FloatingPointError) as exc:
        lower,upper=0.,1.
        report.update({'status':'uninformative_interval','reason':str(exc)})
    tl=int(math.ceil(lower*synthetic_wedges/3-1e-10));tu=int(math.floor(upper*synthetic_wedges/3+1e-10))
    if tl>tu:
        # No integer in the inferred interval: do not manufacture a target.
        report.update({'status':'uninformative_interval','reason':'no integer triangle count in interval'})
        lower,upper=0.,1.;tl,tu=0,int(synthetic_wedges//3)
    report.update({'original_C_interval':[lower,upper],'synthetic_triangle_interval':[tl,tu]})
    return tl,tu,report
