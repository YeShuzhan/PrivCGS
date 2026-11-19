"""Released-degree certificates for mutual-top-D projection loss."""
import math
import numpy as np
from scipy.stats import norm
from closure_base import original_wedge_interval

def projection_loss_bounds(noisy_degree,sigma,degree_cap,alpha_upper):
    y=np.asarray(noisy_degree,dtype=np.float64);n=len(y);D=int(degree_cap)
    if n<3 or D<2 or D>=n or not np.all(np.isfinite(y)) or not math.isfinite(sigma) or sigma<=0:raise ValueError('projection evidence input')
    z=float(norm.isf(alpha_upper/n));upper=np.clip(np.floor(y+sigma*z),0,n-1).astype(np.int64)
    maximum=int(upper.max());tail=[int(u) for u in upper if u>D]
    edges=sum(u-D for u in tail)
    triangles=sum(u*(u-1)//2-D*(D-1)//2 for u in tail)
    wedges=sum((u-D)*(u+maximum-2) for u in tail)
    return {'upper_degree_alpha':float(alpha_upper),'upper_degree_z':z,'maximum_degree_upper':maximum,
        'nodes_with_upper_degree_above_cap':len(tail),'removed_edges_upper':edges,
        'destroyed_triangles_upper':triangles,'destroyed_wedges_upper':wedges,
        'projection_identity_certified_on_degree_event':not tail,'degree_upper':upper}

def physical_C_bounds(gL,gU,hL,hU,jL,jU,triangle_loss,wedge_loss):
    hL=max(0.,hL,jL);hU=max(0.,hU);jL=max(0.,jL);jU=min(jU,hU)
    ggL=max(gL,hL);ggU=min(gU,hU+wedge_loss);hU=min(hU,ggU);jU=min(jU,hU)
    if ggL>ggU or jU<jL or hL>hU or hL>ggU:raise ArithmeticError('count/degree/projection confidence box has no physical intersection')
    if ggU==0:lower=upper=wedge_upper=tail_upper=0.
    else:
        lower=max(0.,jL/ggU)
        wedge_upper=min(1.,1.5+(min(jU,hL)-1.5*hL)/ggU)
        tail_upper=min(1.,(jU+3*triangle_loss)/ggL) if ggL>0 else 1.
        upper=min(wedge_upper,tail_upper)
    if upper<lower:raise ArithmeticError('C interval empty')
    return lower,upper,{'tightened_original_wedge_interval':[ggL,ggU],'projected_wedge_interval':[hL,hU],
        'projected_incidence_interval':[jL,jU],'wedge_relation_C_upper':wedge_upper,'degree_tail_C_upper':tail_upper}

def closure_interval(release,synthetic_wedges,alpha=.05,degree_wedge_state=None):
    d=release.diagnostics;c=release.checkpoint;sigma=float(d['closure_count_noise_std'])
    yH=float(d['noisy_projected_wedge_count']);yJ=float(d['noisy_projected_triangle_incidence'])
    alpha_W=alpha/3;other=(alpha-alpha_W)/3;z=float(norm.isf(other/2))
    gL,gU,inv=original_wedge_interval(c.continuous_noisy_degree,float(c.degree_sigma),alpha_W) if degree_wedge_state is None else degree_wedge_state
    loss=projection_loss_bounds(c.continuous_noisy_degree,float(c.degree_sigma),int(release.transcript.projection_degree_cap),other)
    upper_degree=loss.pop('degree_upper')
    import hashlib
    report={'alpha_per_cell':alpha,'coverage_method':'Union of fixed degree-norm event, two conditional Gaussian counts, simultaneous upper-degree event',
        'alpha_allocation':{'degree_norm':alpha_W,'projected_W':other,'projected_J':other,'simultaneous_degree_upper':other},
        'count_sigma':sigma,'count_z':z,'noisy_projected_wedges':yH,'noisy_projected_incidence':yJ,
        'synthetic_wedges':int(synthetic_wedges),'projection_cap':int(release.transcript.projection_degree_cap),
        'original_wedge_interval':[gL,gU],'degree_norm_inversion':inv,'projection_loss_bounds':loss,
        'upper_degree_sha256':hashlib.sha256(upper_degree.tobytes()).hexdigest(),
        'uses_stabilized_ratio_as_gaussian':False,'uses_original_truth':False,'private_query_added':False}
    try:
        lower,upper,details=physical_C_bounds(gL,gU,yH-z*sigma,yH+z*sigma,yJ-z*sigma,yJ+z*sigma,
            loss['destroyed_triangles_upper'],loss['destroyed_wedges_upper'])
        report.update(details);report['status']='identified_interval'
    except ArithmeticError as exc:
        lower,upper=0.,1.;report.update(status='uninformative_interval',reason=str(exc))
    tl=int(math.ceil(lower*synthetic_wedges/3-1e-10));tu=int(math.floor(upper*synthetic_wedges/3+1e-10))
    if tl>tu:
        lower,upper=0.,1.;tl,tu=0,int(synthetic_wedges//3)
        report.update(status='uninformative_interval',reason='no integer triangle count in interval')
    report.update(original_C_interval=[lower,upper],synthetic_triangle_interval=[tl,tu])
    return tl,tu,report
