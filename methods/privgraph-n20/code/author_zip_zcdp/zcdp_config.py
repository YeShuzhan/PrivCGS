"""Component calibration only; preserving received flows prevents full certification.

Gaussian: sigma=Delta2/sqrt(2*rho). Pure epsilon-DP: rho<=epsilon**2/2.
All budget parameters below are public. Delta defaults to the user's 1e-5.
"""
import math

DEFAULT_DELTA=1e-5
_delta=DEFAULT_DELTA
_rho_override=None

def _positive(value,name):
    value=float(value)
    if not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be finite and positive')
    return value

def rho_to_epsilon(rho,delta=DEFAULT_DELTA):
    rho=_positive(rho,'rho'); delta=_positive(delta,'delta')
    if delta>=1: raise ValueError('delta must be <1')
    return rho+2*math.sqrt(rho*-math.log(delta))

def epsilon_delta_to_rho(epsilon,delta=DEFAULT_DELTA):
    epsilon=_positive(epsilon,'epsilon'); delta=_positive(delta,'delta')
    if delta>=1: raise ValueError('delta must be <1')
    L=-math.log(delta)
    return _positive((epsilon/(math.sqrt(L+epsilon)+math.sqrt(L)))**2,'rho')

def gaussian_sigma(rho,sensitivity):
    return _positive(sensitivity,'sensitivity')/math.sqrt(2*_positive(rho,'rho'))

def configure(delta=DEFAULT_DELTA,rho=None):
    global _delta,_rho_override
    delta=_positive(delta,'delta')
    if delta>=1: raise ValueError('delta must be <1')
    _delta=delta
    _rho_override=None if rho is None else _positive(rho,'rho')

def base_plan(epsilon,fractions,names):
    epsilon=_positive(epsilon,'epsilon')
    fractions=[_positive(x,'fraction') for x in fractions]
    if not math.isclose(sum(fractions),1,rel_tol=1e-12):
        raise ValueError('fractions must sum to1')
    rho=epsilon_delta_to_rho(epsilon,_delta) if _rho_override is None else _rho_override
    if _rho_override is not None and not math.isclose(rho_to_epsilon(rho,_delta),epsilon,rel_tol=1e-10):
        raise ValueError('rho and epsilon conversion disagree')
    stages={name:rho*frac for name,frac in zip(names,fractions)}
    return dict(epsilon_target=epsilon,delta=_delta,rho_total=rho,
        stage_fractions=dict(zip(names,fractions)),rho_stages=stages,
        epsilon_from_rho=rho_to_epsilon(sum(stages.values()),_delta),
        adjacency='one undirected unit edge add/remove on a fixed public vertex universe',
        release_scope='synthetic graph only; raw metrics, data-derived node maps, timing and internal diagnostics are non-private research outputs',
        end_to_end_zcdp_certified=False,ideal_gaussian_component_calibration=True)

def ldpgen_plan(epsilon,type1='LDPGen'):
    if type1 not in ('LDPGen','DPGen'): raise ValueError('unknown type1')
    count=2 if type1=='LDPGen' else 3
    p=base_plan(epsilon,[1/count]*count,['phase'+str(i+1) for i in range(count)])
    sigmas=[gaussian_sigma(r,math.sqrt(2)) for r in p['rho_stages'].values()]
    denominators=[2/s for s in sigmas]
    # Preserve the author's branch; public parameter validation avoids the no-noise branch.
    if any(x>=1000 for x in denominators):
        raise ValueError('requested public budget enters the inherited >=1000 no-noise branch')
    p.update(matrix_L2_sensitivity=math.sqrt(2),noise_std=sigmas,noise_denominators=denominators,
        component_guarantee='For fixed public nodes and conditioned previous transcripts, each node-to-group matrix changes in exactly two unit cells. Gaussian stages compose to rho_total; all subsequent mapping/clipping/sampling uses their transcripts.',
        limitations=['original endpoint-inferred node universe is retained','fixed public test seeds and floating-point Gaussian are research-only','integer overflow and data-dependent algorithm failure channels have not been certified'],
        preserved=['fixed k1=10','integer noise truncation','phase2 nonnegative clipping','original proportional mapping','original intra/inter reconstruction and symmetrization'])
    return p

def tmf_plan(epsilon):
    epsilon=_positive(epsilon,'epsilon')
    count_fraction=min(.1,.5*epsilon)/epsilon
    p=base_plan(epsilon,[1-count_fraction,count_fraction],['filter','count'])
    r=p['rho_stages']
    p.update(filter_precision=math.sqrt(2*r['filter']),count_precision=math.sqrt(2*r['count']),
        nominal_filter_sigma=gaussian_sigma(r['filter'],1),count_sigma=gaussian_sigma(r['count'],1),
        guarantee='No full TmF rho guarantee: the retained graph construction is not post-processing of a complete Gaussian adjacency query.',
        limitations=['one-cell processing selects first m of 2m directed nonzero entries','m and nonzero positions are raw-data-dependent','zero-cell supplement re-queries original adjacency','Laplace-derived threshold formula is deliberately retained; these are not Gaussian-tail threshold equations','duplicates and diagonal candidates in supplement are deliberately retained'])
    return p

def der_plan(epsilon,n_nodes,iterations=5):
    n=int(n_nodes); T=int(iterations)
    if n<=2 or T<=0: raise ValueError('original DER labeling requires n>2 and T>0')
    p=base_plan(epsilon,[1/3,1/6,1/6,1/3],['label','count','partition','arrangement'])
    r=p['rho_stages']
    # An edge is incident to at most two disjoint label pairs per iteration.
    # Each pair's difference-score changes by <=2*(n-1)/(n-2).
    sensitivity=2*(n-1)/(n-2)
    sigma=sensitivity*math.sqrt(T/r['label'])
    label_parameter=2*T/sigma  # original code computes 2/(e1/T)
    p.update(label_pair_difference_sensitivity=sensitivity,label_pair_sigma=sigma,
        label_pair_rho=r['label']/(2*T),label_parameter=label_parameter,
        count_precision=math.sqrt(2*r['count']),
        partition_pure_epsilon=math.sqrt(2*r['partition']),
        arrangement_base_pure_epsilon=math.sqrt(2*r['arrangement']),
        root_count_gaussian=False,
        guarantee='Only label/count component scales and assumed pure-DP EM stage conversion are calibrated. No rho_total bound for the preserved complete DER flow.',
        limitations=['exact root count and true-data root decisions retained','candidate-score indexing and density-counting defects retained','original splitting sensitivity is not proved for the retained flawed score','original early-leaf budget recycling adds incompatible precision/epsilon parameters and is not a rho ledger','data-dependent combination overflow fallback retained','full-rectangle sampling followed by original upper-triangle projection retained'],
        count_component_note='At non-root depths the full symmetric matrix regional-count vector has L2 sensitivity<=2; original geometric precision weights and terminal root transfer are retained. No unnoised root query is included in the Gaussian certificate.')
    return p

def require_end_to_end_certificate(method):
    raise RuntimeError(method+': no full zCDP certificate is available with the user-required preserved implementation flow')
