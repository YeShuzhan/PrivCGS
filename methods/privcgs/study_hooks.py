"""The four separately selected PrivCGS ablations and independent alpha/zeta.

All inputs here are frozen private observations or their postprocessing. This
module never reads an original graph and never changes the legal swap engine.
"""
from __future__ import annotations
from dataclasses import replace
from decimal import Decimal, localcontext, ROUND_FLOOR
from fractions import Fraction
from math import gcd, isfinite
from pathlib import Path
import hashlib, heapq, itertools, json, random, time
import numpy as np


def ablation_a_observation_reuse(release):
    return release,dict(
        intervention='disable nonconstant interaction law; keep common observations and scalar temperature calibration',
        degree_observation_unchanged=True,edge_mass_observation_unchanged=True,
        constant_row_unchanged=True,shared_edge_count_equal=True,degree_projection_fusion_equal=True,
        scalar_temperature_calibration_unchanged=True,new_private_queries=0)


def recover_without_nonconstant_interactions(checkpoint,seed):
    """Degree-only node law; the full method's scalar temperature is a control.

    Removing the observations themselves would also zero the posterior-variance
    temperature. The intervention is on the nonconstant interaction mechanism,
    with the shared observations and the temperature calibration left intact.
    """
    import route_s_clean.p2_recovery as p2
    started=time.perf_counter()
    coherent=p2._recover_coherent_gaussian_base(checkpoint)
    shell=p2._compile_shell_posterior_state(coherent.raw_operator,float(checkpoint.operator_noise_std))
    temperature,temperature_diagnostics=p2.derive_source_temperature(checkpoint,coherent.target_edges,coherent.endpoint_target,shell)
    recovered,operator_diagnostics=p2._recover_operator_shell_state(checkpoint,coherent,shell)
    # Keep the shared degree projection, but no nonconstant eigenfeatures.
    recovered=replace(recovered,spectrum=np.zeros_like(recovered.spectrum),
        node_factors=np.zeros_like(recovered.node_factors),active_rank=1,
        operator=coherent.coherent_operator.copy())
    base,base_diagnostics=p2._freeze_base_graph_law(checkpoint,recovered,coherent.endpoint_target,temperature)
    base=replace(base,component_shift=np.zeros_like(base.component_shift),
        component_fiber=np.zeros_like(base.component_fiber),parent_flux=np.zeros_like(base.parent_flux))
    digest=hashlib.sha256()
    for name in ('sociability','star','interaction','interaction_diagonal','parent_fiber','component_fiber','marked_shift','component_shift','parent_flux'):
        digest.update(np.asarray(getattr(base,name)).tobytes())
    base=replace(base,state_hash=digest.hexdigest())
    law,functional_diagnostics=p2._freeze_commutant_functional(base,checkpoint,recovered,int(seed))
    carrier,carrier_diagnostics=p2.recover_rank_coupled_degree_carrier(checkpoint,coherent.endpoint_target,law.sociability,law.target_edges)
    carrier=np.asarray(carrier,dtype=np.float64)
    digest=hashlib.sha256(law.state_hash.encode('ascii')+np.ascontiguousarray(carrier).tobytes()).hexdigest()
    law=replace(law,target_endpoint_mass=carrier,noisy_degree=carrier,state_hash=digest)
    assert not np.any(law.interaction) and not np.any(law.interaction_diagonal) and not np.any(law.parent_flux) and not np.any(law.component_shift)
    return p2.RecoveredCheckpointState(checkpoint_path=Path('<in-memory-private-transcript>'),checkpoint=checkpoint,
        temperature=temperature,temperature_diagnostics=dict(temperature_diagnostics),graph_law=law,
        endpoint_target=carrier,degree_carrier_target=carrier,
        recovery_diagnostics={**dict(base_diagnostics),**dict(functional_diagnostics),**dict(carrier_diagnostics),
            'nonconstant_interaction_disabled':True,'relation_quotient_flux_disabled':True,'relation_component_offsets_disabled':True,
            'shared_mass_and_degree_projection_preserved':True,'scalar_temperature_calibration_unchanged':True,
            'private_graph_access_after_release':False,'checkpoint_serialization_count':0},
        public_experiment_id=int(seed),stage_seconds={'checkpoint_load_seconds':0.,'p2_recovery_seconds':time.perf_counter()-started},
        audit={'checkpoint_load_requests':0,'checkpoint_load_materializations':0,'p2_recovery_requests':1,'p2_recovery_materializations':1})


def direct_degree_assignment(multiset,carrier_degree,recovered):
    nodes=np.arange(len(multiset),dtype=np.int64)
    recovered_degree=np.asarray(recovered.endpoint_target,dtype=np.float64)
    assert len(recovered_degree)==len(nodes) and np.all(np.isfinite(recovered_degree))
    order=np.lexsort((nodes,recovered_degree))
    target=np.empty_like(multiset);target[order]=multiset
    return np.ascontiguousarray(target),dict(
        method='ascending recovered d_hat; stable node-ID ties',
        recovered_degree_sha256=hashlib.sha256(recovered_degree.tobytes()).hexdigest(),
        Q_guidance_in_HH_equal_degree_candidates_retained=True,
        shared_edge_count_and_integer_multiset_unchanged=True)


def degree_only_spectral_score(law,path,tape,source_degree,target_endpoint_mass,degree_noise_std,
        *,feature_cache=None,precompiled_spectral_columns=None,_calibration_only=False,_retain_spectral_columns=True):
    """Ablation a removes the relation score rather than normalizing a zero score.

    Its remaining natural parameter is exactly the original Gaussian degree
    energy (coefficient one). The unit denominator never scales a nonzero
    relation statistic; this is an explicit mechanism deletion, not clamping.
    """
    import route_s_clean.chain_moments as chain
    spectral=precompiled_spectral_columns
    if spectral is None:spectral=chain.compile_spectral_columns(law,tape,feature_cache=feature_cache)
    assert not np.any(spectral.negative_delta) and not np.any(spectral.positive_delta) and not np.any(spectral.additive_gain)
    scored=replace(tape,negative_energy=np.zeros_like(tape.negative_energy),
        positive_energy=np.zeros_like(tape.positive_energy),structural_gain=np.zeros_like(tape.structural_gain))
    messages=None if _calibration_only else chain.build_gaussian_messages(path,scored,source_degree,target_endpoint_mass,degree_noise_std,
        quadratic_coefficient=1.0,structural_temperature=1.0)
    return chain.EfficientSpectralScore(messages=messages,quadratic_coefficient=1.0,residual_scale=1.0,
        projection_coefficient=0.0,spectral_columns=spectral if _retain_spectral_columns else None,
        scored_tape=None if _calibration_only else scored,moments=None,
        diagnostics={**dict(spectral.statistics),'nonconstant_relation_score_disabled':True,
            'residual_variance':0.,'residual_scale':1.,'projection_coefficient':0.,'signed_degree_energy_coefficient':1.,
            'fisher_quotient_zero_score_branch':False,'degree_gaussian_energy_retained':True,
            'unit_denominator_for_identically_zero_relation_statistic':True,
            'base_log_normalizer':None if messages is None else float(messages.log_normalizer),
            'tilted_log_normalizer':None if messages is None else float(messages.log_normalizer),
            'log_normalizer_change':0.,'new_fitted_parameter_count':0,'dense_covariance_materialized':False})


def integerize_two_moments(p,logp,n,M,seed,output,progress):
    """Same free-basis design with only N and M; no wedge constraint/objective."""
    import tmmr
    started=time.perf_counter();K=len(p);allowed=np.isfinite(logp)
    pairs=[a for a in range(K-1) if all(allowed[a:a+2])]
    with localcontext() as context:
        context.prec=100
        x=[Decimal.from_float(float(v))*n for v in p];x0=x.copy()
        moment=lambda j:sum((v*(1 if j==0 else k) for k,v in enumerate(x)),Decimal(0))
        initial=[moment(0),moment(1)]
        fractional={k for k,v in enumerate(x) if v!=v.to_integral_value(rounding=ROUND_FLOOR)}
        initial_count=len(fractional)
        rng=random.Random(int.from_bytes(hashlib.sha256(('TMMR-freebasis-v2:'+str(seed)).encode()).digest(),'big'))
        order=sorted(fractional)
        previous={k:(order[i-1] if i else None) for i,k in enumerate(order)}
        following={k:(order[i+1] if i+1<len(order) else None) for i,k in enumerate(order)}
        heap=[];steps=0;roundoff=Decimal(0)
        def triple(k):
            ids=[]
            for _ in range(3):
                if k is None or k not in fractional:return None
                ids.append(k);k=following[k]
            return tuple(ids)
        def push(k):
            ids=triple(k)
            if ids:heapq.heappush(heap,(ids[-1]-ids[0],ids))
        for k in order:push(k)
        with (Path(output)/'TMMR_flight.jsonl').open('x') as log:
            while len(fractional)>2:
                while heap:
                    _,ids=heapq.heappop(heap)
                    if triple(ids[0])==ids:break
                else:raise ArithmeticError('two-moment active-triple heap empty')
                a,b,c=ids;direction=[b-c,c-a,a-b]
                divisor=gcd(*direction);direction=[v//divisor for v in direction]
                assert sum(direction)==sum(k*v for k,v in zip(ids,direction))==0
                plus=[];minus=[]
                for k,v in zip(ids,direction):
                    low=x[k].to_integral_value(rounding=ROUND_FLOOR);high=low+1
                    plus.append(((high-x[k])/v if v>0 else (x[k]-low)/(-v),k,high if v>0 else low))
                    minus.append(((x[k]-low)/v if v>0 else (high-x[k])/(-v),k,low if v>0 else high))
                tp,kp,bp=min(plus);tm,km,bm=min(minus);assert tp>0 and tm>0
                positive=Decimal(rng.getrandbits(256))/Decimal(1<<256)<tm/(tp+tm)
                step=tp if positive else -tm;chosen=kp if positive else km;desired=bp if positive else bm
                for k,v in zip(ids,direction):x[k]+=step*v
                roundoff=max(roundoff,abs(x[chosen]-desired));x[chosen]=desired;old_count=len(fractional)
                for k in ids:
                    if k in fractional and x[k]==x[k].to_integral_value(rounding=ROUND_FLOOR):
                        before,after=previous[k],following[k];fractional.remove(k)
                        if before is not None:following[before]=after
                        if after is not None:previous[after]=before
                        walk=before
                        for _ in range(2):
                            if walk is None:break
                            push(walk);walk=previous[walk]
                steps+=1;assert len(fractional)<old_count and steps<=max(0,initial_count-2)
                log.write(json.dumps(dict(step=steps,ids=ids,direction=direction,remaining_fractional=len(fractional),positive=positive))+'\n')
                if steps%128==0:
                    log.flush();progress(dict(stage='TWO_MOMENT_FLIGHT',steps=steps,remaining_fractional=len(fractional)))
        drift=[moment(j)-initial[j] for j in range(2)]
        assert all(abs(v)<=Decimal('1e-65')*(1+abs(initial[j])) for j,v in enumerate(drift))
        floor_hist=[int(v.to_integral_value(rounding=ROUND_FLOOR)) for v in x]
        N0=sum(floor_hist);M0=sum(k*v for k,v in enumerate(floor_hist))
        loss0=sum((Decimal(v)-x0[k])**2 for k,v in enumerate(floor_hist))
        best=None;candidate_count=0;feasible=0
        if not fractional and N0==n and M0==M:
            best=((loss0,-1,()),None,[],(),(),None)
        for a in pairs:
            ids=(a,a+1);outside=sorted(fractional-set(ids))
            for bits in itertools.product((0,1),repeat=len(outside)):
                candidate_count+=1
                N=n-N0+sum(floor_hist[k] for k in ids)-sum(bits)
                mass=M-M0+sum(k*floor_hist[k] for k in ids)-sum(k*b for k,b in zip(outside,bits))
                q=mass-a*N
                if N<0 or not 0<=q<=N:continue
                counts=(N-q,q);feasible+=1
                loss=loss0+sum((Decimal(c)-x0[k])**2-(Decimal(floor_hist[k])-x0[k])**2 for k,c in zip(ids,counts))+sum((Decimal(floor_hist[k]+b)-x0[k])**2-(Decimal(floor_hist[k])-x0[k])**2 for k,b in zip(outside,bits))
                key=(loss,a,bits)
                if best is None or key<best[0]:best=(key,a,outside,bits,counts,None)
        if best is None:
            raise tmmr.MethodBoundary(dict(status='TWO_MOMENT_LANDING_BOUNDARY',reason='No nonnegative adjacent-pair landing in the finite class',global_infeasibility_claim=False,steps=steps,seconds=time.perf_counter()-started))
        key,a,outside,bits,counts,_=best;histogram=np.asarray(floor_hist,dtype=np.int64)
        for k,b in zip(outside,bits):histogram[k]+=b
        if a is not None:histogram[a:a+2]=counts
        assert np.all(histogram>=0) and not np.any(histogram[~allowed])
        assert sum(map(int,histogram))==n and sum(k*int(v) for k,v in enumerate(histogram))==M
        # W is computed only after the landing decision, for diagnostics.
        W=sum(k*(k-1)//2*int(v) for k,v in enumerate(histogram))
        Wp=sum(Decimal(k*(k-1)//2)*v for k,v in enumerate(x0))
        certificate=dict(status='LANDED',N=n,M=M,W_p=float(Wp),W_p_decimal=str(Wp),W_landing=W,W_error=float(Decimal(W)-Wp),
            constrained_moments=['node_count','total_degree'],wedge_constraint=False,wedge_objective=False,
            basis=a,basis_counts=list(counts),residual_bins=outside,residual_ceil_bits=list(bits),steps=steps,initial_fractional=initial_count,
            remaining_fractional=len(fractional),pivots=0,landing_candidates=candidate_count,feasible_landing_candidates=feasible,
            squared_histogram_distortion=str(key[0]),flight_moment_drift=list(map(str,drift)),max_decimal_boundary_roundoff=str(roundoff),
            finite_log_underflow_bins=int(np.sum(allowed&(p==0))),seconds=time.perf_counter()-started,
            optimality_scope='Adjacent-pair and residual floor-ceil class; histogram loss only',
            unbiasedness='Ideal flight only; deterministic landing and unchanged graphical repair are not unbiased')
        return histogram,certificate


def make_triangle_policy(A,B,W,lo,hi,zeta):
    from joint_risk import RiskBand
    class TriangleOnlyRiskBand(RiskBand):
        def __init__(self):
            self.A,self.B=int(A),int(B);self.W,self.lo,self.hi=int(W),int(lo),int(hi)
            self.lower,self.upper,self.alpha=float('nan'),float('nan'),float(zeta)
            left,right=Fraction(1,self.A),Fraction(1,self.B)
            self.den=left.denominator*right.denominator//gcd(left.denominator,right.denominator)
            self.a=left.numerator*(self.den//left.denominator);self.b=right.numerator*(self.den//right.denominator)
            center=Fraction(2*self.den,self.a+self.b);floor=center.numerator//center.denominator
            candidates={max(self.lo,min(self.hi,t)) for t in (floor,floor+1)}
            self.target=min(candidates,key=lambda t:(self.numerator(t),t));self.opt_num=self.numerator(self.target)
            self.threshold=(1+Fraction(str(float(zeta))))*self.opt_num
            low=(self.den-self.threshold)/self.b;high=(self.den+self.threshold)/self.a
            self.band_lo=max(self.lo,-(-low.numerator//low.denominator));self.band_hi=min(self.hi,high.numerator//high.denominator)
            assert self.band_lo<=self.target<=self.band_hi
            self.candidate_scans=self.valid_candidates=self.accepted_moves=0
            self.rejected_hard_interval=self.rejected_risk=self.cross_center_moves=0
            self.last_triangle=None;self.audit_callback=None
    return TriangleOnlyRiskBand()


def triangle_only_refine(dataset,edges,adjacency,target,current,W,lo,hi,contract,repair,zeta):
    from joint_risk import triangle_interval
    started=time.perf_counter();start=int(current);sweeps=[]
    A,B,certificate=triangle_interval(contract)
    if not (W>0 and A is not None and 0<A<=B):
        return current,dict(status='TRIANGLE_RISK_UNDEFINED',triangle_certificate=certificate,
            T_start=start,T_final=start,closure_interval_hit=lo<=current<=hi,cstar_risk_band_hit=False,cstar_exact_hit=False,
            accepted_moves=0,candidate_scans=0,local_stall=False,sweeps_completed=0,sweeps_started=0,seconds=time.perf_counter()-started),sweeps
    physical_lo,physical_hi=0,W//3
    policy=make_triangle_policy(A,B,W,physical_lo,physical_hi,zeta)
    while not policy.hit(current):
        old=current;before=policy.candidate_scans
        method=repair.increase_sweep if current<policy.band_lo else repair.decrease_sweep
        current,log=method(dataset,edges,adjacency,target,current,physical_lo,physical_hi,len(sweeps),risk_policy=policy)
        assert repair.count_triangles(edges,adjacency)==current and physical_lo<=current<=physical_hi
        log.update(complete_sweep=not policy.hit(current),candidate_scans=policy.candidate_scans-before,
                   r_before=policy.risk(old),r_after=policy.risk(current),risk_band_hit=policy.hit(current))
        sweeps.append(log)
        if not log['accepted']:break
        assert policy.numerator(current)<policy.numerator(old)
    status='TRIANGLE_RISK_BAND_HIT' if policy.hit(current) else 'TRIANGLE_LOCAL_STALL'
    report=policy.summary(start,current,status,sweeps,time.perf_counter()-started)
    report.update(objective='triangle relative minimax risk only',closure_interval_hit=lo<=current<=hi,
                  triangle_certificate=certificate,optimization_zeta=zeta,coverage_alpha=contract['alpha_per_cell'])
    report.pop('alpha',None)
    return current,report,sweeps


def install(config):
    import tmmr,joint_risk
    variant=config['variant']
    if variant=='a':
        import sys
        import route_s_clean.chain_moments as chain
        import route_s_clean.hamiltonian_completion
        import route_s_clean.endpoint_laws
        import route_s_clean.pipeline
        # The original numerical audit wrapper expects this named global.
        globals()['fused_one_channel_moments']=chain.fused_one_channel_moments
        original=chain.efficient_spectral_score_messages
        for module in tuple(sys.modules.values()):
            if module is None:continue
            for key,value in tuple(vars(module).items()):
                if value is original:setattr(module,key,degree_only_spectral_score)
    elif variant=='b':tmmr.assignment=direct_degree_assignment
    elif variant=='c':tmmr.integerize=integerize_two_moments
    assert variant in ('full','a','b','c','d','e','f','g')
    original_refine=joint_risk.refine
    def refine(dataset,edges,adjacency,target,current,W,lo,hi,contract,repair,audit_callback=None):
        if variant=='d':return triangle_only_refine(dataset,edges,adjacency,target,current,W,lo,hi,contract,repair,config['zeta'])
        local=dict(contract,alpha_per_cell=config['zeta'])
        result,report,sweeps=original_refine(dataset,edges,adjacency,target,current,W,lo,hi,local,repair,audit_callback)
        if 'JR_triangle_certificate' in local:contract['JR_triangle_certificate']=local['JR_triangle_certificate']
        report.update(coverage_alpha=config['alpha'],optimization_zeta=config['zeta'])
        report.pop('alpha',None)
        return result,report,sweeps
    joint_risk.refine=refine
