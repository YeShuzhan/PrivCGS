"""Cstar-RB2: exact risk comparisons on the frozen binary64 closure interval.

No private inputs, new queries, random draws, or search limits. The confidence
alpha is reused as an explicitly chosen optimization approximation factor.
"""
from fractions import Fraction
from math import isfinite, gcd
import time


def triangle_interval(contract):
    """Same published confidence box; triangle interval is soft risk only."""
    if contract['status']!='identified_interval':
        return None,None,dict(status='UNDEFINED',reason='Original physical confidence box not identified')
    F=lambda x:Fraction.from_float(float(x))
    gL,gU=map(F,contract['tightened_original_wedge_interval'])
    hL,hU=map(F,contract['projected_wedge_interval'])
    jL,jU=map(F,contract['projected_incidence_interval'])
    BT=int(contract['projection_loss_bounds']['destroyed_triangles_upper'])
    lower=max(Fraction(0),jL/3)
    upper=min(jU/3+BT,jU/3+(gU-hL)/2,gU/3)
    A=-(-lower.numerator//lower.denominator);B=upper.numerator//upper.denominator
    return A,B,dict(status='POSITIVE_INTEGER_INTERVAL' if 0<A<=B else 'UNDEFINED',integer_interval=[A,B],lower_rational=str(lower),upper_rational=str(upper),same_confidence_events=True,new_query=False,triangle_loss_upper=BT,role='soft joint risk; original closure remains sole hard band')

class RiskBand:
    def __init__(self, lower, upper, W, lo, hi, alpha, A, B):
        if not (isfinite(lower) and isfinite(upper) and 0 < lower <= upper
                and W > 0 and 0 <= lo <= hi and isfinite(alpha) and alpha >= 0):
            raise ValueError('undefined Cstar relative-risk domain')
        assert 0<A<=B
        self.A,self.B=int(A),int(B)
        self.W, self.lo, self.hi = int(W), int(lo), int(hi)
        self.lower, self.upper, self.alpha = float(lower), float(upper), float(alpha)
        # Exact rational values of the already frozen floating point endpoints.
        left = max(Fraction(1,self.A),Fraction(3, self.W) / Fraction.from_float(float(lower)))
        right = min(Fraction(1,self.B),Fraction(3, self.W) / Fraction.from_float(float(upper)))
        self.den = left.denominator * right.denominator // gcd(left.denominator, right.denominator)
        self.a = left.numerator * (self.den // left.denominator)
        self.b = right.numerator * (self.den // right.denominator)
        center = Fraction(2 * self.den, self.a + self.b)
        floor = center.numerator // center.denominator
        candidates = {max(self.lo, min(self.hi, t)) for t in (floor, floor + 1)}
        self.target = min(candidates, key=lambda t: (self.numerator(t), t))
        self.opt_num = self.numerator(self.target)
        self.threshold = (1 + Fraction(str(float(alpha)))) * self.opt_num
        low = (self.den - self.threshold) / self.b
        high = (self.den + self.threshold) / self.a
        self.band_lo = max(self.lo, -(-low.numerator // low.denominator))
        self.band_hi = min(self.hi, high.numerator // high.denominator)
        assert self.band_lo <= self.target <= self.band_hi
        assert self.hit(self.band_lo) and self.hit(self.band_hi)
        if self.band_lo > self.lo: assert not self.hit(self.band_lo - 1)
        if self.band_hi < self.hi: assert not self.hit(self.band_hi + 1)
        self.candidate_scans = self.valid_candidates = self.accepted_moves = 0
        self.rejected_hard_interval = self.rejected_risk = self.cross_center_moves = 0
        self.last_triangle = None
        self.audit_callback = None

    def numerator(self, t):
        t = int(t)
        return max(self.a*t-self.den, self.den-self.b*t)

    def risk(self, t):
        return float(Fraction(self.numerator(t), self.den))

    def hit(self, t):
        return self.lo <= t <= self.hi and self.numerator(t) <= self.threshold

    def allow(self, current, new):
        self.valid_candidates += 1
        if not self.lo <= new <= self.hi:
            self.rejected_hard_interval += 1
            return False
        if self.numerator(new) >= self.numerator(current):
            self.rejected_risk += 1
            return False
        return True

    def committed(self, old, new):
        assert self.lo <= new <= self.hi
        assert self.numerator(new) < self.numerator(old)
        self.accepted_moves += 1
        self.cross_center_moves += int((old-self.target)*(new-self.target) < 0)
        self.last_triangle = int(new)
        if self.audit_callback is not None: self.audit_callback(old, new)

    def summary(self, start, final, status, sweeps, seconds):
        ratio = float(Fraction(self.numerator(final), self.opt_num)) if self.opt_num else (1.0 if self.numerator(final)==0 else None)
        return dict(status=status, objective="joint triangle/closure relative minimax risk", triangle_interval=[self.A,self.B], alpha=self.alpha, approximation_factor=1+self.alpha,
            hard_triangle_interval=[self.lo,self.hi], T_star=self.target,
            r_star=float(Fraction(self.opt_num,self.den)), risk_band=[self.band_lo,self.band_hi],
            T_start=int(start),T_final=int(final),r_start=self.risk(start),r_final=self.risk(final),
            risk_ratio=ratio, risk_ratio_zero_optimum_convention='1 for 0/0; null for positive/0',
            closure_interval_hit=self.lo<=final<=self.hi,cstar_risk_band_hit=self.hit(final),
            cstar_exact_hit=final==self.target,local_stall=status=='JR_LOCAL_STALL',
            accepted_moves=self.accepted_moves,candidate_scans=self.candidate_scans,
            valid_candidates=self.valid_candidates,rejected_hard_interval=self.rejected_hard_interval,
            rejected_risk=self.rejected_risk,cross_center_moves=self.cross_center_moves,
            sweeps_completed=sum(bool(x['complete_sweep']) for x in sweeps),sweeps_started=len(sweeps),
            seconds=seconds,risk_comparison='exact integer numerators; binary64 interval endpoints',
            risk_certificate=dict(denominator=str(self.den),left_slope=str(self.a),right_slope=str(self.b),
                optimum_numerator=str(self.opt_num),threshold_numerator=str(self.threshold.numerator),
                threshold_denominator=str(self.threshold.denominator)),
            candidate_family='Frozen W4 donor ranking, slots, pairing, pivot order and sweep seeds. Direction chooses proposal family only; acceptance uses risk, never point overshoot.',
            local_stall_scope='No improving move found by this completed frozen candidate sweep; not global infeasibility')


def refine(dataset, edges, adjacency, target_degree, current, W, lo, hi, contract, repair,
           audit_callback=None):
    began=time.perf_counter();start=int(current);sweeps=[]
    L,U=contract['original_C_interval']
    A,B,tc=triangle_interval(contract)
    contract['JR_triangle_certificate']=tc
    undefined = not (isfinite(L) and isfinite(U) and 0<L<=U and W>0 and lo<=hi and A is not None and 0<A<=B)
    if not lo<=current<=hi or undefined:
        status='JR_CLOSURE_DOMAIN_FAILURE' if not lo<=current<=hi else 'JR_UNDEFINED'
        return current,dict(status=status,closure_interval_hit=lo<=current<=hi,
            cstar_risk_band_hit=False,cstar_exact_hit=False,local_stall=False,
            triangle_certificate=tc,T_start=start,T_final=start,accepted_moves=0,candidate_scans=0,
            sweeps_completed=0,sweeps_started=0,seconds=time.perf_counter()-began),sweeps
    policy=RiskBand(L,U,W,lo,hi,contract['alpha_per_cell'],A,B)
    policy.audit_callback=audit_callback
    while not policy.hit(current):
        old=current;before=policy.candidate_scans
        # Outside the band, select the original increase/decrease proposal family.
        # Signed candidate deltas are then judged solely by exact risk descent.
        method=repair.increase_sweep if current<policy.band_lo else repair.decrease_sweep
        current,log=method(dataset,edges,adjacency,target_degree,current,lo,hi,len(sweeps),risk_policy=policy)
        assert repair.count_triangles(edges,adjacency)==current and lo<=current<=hi
        log.update(complete_sweep=not policy.hit(current),candidate_scans=policy.candidate_scans-before,
                   r_before=policy.risk(old),r_after=policy.risk(current),risk_band_hit=policy.hit(current))
        sweeps.append(log)
        repair.write_json(repair.OUT/'Cstar_RB2_progress.json',policy.summary(start,current,
            'JR_RISK_BAND_HIT' if policy.hit(current) else 'RUNNING',sweeps,time.perf_counter()-began))
        if not log['accepted']:break
        assert policy.numerator(current)<policy.numerator(old)
    status='JR_RISK_BAND_HIT' if policy.hit(current) else 'JR_LOCAL_STALL'
    report=policy.summary(start,current,status,sweeps,time.perf_counter()-began)
    from cstar_rb2 import RiskBand as OriginalRisk
    original=OriginalRisk(L,U,W,lo,hi,contract['alpha_per_cell'])
    report.update(triangle_certificate=tc,original_closure_risk_start=original.risk(start),original_closure_risk_final=original.risk(current),original_closure_risk_band_hit=original.hit(current))
    return current,report,sweeps
