"""TMMR-v2 free active basis. JR and downstream are frozen v1."""
import math,time,json,random,hashlib,itertools,heapq
from decimal import Decimal,localcontext,ROUND_FLOOR
from pathlib import Path
import numpy as np
from assignment import assignment
from reference import original_spectrum
class MethodBoundary(RuntimeError):
 def __init__(self,report):self.report=report;super().__init__(report['reason'])
def c2(k):return k*(k-1)//2
def moments(x):return tuple(sum((v*[1,k,c2(k)][j] for k,v in enumerate(x)),Decimal(0)) for j in range(3))
def integerize(p,logp,n,M,seed,output,progress):
 start=time.perf_counter();K=len(p);allowed=np.isfinite(logp)
 triples=[a for a in range(K-2) if all(allowed[a:a+3])]
 with localcontext() as ctx:
  ctx.prec=100
  x=[Decimal.from_float(float(v))*n for v in p];x0=x.copy();initial=moments(x);wp=initial[2]
  frac={k for k,v in enumerate(x) if v!=v.to_integral_value(rounding=ROUND_FLOOR)}
  initial_q=len(frac);rng=random.Random(int.from_bytes(hashlib.sha256(('TMMR-freebasis-v2:'+str(seed)).encode()).digest(),'big'))
  order=sorted(frac);prev={k:(order[i-1] if i else None) for i,k in enumerate(order)};nxt={k:(order[i+1] if i+1<len(order) else None) for i,k in enumerate(order)}
  heap=[];steps=0;max_defect=Decimal(0)
  def quartet(k):
   ids=[]
   for _ in range(4):
    if k is None or k not in frac:return None
    ids.append(k);k=nxt[k]
   return tuple(ids)
  def push(k):
   ids=quartet(k)
   if ids:heapq.heappush(heap,(ids[-1]-ids[0],ids))
  for k in order:push(k)
  with (output/'TMMR_flight.jsonl').open('x',encoding='utf8') as log:
   while len(frac)>3:
    while heap:
     _,ids=heapq.heappop(heap)
     if quartet(ids[0])==ids:break
    else:raise ArithmeticError('Active-quartet heap empty')
    def det(t):
     a,b,c=t;return (b-a)*(c-a)*(c-b)//2
    d=[(-1)**i*det(ids[:i]+ids[i+1:]) for i in range(4)];g=math.gcd(*d);d=[v//g for v in d]
    assert sum(d)==sum(k*v for k,v in zip(ids,d))==sum(c2(k)*v for k,v in zip(ids,d))==0
    plus=[];minus=[]
    for k,v in zip(ids,d):
     low=x[k].to_integral_value(rounding=ROUND_FLOOR);high=low+1
     plus.append(((high-x[k])/v if v>0 else (x[k]-low)/(-v),k,high if v>0 else low))
     minus.append(((x[k]-low)/v if v>0 else (high-x[k])/(-v),k,low if v>0 else high))
    tp,kp,bp=min(plus);tm,km,bm=min(minus);assert tp>0 and tm>0
    positive=Decimal(rng.getrandbits(256))/Decimal(1<<256)<tm/(tp+tm)
    step=tp if positive else -tm;k=kp if positive else km;desired=bp if positive else bm
    for idx,v in zip(ids,d):x[idx]+=step*v
    max_defect=max(max_defect,abs(x[k]-desired));x[k]=desired;old_count=len(frac)
    for idx in ids:
     if x[idx]==x[idx].to_integral_value(rounding=ROUND_FLOOR) and idx in frac:
      p0,q0=prev[idx],nxt[idx];frac.remove(idx)
      if p0 is not None:nxt[p0]=q0
      if q0 is not None:prev[q0]=p0
      walk=p0
      for _ in range(3):
       if walk is None:break
       push(walk);walk=prev[walk]
    steps+=1;assert len(frac)<old_count and steps<=max(0,initial_q-3)
    log.write(json.dumps(dict(step=steps,ids=ids,direction=d,remaining_fractional=len(frac),positive=positive))+'\n')
    if steps%128==0:log.flush();progress(dict(stage='FREE_BASIS_FLIGHT',steps=steps,remaining_fractional=len(frac),seconds=time.perf_counter()-start))
  final=moments(x);drift=[final[i]-initial[i] for i in range(3)]
  assert all(abs(v)<=Decimal('1e-65')*(1+abs(initial[i])) for i,v in enumerate(drift))
  floorhist=[int(v.to_integral_value(rounding=ROUND_FLOOR)) for v in x]
  totals=[sum(floorhist),sum(k*v for k,v in enumerate(floorhist)),sum(c2(k)*v for k,v in enumerate(floorhist))]
  loss0=sum((Decimal(v)-x0[k])**2 for k,v in enumerate(floorhist));best=None;count=0;feasible=0
  for a in triples:
   ids=(a,a+1,a+2);outside=sorted(frac-set(ids))
   for bits in itertools.product((0,1),repeat=len(outside)):
    count+=1
    N=n-totals[0]+sum(floorhist[k] for k in ids)-sum(bits)
    mass=M-totals[1]+sum(k*floorhist[k] for k in ids)-sum(k*b for k,b in zip(outside,bits));q=mass-a*N
    if N<0 or q<0 or q>2*N:continue
    low=max(0,q-N);high=q//2
    if low>high:continue
    base=totals[2]-sum(c2(k)*floorhist[k] for k in ids)+sum(c2(k)*b for k,b in zip(outside,bits))+c2(a)*N+a*q
    wf=int((wp-base).to_integral_value(rounding=ROUND_FLOOR))
    for t in {max(low,min(high,wf)),max(low,min(high,wf+1))}:
     counts=(N-q+t,q-2*t,t);W=base+t;feasible+=1
     loss=loss0+sum((Decimal(c)-x0[k])**2-(Decimal(floorhist[k])-x0[k])**2 for k,c in zip(ids,counts))+sum((Decimal(floorhist[k]+b)-x0[k])**2-(Decimal(floorhist[k])-x0[k])**2 for k,b in zip(outside,bits))
     key=(abs(Decimal(W)-wp),loss,a,bits,W)
     if best is None or key<best[0]:best=(key,a,outside,bits,counts,W)
  if best is None:raise MethodBoundary(dict(status='TMMR2_LANDING_BOUNDARY',reason='No nonnegative adjacent-lattice landing in frozen finite class',global_infeasibility_claim=False,steps=steps,remaining_fractional=len(frac),initial_fractional=initial_q,seconds=time.perf_counter()-start))
  key,a,outside,bits,counts,W=best;h=np.asarray(floorhist,dtype=np.int64)
  for k,b in zip(outside,bits):h[k]+=b
  h[a:a+3]=counts
  assert np.all(h>=0) and not np.any(h[~allowed])
  assert sum(map(int,h))==n and sum(k*int(v) for k,v in enumerate(h))==M
  cert=dict(status='LANDED',N=n,M=M,W_p=float(wp),W_p_decimal=str(wp),W_landing=W,W_error=float(Decimal(W)-wp),basis=a,basis_counts=list(counts),residual_bins=outside,residual_ceil_bits=list(bits),steps=steps,initial_fractional=initial_q,remaining_fractional=len(frac),pivots=0,landing_candidates=count,feasible_landing_candidates=feasible,squared_histogram_distortion=str(key[1]),flight_moment_drift=list(map(str,drift)),max_decimal_boundary_roundoff=str(max_defect),finite_log_underflow_bins=int(np.sum(allowed&(p==0))),unbiasedness='Ideal flight only. Decimal100 and 256-bit branch implement numerically. Deterministic landing and graphical balancing not unbiased.',optimality_scope='Enumerated adjacent-triple and residual floor-ceil class only',seconds=time.perf_counter()-start)
  return h,cert


def degree_target(degree,edge_count,release,recovered,released,log_probability,degree_wedge_state,seed,output,progress):
 began=time.perf_counter();n=len(degree);M=2*int(edge_count);out=Path(output)
 oldM=2*int(round(float(release.checkpoint.continuous_noisy_edge_mass)))
 align_start=time.perf_counter()
 if M!=oldM:released,log_probability=original_spectrum(release.checkpoint,True,endpoint_mass=M)
 align_seconds=time.perf_counter()-align_start
 p=np.asarray(released.probability,np.float64);lp=np.asarray(log_probability,np.float64)
 assert len(p)==len(lp)<=n and np.all(np.isfinite(p)) and np.all(p>=0)
 np.save(out/'TMMR_p.npy',p);np.save(out/'TMMR_logp.npy',lp)
 progress(dict(stage='TMMR_FLIGHT',n=n,M=M,mean_alignment_required=M!=oldM,mean_alignment_seconds=align_seconds))
 h,cert=integerize(p,lp,n,M,seed,out,progress);np.save(out/'TMMR_histogram.npy',h)
 import target_histogram
 graph_start=time.perf_counter();gh,gr=target_histogram._graphicalize_by_balancing(h.copy());graph_seconds=time.perf_counter()-graph_start
 gh=np.asarray(gh,np.int64);np.save(out/'TMMR_graphical_histogram.npy',gh)
 assert sum(map(int,gh))==n and sum(k*int(v) for k,v in enumerate(gh))==M
 Wg=sum(c2(k)*int(v) for k,v in enumerate(gh));multiset=np.repeat(np.arange(len(gh),dtype=np.int64),gh)
 target,ass=assignment(multiset,degree,recovered)
 outside_support=[k for k,v in enumerate(gh) if v and (k>=len(lp) or not np.isfinite(lp[k]))]
 cert.update(status='TARGET_VALIDATED',release_endpoint_mass=oldM,final_endpoint_mass=M,mean_alignment_required=M!=oldM,mean_alignment_seconds=align_seconds,spectrum_diagnostics=released.diagnostics,W_graphical=Wg,W_graphical_drift=Wg-cert['W_landing'],graphicality_repair=gr,graphicality_seconds=graph_seconds,graphical_balancing_outside_original_support=outside_support,original_ID_assignment_preserved=True,original_integerization_executed=False,assignment=ass,total_target_wall_seconds=time.perf_counter()-began)
 (out/'TMMR_certificate.json').write_text(json.dumps(cert,indent=2,allow_nan=False),encoding='utf-8')
 return target,cert
