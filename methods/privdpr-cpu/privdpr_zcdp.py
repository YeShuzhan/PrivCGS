"""Paper-parameter PrivDPR repair, with explicit edge-zCDP adaptation.

Only noised embedding updates and an accounted edge count enter generation.
The unnoised PageRank MLP remains inside each bounded private query and is never
released or used directly by reconstruction. This is not the original node-DP
mechanism, nor a bitwise TensorFlow reproduction.
"""
import math
import random
import secrets
import time
import numpy as np
import torch
from torch import nn
from privacy import Ledger, gaussian

PAPER=dict(epochs=5,batch_size=16,embedding_dim=128,hidden_dim=64,learning_rate=.001,
           walk_num=2,walk_len=16,gamma=.85,normalization_s=8.,preset_sensitivity=5.)

def depth(n):
    p=PAPER;steps=p['epochs']*(n//p['batch_size'])
    if not steps:raise ValueError('Paper batch size requires at least 16 public nodes')
    gamma=p['gamma']
    M=(2*(n-1)*gamma**2+2*gamma+2*gamma*(1-gamma)/n)*(1+1/gamma)
    pairs=p['batch_size']*p['walk_num']*p['walk_len']
    # Paper Eq13 specifies L hidden depth and L+1 normalized weight matrices.
    matrices=max(2,math.ceil(math.log(pairs*M*steps/p['preset_sensitivity'],p['normalization_s'])))
    return matrices,steps,M

class DeepPageRank(nn.Module):
    def __init__(self,n):
        super().__init__();h,t,m=depth(n)
        self.embedding=nn.Parameter(torch.empty(n,PAPER['embedding_dim']))
        widths=[PAPER['embedding_dim']]+[PAPER['hidden_dim']]*(h-1)+[1]
        self.weights=nn.ParameterList([nn.Parameter(torch.empty(a,b)) for a,b in zip(widths,widths[1:])])
        self.biases=nn.ParameterList([nn.Parameter(torch.empty(b)) for b in widths[1:]])
        nn.init.xavier_uniform_(self.embedding)
        for w,b in zip(self.weights,self.biases):
            nn.init.xavier_uniform_(w);nn.init.zeros_(b)
        self.weight_matrices=h;self.steps=t

    def forward(self,ids):
        value=self.embedding[ids]
        for w,b in zip(self.weights,self.biases):
            clean=torch.nan_to_num(w,nan=0.,posinf=0.,neginf=0.)
            norm=torch.linalg.matrix_norm(clean,ord=2).clamp_min(torch.finfo(w.dtype).tiny)
            value=torch.sigmoid(value@(clean/(norm*PAPER['normalization_s']))+torch.nan_to_num(b,nan=0.,posinf=0.,neginf=0.))
        return value.reshape(-1)

def pair_loss(model,heads,tails,degrees,n):
    if not len(heads):return model.embedding.sum()*0
    device=model.embedding.device
    hi=torch.as_tensor(heads,dtype=torch.long,device=device)
    ti=torch.as_tensor(tails,dtype=torch.long,device=device)
    dh=torch.as_tensor(degrees[heads],dtype=torch.float32,device=device)
    dt=torch.as_tensor(degrees[tails],dtype=torch.float32,device=device)
    fh=model(hi);ft=model(ti);g=PAPER['gamma']
    diff=fh/dh-ft/(g*dt)
    # Paper sums sampled pair losses; the public bound uses the maximum 512 pairs.
    return (dt*g*g*diff.square()+2*g*(1-g)/n*diff+(1-g)**2/(dt*n*n)).sum()

def sample_pairs(neighbors,batch,rng):
    heads=[];tails=[]
    for start in batch:
        for _ in range(PAPER['walk_num']):
            node=int(start)
            for __ in range(PAPER['walk_len']):
                choices=neighbors[node]
                if not len(choices):break
                node=int(choices[rng.randrange(len(choices))]);heads.append(start);tails.append(node)
    return np.asarray(heads,dtype=np.int64),np.asarray(tails,dtype=np.int64)

def clip_embedding_query(gradient,radius,shape):
    if gradient is None:return torch.zeros(shape,dtype=torch.float64)
    value=gradient.detach().cpu().double()
    if not bool(torch.isfinite(value).all()):return torch.zeros_like(value)
    norm=float(torch.linalg.vector_norm(value))
    return value*min(1.,radius*(1-1e-10)/max(norm,1e-300))

def count_transitions(embedding,counts,public_rng,block_rows):
    # argmax(softmax((score+Gumbel)/tau)) == argmax(score+Gumbel) for every tau>0.
    # All N candidates remain. Blocking changes storage, not candidate support.
    n=len(counts)
    with torch.no_grad():
        for first in range(0,n,block_rows):
            last=min(n,first+block_rows)
            score=embedding[first:last]@embedding.T
            u=public_rng.random((last-first,n),dtype=np.float32)
            np.maximum(u,np.finfo(np.float32).tiny,out=u)
            np.log(u,out=u);u*=-1;np.log(u,out=u);u*=-1
            selected=torch.argmax(score+torch.as_tensor(u,device=score.device),dim=1).cpu().numpy()
            counts[np.arange(first,last),selected]+=1

def reconstruct(counts,target,rng,block_rows=128):
    """Appendix-A min-neighbor/weighted reconstruction, with DP target count.

    The official entry passes a true-edge target but its helper ignores it and
    divides a heuristic by250. The repair honors a separately noised target;
    90/10 rho allocation is an explicit adaptation choice, not a paper parameter.
    """
    n=len(counts);target=min(max(int(target),0),n*(n-1)//2)
    if not target:return np.empty((0,2),dtype=np.int64)
    edges=set()
    for i in range(n):
        if len(edges)>=target:break
        w=np.maximum(counts[i],counts[:,i]).astype(np.float64);w[i]=0
        if w.sum():j=int(rng.choice(n,p=w/w.sum()))
        else:
            j=int(rng.integers(n-1));j+=j>=i
        edges.add((min(i,j),max(i,j)))
    needed=target-len(edges)
    if needed:
        keys=np.empty(0);pairs=np.empty((0,2),dtype=np.int64)
        for first in range(0,n,block_rows):
            last=min(n,first+block_rows)
            block=np.maximum(counts[first:last],counts[:,first:last].T)
            rows,cols=np.nonzero(block)
            keep=cols>rows+first;rows=rows[keep]+first;cols=cols[keep]
            if not len(rows):continue
            keep=np.asarray([(int(a),int(b)) not in edges for a,b in zip(rows,cols)])
            rows=rows[keep];cols=cols[keep]
            if not len(rows):continue
            weights=np.maximum(counts[rows,cols],counts[cols,rows]).astype(np.float64)
            u=np.maximum(rng.random(len(rows)),np.finfo(float).tiny)
            race=-np.log(u)/weights
            keys=np.concatenate((keys,race));pairs=np.concatenate((pairs,np.column_stack((rows,cols))))
            if len(keys)>needed:
                take=np.argpartition(keys,needed-1)[:needed];keys=keys[take];pairs=pairs[take]
        edges.update((int(a),int(b)) for a,b in pairs)
    while len(edges)<target:
        a=int(rng.integers(n));b=int(rng.integers(n-1));b+=b>=a
        edges.add((min(a,b),max(a,b)))
    return np.asarray(sorted(edges),dtype=np.int64).reshape(-1,2)

def generate(adj,epsilon,delta=1e-5,seed=42,device='cpu',threads=4,progress=None,
             block_rows=128,edge_fraction=.1,test_noise_seed=None,test_mode=False):
    assert test_noise_seed is None or test_mode,'A public private-noise seed is forbidden in formal runs'
    torch.set_num_threads(int(threads));torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    n=len(adj);assert n>=16 and adj.shape==(n,n)
    model=DeepPageRank(n).to(device)
    raw_parameters=list(model.weights)+list(model.biases)
    raw_optimizer=torch.optim.Adam(raw_parameters,lr=PAPER['learning_rate'])
    private_optimizer=torch.optim.Adam([model.embedding],lr=PAPER['learning_rate'])
    ledger=Ledger(float(epsilon),float(delta));total=model.steps
    assert 0<edge_fraction<1
    rho_each=ledger.limit*(1-edge_fraction)/total
    # Hard clipping makes this bound valid regardless of private MLP state,
    # random-walk membership, degree changes, or normalization numerics.
    radius=PAPER['preset_sensitivity']/total
    sigma,cost=gaussian(2*radius,rho_each)
    secret=torch.Generator(device='cpu').manual_seed(secrets.randbits(63) if test_noise_seed is None else test_noise_seed)
    walk=random.Random(seed)
    scores=np.random.default_rng(np.random.SeedSequence([seed,101]))
    post=np.random.default_rng(np.random.SeedSequence([seed,202]))
    neighbors=[np.flatnonzero(adj[i]) for i in range(n)];degrees=np.asarray([len(x) for x in neighbors])
    counts=np.zeros((n,n),dtype=np.uint32)
    initial=model.embedding.detach().cpu().clone();started=time.perf_counter()
    step=0
    for epoch in range(PAPER['epochs']):
        for batch in range(n//PAPER['batch_size']):
            ledger.reserve('embedding_update_%05d'%step,cost)
            heads,tails=sample_pairs(neighbors,range(batch*16,(batch+1)*16),walk)
            raw_optimizer.zero_grad(set_to_none=True)
            loss=pair_loss(model,heads,tails,degrees,n)
            grads=torch.autograd.grad(loss,raw_parameters,allow_unused=True)
            for p,g in zip(raw_parameters,grads):p.grad=torch.zeros_like(p) if g is None else torch.nan_to_num(g,nan=0.,posinf=0.,neginf=0.)
            raw_optimizer.step()
            loss=pair_loss(model,heads,tails,degrees,n)
            grad=torch.autograd.grad(loss,model.embedding,allow_unused=True)[0]
            mean=clip_embedding_query(grad,radius,model.embedding.shape)
            private_optimizer.zero_grad(set_to_none=True)
            noise=torch.randn(mean.shape,generator=secret,dtype=torch.float64)*sigma
            model.embedding.grad=(mean+noise).to(dtype=model.embedding.dtype,device=device)
            private_optimizer.step()
            count_transitions(model.embedding.detach(),counts,scores,block_rows)
            step+=1
            if progress:progress(dict(stage='train',step=step,total_steps=total,epoch=epoch+1,batch=batch+1))
    assert step==total
    count_sigma,count_cost=gaussian(1.,ledger.limit*edge_fraction)
    ledger.reserve('edge_count',count_cost)
    true_m=float(np.count_nonzero(np.triu(adj,1)))
    released_m=true_m+float(torch.randn((),generator=secret,dtype=torch.float64))*count_sigma
    target=int(round(min(max(released_m,0),n*(n-1)/2)))
    if progress:progress(dict(stage='reconstruct',step=step,total_steps=total))
    edges=reconstruct(counts,target,post,block_rows)
    certificate=ledger.report()
    certificate.update(variant='PrivDPR_paper_parameters_embedding_query_edge_zCDP_v1',paper_parameters=PAPER,
        weight_matrices=model.weight_matrices,paper_hidden_depth_L=model.weight_matrices-1,public_training_updates=total,
        clip_radius_per_update=radius,l2_sensitivity_per_update=2*radius,gaussian_sigma=sigma,
        embedding_update_rho=cost,edge_fraction=edge_fraction,edge_count_sigma=count_sigma,target_edges=target,
        private_noise_seed_exposed=False,private_noise_is_OS_seeded=test_noise_seed is None,test_public_noise=bool(test_mode),
        edge_adjacency='one undirected edge add/remove; fixed public labeled vertex set',
        privacy_proof='Condition on noised embedding-gradient transcript: each full embedding query is jointly clipped to radius 5/T, so sensitivity <=10/T. Raw MLP is internal to the query. Sequential Gaussian rho composition plus sensitivity-1 edge count. Reconstruction uses DP transcript, DP count and independent public tapes only.',
        raw_MLP_released=False,generation_queries_raw_graph=False,no_sampling_amplification_claim=True,
        output_scope='synthetic graph and accounted information only; raw metrics/timing/diagnostics trusted audit',
        explicit_adaptations=['PyTorch repair of broken TF1 training/interfaces','L+1 weight matrices per paper Eq13',
            'joint embedding-gradient clipping to the paper public sensitivity target/T',
            'Gaussian zCDP composition replacing original node-DP accounting','DP edge target with explicit rho split, no undocumented /250',
            'argmax-softmax algebra and row blocking retain all public nodes/candidates'])
    metadata=dict(seconds=time.perf_counter()-started,embedding_changed=not torch.equal(initial,model.embedding.detach().cpu()),
        actual_optimizer_updates=step,threads=int(threads),device=str(device),block_rows=block_rows)
    return edges,certificate,metadata
