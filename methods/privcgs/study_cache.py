"""Exact stage reuse from this configuration's same private checkpoint."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json,pickle
import numpy as np
from common import sha_array
from execution import sha,write


def load(config,release,output):
    if not config.get('upstream_cache'):return None
    assert config['variant']!='a'
    descriptor=Path(config['upstream_cache'])
    metadata=json.loads(descriptor.read_bytes())
    assert metadata['dataset']==config['dataset'] and metadata['epsilon']==config['epsilon'] and metadata['seed']==config['seed']
    assert metadata['checkpoint_integrity_hash']==release.checkpoint.integrity_hash
    files=metadata['files']
    for name,record in files.items():assert sha(record['path'])==record['sha256'],name
    source=json.loads(Path(files['generation_report']['path']).read_bytes())
    S=np.load(files['S']['path'],allow_pickle=False)
    Q=np.load(files['Q']['path'],allow_pickle=False)
    target=np.load(files['target']['path'],allow_pickle=False)
    n=len(target)
    assert sha_array(S)==source['S_sha256'] and sha_array(target)==source['target_sha256']
    assert len(S)==len(Q)==source['m'] and target.sum()==2*len(S)
    assert S.shape==(len(S),2) and Q.ndim==1 and len(np.unique(Q))==len(Q)
    Skeys=S[:,0]*n+S[:,1]
    removed=np.setdiff1d(Skeys,Q,assume_unique=True);added=np.setdiff1d(Q,Skeys,assume_unique=True)
    rem=np.column_stack((removed//n,removed%n)).astype(np.int64)
    add=np.column_stack((added//n,added%n)).astype(np.int64)
    probability=np.load(files['aligned_probability']['path'],allow_pickle=False)
    logp=np.load(files['aligned_log_probability']['path'],allow_pickle=False)
    certificate=json.loads(Path(files['target_certificate']['path']).read_bytes())
    spectrum=SimpleNamespace(probability=probability,diagnostics=certificate.get('spectrum_diagnostics',{}))
    cache=dict(S=S,Q=Q,target=target,rem=rem,add=add,spectrum=spectrum,logp=logp,
               certificate=certificate,source=source,metadata=metadata)
    if files.get('H0'):cache['H0']=np.load(files['H0']['path'],allow_pickle=False)
    write(Path(output)/'REUSE_EVIDENCE.json',dict(descriptor=str(descriptor),descriptor_sha256=sha(descriptor),
          checkpoint_integrity_hash=release.checkpoint.integrity_hash,files=files,
          S_sha256=sha_array(S),Q_sha256=sha_array(Q),target_sha256=sha_array(target),new_private_queries=0))
    return cache


def recovered(cache,release,seed,config):
    if config['variant']=='a':
        from study_hooks import recover_without_nonconstant_interactions
        return recover_without_nonconstant_interactions(release.checkpoint,seed)
    if cache is not None and config['variant'] not in ('b','c'):
        return SimpleNamespace(endpoint_target=cache['target'])
    from recovery import recover_checkpoint_in_memory
    return recover_checkpoint_in_memory(release.checkpoint,seed)


def spectrum(cache,release,original):
    if cache is not None:return cache['spectrum'],cache['logp']
    return original(release.checkpoint,skip_integerization=True)


def source_and_flow(cache,recovered_state,seed,capture):
    if cache is not None:
        return SimpleNamespace(edge_array=cache['S']),None,None,SimpleNamespace(diagnostics=dict(S_Q_reused=True,new_private_queries=0))
    return capture(recovered_state,seed)


def selected(cache,tape,flow,original):
    return (cache['rem'],cache['add']) if cache is not None else original(tape,flow)


def target(cache,config,original,*args,**kwargs):
    if cache is None:return original(*args,**kwargs)
    variant=config['variant']
    if variant=='b':
        from study_hooks import direct_degree_assignment
        result,assignment=direct_degree_assignment(np.sort(cache['target']),args[0],args[3])
        certificate=dict(cache['certificate'],assignment=assignment,integer_multiset_reused=True)
        write(Path(kwargs['output'])/'TMMR_certificate.json',certificate)
        return result,certificate
    if variant=='c':
        import tmmr
        saved=tmmr.original_spectrum
        tmmr.original_spectrum=lambda *_a,**_k:(cache['spectrum'],cache['logp'])
        try:return original(*args,**kwargs)
        finally:tmmr.original_spectrum=saved
    certificate=dict(cache['certificate'],target_reused=True)
    write(Path(kwargs['output'])/'TMMR_certificate.json',certificate)
    return cache['target'].copy(),certificate


def initial_graph(cache,config,target,S,rem,add,Q,partial,finish,canonical):
    if cache is not None and 'H0' in cache and config['variant'] not in ('b','c'):
        return cache['H0'].copy()
    prefix,residual,active,closed,boundary=partial(target,S,rem,add)
    n=len(target);tail=finish(active,set(zip(map(int,Q//n),map(int,Q%n))))
    return canonical(np.concatenate((prefix,np.asarray(tail,dtype=np.int64).reshape(-1,2))))
