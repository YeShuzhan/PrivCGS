"""Only the active release, recovery, S/Q and exact-completion implementations."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import os
import numpy as np
import route_s_clean.pipeline as pipeline
import route_s_clean.source_order_native as source_native
import degree_chain_gain_cache_candidate_v1 as gain
import horizontal_palm_parallel_grouped_candidate_v2 as palm
import positive_hamiltonian_native_candidate_v1 as hamiltonian
import information_transport_singleton_exact_candidate_v1 as transport
import sample_backward_singleton_advance_candidate_v1 as backward
import gfl_mpe_rkss1_pprs1_v1 as spectrum_native
import spectrum_solver
from recovery import recover_checkpoint_in_memory
from common import canonical, sha_array

ROOT=Path(__file__).resolve().parents[1]
@contextmanager
def installed(out,threads=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bundled=ROOT/'native/source_order.dll'
    if os.name=='nt' and bundled.is_file():
        source_native._build=lambda:bundled
    from route_s_clean.source_rh2b import load_rh2b_library
    load_rh2b_library()
    gain.load_library();palm.load_library();hamiltonian.load_library()
    spectrum_native.THREAD_COUNT=int(threads);spectrum_native._activate()
    spectrum_native.implementation._load_library()
    from constant_closure_certificate import install as support
    from spectrum_log_bracket import install as bracket
    from ragged_work_guard import install as ragged
    import stable_moments
    support(out/'closure_support.json')
    bracket(SimpleNamespace(GFL=SimpleNamespace(V41=spectrum_solver)),out/'spectrum_bracket.json')
    ragged(out)
    restores=[]
    def patch(owner,key,value):
        old=owner[key] if isinstance(owner,dict) else getattr(owner,key)
        def restore():
            if isinstance(owner,dict):owner[key]=old
            else:setattr(owner,key,old)
        restores.append(restore)
        if isinstance(owner,dict):owner[key]=value
        else:setattr(owner,key,value)
    patch(gain,'_FROZEN',pipeline)
    patch(pipeline,'build_unique_dyad_chart',gain.build_unique_dyad_chart)
    patch(pipeline,'build_descent_support_chart',gain.build_descent_support_chart)
    endpoint=pipeline._horizontal_endpoints.__globals__
    patch(endpoint,'compile_horizontal_palm_score_columns',palm.compile_horizontal_palm_score_columns)
    patch(endpoint,'positive_hamiltonian_completion',hamiltonian.positive_hamiltonian_completion)
    transport.configure_production(pipeline.baseline_preserving_information_transport)
    patch(pipeline,'baseline_preserving_information_transport',transport.baseline_preserving_information_transport)
    backward.configure_production(pipeline.sample_backward)
    patch(pipeline,'sample_backward',backward.sample_backward)
    try:
        with stable_moments.installed(out/'moments',allow_repair=True):
            yield
    finally:
        for restore in reversed(restores):restore()

def accelerated_spectrum(checkpoint,threads=4):
    spectrum_native.THREAD_COUNT=int(threads)
    spectrum_native.reset_plan()
    return spectrum_solver.released_spectrum_in_memory_v41(checkpoint)

def capture_source_and_flow(recovered,experiment_id):
    """The original capture boundary; terminal and activation were identities."""
    captured={}
    saved={name:getattr(pipeline,name) for name in
           ('recover_checkpoint_state','apply_relocation_state','prepare_terminal_state','apply_terminal')}
    def retain(source,tape,state):
        captured.update(source=source,tape=tape,flow=np.ascontiguousarray(state,dtype=np.int64).copy())
        return source
    def prepare(edges,_law,**_kw):
        return SimpleNamespace(edge_array=edges,preparation_statistics={'terminal_bypassed':True})
    def apply(prepared,**_kw):
        edge=np.ascontiguousarray(prepared.edge_array,dtype=np.int64)
        return edge,dict(terminal_bypassed=True,edge_array_sha256=sha_array(edge))
    pipeline.recover_checkpoint_state=lambda *_a,**_kw:recovered
    pipeline.apply_relocation_state=retain
    pipeline.prepare_terminal_state=prepare
    pipeline.apply_terminal=apply
    try:
        result=pipeline.synthesize_from_checkpoint(Path('<in-memory-private-transcript>'),public_experiment_id=int(experiment_id))
    finally:
        for name,value in saved.items():setattr(pipeline,name,value)
    assert np.array_equal(canonical(result.edge_array),captured['source'].edge_array)
    return captured['source'],captured['tape'],captured['flow'],result
