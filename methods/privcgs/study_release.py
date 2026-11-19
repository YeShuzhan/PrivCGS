"""Exact original release with configurable r/D and explicit base-stage reuse."""
from __future__ import annotations
import hashlib,inspect,json,pickle
from pathlib import Path
import numpy as np


def install_fixed_D_reuse(module,manifest_path):
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_bytes())
    checkpoint_path=manifest_path.parent/manifest['path']
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==manifest['sha256']
    with checkpoint_path.open('rb') as reader:retained=pickle.load(reader)
    original=module.release_count40_from_csr
    source=inspect.getsource(original)
    begin=source.index('    timer = time.perf_counter()\n    degree_sigma = ')
    end=source.index('    cap, _released_proxy_lower, retained, cap_diagnostics = ')
    # The function's later local variable named retained is a scalar proxy;
    # the immutable release is bound under a distinct namespace name.
    replacement='''    saved = _STUDY_RETAINED_RELEASE
    assert n_nodes == saved.checkpoint.n_nodes
    assert int(operator_basis_rank) == saved.checkpoint.basis.shape[1]
    assert projection_candidate_caps is not None and len(projection_candidate_caps)==1
    assert np.allclose([rho_degree,rho_operator,rho_closure], [saved.transcript.rho_degree,saved.transcript.rho_operator,saved.transcript.rho_closure],rtol=1e-13,atol=1e-15)
    rho_total=saved.transcript.rho_total
    rho_degree=saved.transcript.rho_degree
    rho_operator=saved.transcript.rho_operator
    rho_closure=saved.transcript.rho_closure
    degree_sigma=float(saved.checkpoint.degree_sigma)
    continuous_degree=np.asarray(saved.checkpoint.continuous_noisy_degree)
    basis=np.asarray(saved.checkpoint.basis)
    rank=int(basis.shape[1])
    operator_sensitivity=float(saved.checkpoint.mass_coordinate_weight)
    operator_sigma=float(saved.checkpoint.operator_noise_std)
    degree_seconds=operator_seconds=0.0
'''
    source=source[:begin]+replacement+source[end:]
    begin=source.index('    base_diagnostics = {')
    end=source.index('    transcript_hash = _transcript_hash(')
    source=source[:begin]+'''    base = _STUDY_RETAINED_RELEASE.checkpoint
    base_hash = base.integrity_hash
    assert base_hash
'''+source[end:]
    namespace=dict(module.__dict__,_STUDY_RETAINED_RELEASE=retained)
    exec(compile(source,str(Path(module.__file__))+'::fixed_D_base_reuse','exec'),namespace)
    module.release_count40_from_csr=namespace['release_count40_from_csr']
    return dict(base_manifest=str(manifest_path),base_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        base_release_sha256=manifest['sha256'],checkpoint_integrity_hash=retained.checkpoint.integrity_hash,
        degree_observation_reused=True,operator_observation_reused=True,
        closure_sensitivity_recomputed_for_actual_D=True,
        incremental_release_scope='fixed-D closure query only; original degree/operator bytes retained')
