"""Capture-first numerical diagnostics; never edits frozen files on disk.

Only the two non-positive variance failures select centered diagnostics.
Applying a repair requires the explicit allow_repair=True integration flag.
All successful historical calls retain their exact return value and operations.
"""
from __future__ import annotations
import contextlib
import ctypes
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import types
import numpy as np

HERE = Path(__file__).resolve().parent
_LIB = None

def _kernel():
    global _LIB
    if _LIB is None:
        from route_s_clean._native_build import build_shared_library
        path = build_shared_library(
            'centered-degree-moments-v10', HERE / 'centered_moments.cpp',
            compiler_flags=('-O3', '-std=c++20', '-shared', '-static-libgcc',
                            '-static-libstdc++', '-ffp-contract=off'))
        lib = ctypes.CDLL(str(path))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        fn = lib.route_s_centered_one_channel
        fn.argtypes = [ctypes.c_int64, i64, i64, f64, f64, f64, f64,
                       ctypes.c_double, ctypes.c_double, ctypes.c_double, f64, i64]
        fn.restype = ctypes.c_int
        _LIB = (lib, fn, str(path))
    return _LIB

def centered_one_channel_moments(offsets, states, structural_gain, additive,
                                  path_degree, path_target,
                                  inverse_two_variance, coefficient,
                                  *, residual_alpha=0.0):
    arrays = [np.ascontiguousarray(a, dtype=np.int64 if i < 2 else np.float64)
              for i, a in enumerate((offsets, states, structural_gain, additive,
                                     path_degree, path_target))]
    off, states, gain, score, degree, target = arrays
    count = len(off)-1
    if (count <= 0 or off[0] != 0 or np.any(np.diff(off) <= 0)
            or states.shape != (off[-1],) or gain.shape != states.shape
            or score.shape != states.shape or degree.shape != (count+1,)
            or target.shape != degree.shape):
        raise ValueError('invalid centered one-channel arrays')
    _, fn, dll_path = _kernel()
    output = np.empty(6, dtype=np.float64)
    counters = np.empty(3, dtype=np.int64)
    began = time.perf_counter()
    status = fn(count, *arrays, float(inverse_two_variance), float(coefficient),
                float(residual_alpha), output, counters)
    if status:
        raise ArithmeticError(f'centered moment recurrence returned {status}')
    ms, md, vs, vd, cov, log_z = map(float, output)
    if counters[2] <= 53:
        raise ArithmeticError('centered moment build lacks extended accumulator precision')
    # These raw fields are diagnostics only: consumers below use central moments.
    return dict(mean=ms, degree_mean=md, variance=vs, degree_variance=vd,
                covariance=cov, second=vs+ms*ms, degree_second=vd+md*md,
                cross=cov+ms*md, log_normalizer=log_z,
                transition_work=int(counters[0]), underflowed_transition_weights=int(counters[1]),
                accumulator_mantissa_bits=int(counters[2]), dll_path=dll_path,
                seconds=time.perf_counter()-began,
                _args=(*arrays, float(inverse_two_variance), float(coefficient)))

def _residual_variance(raw, alpha):
    residual = centered_one_channel_moments(*raw['_args'], residual_alpha=alpha)
    raw['residual_variance_direct'] = residual['variance']
    raw['residual_pass_seconds'] = residual['seconds']
    raw['residual_log_normalizer_identical'] = (
        residual['log_normalizer'] == raw['log_normalizer'])
    if not raw['residual_log_normalizer_identical']:
        raise ArithmeticError('residual second pass changed the chain weights')
    return residual['variance']

def _stable_consumer(original, recorder):
    source = inspect.getsource(original)
    replacements = {
        'raw = fused_one_channel_moments(':'raw = centered_one_channel_moments(',
        'float(raw["second"]) - mean_score * mean_score':'raw["variance"]',
        'float(raw["degree_second"]) - mean_degree * mean_degree':'raw["degree_variance"]',
        'float(float(raw["cross"]) - mean_score * mean_degree)':'float(raw["covariance"])',
        'residual_variance = float(\n        moments.variance_structural_score\n        + alpha * alpha * variance_degree\n        - 2.0 * alpha * covariance\n    )':
            'residual_variance = _residual_variance(raw, alpha)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f'frozen consumer source mismatch: {old!r}')
        source = source.replace(old, new)
    namespace = dict(original.__globals__)
    def recorded(*args, **kwargs):
        result = centered_one_channel_moments(*args, **kwargs)
        recorder['stable'] = result
        if result['log_normalizer'] != recorder['raw']['log_normalizer']:
            raise ArithmeticError('centered recurrence changed the frozen log normalizer')
        return result
    namespace.update(centered_one_channel_moments=recorded,
                     _residual_variance=_residual_variance)
    exec(compile(source, str(HERE/'in_memory_centered_consumer.py'), 'exec'), namespace)
    return namespace[original.__name__]

def _summary(raw):
    return {k: v for k, v in raw.items() if not k.startswith('_')}

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _save_failure(directory, args, raw, reason):
    directory.mkdir(parents=True, exist_ok=True)
    names=('offsets', 'states', 'structural_gain', 'additive', 'path_degree', 'path_target')
    paths={}
    for name, array in zip(names, args[:6]):
        path=directory/(name+'.npy')
        np.save(path, np.asarray(array), allow_pickle=False)
        paths[name]=dict(path=str(path), shape=list(np.shape(array)),
                         sha256=file_sha256(path))
    record=dict(reason=reason, inputs=paths, inverse_two_variance=float(args[6]),
                coefficient=float(args[7]), raw=_summary(raw),
                raw_variance=float(raw['second']-raw['mean']**2),
                raw_degree_variance=float(raw['degree_second']-raw['degree_mean']**2),
                raw_covariance=float(raw['cross']-raw['mean']*raw['degree_mean']),
                no_private_query_added=True, no_graph_generation_in_this_module=True,
                implementation_sha256={name: file_sha256(HERE/name) for name in
                    ('stable_moments.py', 'centered_moments.cpp')})
    (directory/'failure.json').write_text(json.dumps(record, indent=2)+'\n',encoding='utf-8')
    return record

@contextlib.contextmanager
def installed(audit_directory, *, allow_repair=False, capture_success=False):
    """Patch exact in-memory aliases, restoring them on exit; serial use only.

    Frozen forward / generation files and successful historical calls stay intact.
    By default a failed call is frozen and diagnosed, then the original failure
    is re-raised. Set allow_repair=True only after reviewing actual-tape evidence.
    A repair may change numerical scores and therefore downstream graph output;
    it is a new numerical version, not a bitwise-equivalent acceleration claim.
    """
    import route_s_clean.chain_moments as module
    original = module.efficient_spectral_score_messages
    audit_directory=Path(audit_directory)
    counter=[0]
    def replacement(*args, **kwargs):
        recorder={}
        old_fused=original.__globals__['fused_one_channel_moments']
        def capture(*inputs, **kw):
            result=old_fused(*inputs, **kw)
            recorder['raw']=result
            recorder['args']=inputs
            return result
        namespace=dict(original.__globals__)
        namespace['fused_one_channel_moments']=capture
        legacy=types.FunctionType(original.__code__, namespace, original.__name__,
                                  original.__defaults__, original.__closure__)
        legacy.__kwdefaults__=original.__kwdefaults__
        try:
            result = legacy(*args, **kwargs)
            if capture_success and 'raw' in recorder:
                counter[0] += 1
                _save_failure(audit_directory/f'success_{counter[0]:03d}',
                              recorder['args'], recorder['raw'], 'successful_frozen_call')
            return result
        except ArithmeticError as exc:
            if str(exc) not in ('degree-energy nuisance variance is not positive',
                                'efficient spectral residual variance is not positive'):
                raise
            if 'raw' not in recorder:
                raise
            counter[0]+=1
            directory=audit_directory/f'failure_{counter[0]:03d}'
            report=_save_failure(directory,recorder['args'],recorder['raw'],str(exc))
            consumer=_stable_consumer(original,recorder)
            try:
                stable_kwargs = dict(kwargs)
                if not allow_repair:
                    stable_kwargs['_calibration_only'] = True
                    stable_kwargs['_retain_spectral_columns'] = False
                result=consumer(*args,**stable_kwargs)
                result=replace(result, diagnostics={
                    **dict(result.diagnostics),
                    'numerical_moments_version': 'v10_centered_long_double',
                    'historical_raw_variance_failure': str(exc),
                    'variance_clamping_applied': False,
                    'frozen_log_normalizer_identical': True,
                })
                report['repair_status']='centered_variance_domain_passed'
                report['consumer_diagnostics']=dict(result.diagnostics)
                report['repair_applied']=bool(allow_repair)
                if allow_repair:
                    return result
            except Exception as stable_exc:
                report['repair_status']='still_invalid_fail_closed'
                report['repair_error']=repr(stable_exc)
                raise
            finally:
                if 'stable' in recorder:
                    stable=recorder['stable']
                    report['centered']=_summary(stable)
                    report['frozen_forward_normalizer_identical']=(
                        stable['log_normalizer']==recorder['raw']['log_normalizer'])
                (directory/'repair.json').write_text(
                    json.dumps(report,indent=2)+'\n',encoding='utf-8')
            raise
    bindings=[]
    for module in tuple(sys.modules.values()):
        if module is None: continue
        for name,value in tuple(vars(module).items()):
            if value is original:
                bindings.append((module,name))
    for module,name in bindings: setattr(module,name,replacement)
    try:
        yield dict(patched_alias_count=len(bindings), audit_directory=str(audit_directory),
                   allow_repair=bool(allow_repair), capture_success=bool(capture_success))
    finally:
        for module,name in bindings:
            if getattr(module,name,None) is replacement: setattr(module,name,original)

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('failure_directory',type=Path)
    args=parser.parse_args()
    data=json.loads((args.failure_directory/'failure.json').read_text(encoding='utf-8'))
    names=('offsets','states','structural_gain','additive','path_degree','path_target')
    arrays=[np.load(data['inputs'][name]['path'],allow_pickle=False) for name in names]
    stable=centered_one_channel_moments(*arrays,data['inverse_two_variance'],data['coefficient'])
    if stable['degree_variance']>0:
        stable['residual_variance_direct']=_residual_variance(stable,stable['covariance']/stable['degree_variance'])
    result=_summary(stable)
    (args.failure_directory/'standalone_centered.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
