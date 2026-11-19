"""Exact fallback for finite chains outside the legacy work certificate.

Preserves every arithmetic operation and all overflow checks. The fallback is
quadratic in adjacent support widths; it does NOT claim the old work bound.
"""
import ctypes
import hashlib
import importlib
import json
import time
from pathlib import Path
import numpy as np


def install(out, stop_after_fallback=False):
    native = importlib.import_module('route_s_clean.degree_chain_native')
    original = native._load_function()
    out = Path(out) / 'ragged_diagnostic'
    out.mkdir(parents=True, exist_ok=True)
    fallback = None
    library = None
    checked = False
    records = []

    def load():
        nonlocal fallback, library
        if fallback is None:
            source = Path(native.SOURCE)
            old = source.read_text(encoding='utf-8')
            guard = '    if (total_pair_work > total_budget) {\n        return 3;\n    }'
            assert old.count(guard) == 1
            patched = Path(__file__).with_name('ragged_forward_exact.cpp')
            content = old.replace(guard, '    // Exact execution beyond the legacy work certificate; work is reported.')
            if patched.exists():
                assert patched.read_text(encoding='utf-8') == content
            else:
                patched.write_text(content, encoding='utf-8')
            dll = native.build_shared_library('r7-ragged-exact', patched,
                compiler_flags=('-O3','-std=c++20','-shared','-static-libgcc',
                                '-static-libstdc++','-ffp-contract=off'))
            library = ctypes.CDLL(str(dll))
            fallback = library.route_s_ragged_forward
            fallback.argtypes = original.argtypes
            fallback.restype = original.restype
        return fallback

    def call(*args):
        nonlocal checked
        status = int(original(*args))
        if status == 0 and not checked:
            # Actual production tape, with identical floating point ordering.
            other = list(args[:8]) + [np.empty_like(a) for a in args[8:]]
            assert int(load()(*other)) == 0
            assert all(np.array_equal(a,b) for a,b in zip(args[8:],other[8:]))
            checked = True
            (out/'successful_tape_regression.json').write_text(
                json.dumps({'bitwise_equal': True, 'states':len(args[2])}),encoding='utf-8')
        if status != 3:
            return status
        widths = np.diff(args[1]).tolist()
        pairs = sum(a*b for a,b in zip(widths,widths[1:]))
        budget = sum((a+b-1)*max(1,(a+b-1).bit_length())**2
                     for a,b in zip(widths,widths[1:]))
        if not (0 < budget < pairs <= np.iinfo(np.int64).max):
            raise ArithmeticError('ragged-forward code 3 is not a work-bound failure')
        idx = len(records)
        np.savez_compressed(out/('failed_tape_%03d.npz'%idx),
            offsets=args[1],states=args[2],gain=args[3],degree=args[4],target=args[5],
            quadratic=args[6],temperature=args[7])
        record = {'legacy_status':3,'pair_work':pairs,'legacy_budget':budget,
                  'ratio':pairs/budget,'max_width':max(widths),'status':'RUNNING',
                  'legacy_source_sha256':hashlib.sha256(Path(native.SOURCE).read_bytes()).hexdigest()}
        records.append(record)
        def save():
            (out/'fallback.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        save()
        print('RAGGED exact work-bound fallback',record,flush=True)
        start = time.perf_counter()
        result = int(load()(*args))
        record.update(status='COMPLETE' if result==0 else 'FAILED',
                      returncode=result,seconds=time.perf_counter()-start)
        if result == 0:
            assert int(args[10][0]) == pairs and int(args[11][0]) == budget
            assert np.all(np.isfinite(args[8])) and np.isfinite(args[9][0])
        save()
        if stop_after_fallback and result == 0:
            raise StopIteration('Real failing tape recovered exactly; diagnostic complete')
        return result
    native._FUNCTION = call
