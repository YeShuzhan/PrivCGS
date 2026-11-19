"""Timing categories only; never inspects graph data or changes random streams."""
from contextlib import contextmanager
from functools import wraps
import threading
import time

TOTALS = {'native_setup': 0.0, 'diagnostic_io': 0.0, 'mechanism_validation': 0.0}
_LOCAL = threading.local()

@contextmanager
def charge(category):
    depth = getattr(_LOCAL, 'depth', 0)
    _LOCAL.depth = depth + 1
    started = time.perf_counter()
    try:
        yield
    finally:
        if not depth:
            TOTALS[category] += time.perf_counter() - started
        _LOCAL.depth = depth

def native_setup(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with charge('native_setup'):
            return function(*args, **kwargs)
    return wrapped

@contextmanager
def diagnostic_writes(directory):
    """Account diagnostic serialization under this run; required queue I/O stays in S."""
    from pathlib import Path
    import numpy as np
    root = Path(directory).resolve()
    saved = []
    def wrap(owner, name):
        original = getattr(owner, name)
        @wraps(original)
        def invoke(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to(root):
                with charge('diagnostic_io'):
                    return original(path, *args, **kwargs)
            return original(path, *args, **kwargs)
        saved.append((owner, name, original))
        setattr(owner, name, invoke)
    wrap(Path, 'write_text')
    wrap(Path, 'write_bytes')
    for name in ('save', 'savez', 'savez_compressed'):
        wrap(np, name)
    try:
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)
