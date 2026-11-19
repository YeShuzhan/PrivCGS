"""Local execution configuration, stage heartbeat, and declared timing boundaries."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import ctypes
import hashlib
import json
import os
import sys
import threading
import time
import psutil
from execution_accounting import TOTALS, charge

ROOT = Path(__file__).resolve().parents[1]
_DLL_DIRECTORIES = []

def configure(cpu='', compiler=None):
    if cpu:
        psutil.Process().cpu_affinity([int(x) for x in cpu.split(',')])
    if compiler:
        os.environ['CXX'] = str(Path(compiler).resolve())
    selected = os.environ.get('CXX')
    if selected and Path(selected).is_file():
        directory = str(Path(selected).resolve().parent)
        os.environ['PATH'] = directory + os.pathsep + os.environ.get('PATH', '')
        if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
            _DLL_DIRECTORIES.append(os.add_dll_directory(directory))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def external_imports():
    # Compare actual module origins with stdlib/site-packages, not a project-name substring.
    from sysconfig import get_paths
    import site
    allowed = [ROOT, Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()]
    allowed += [Path(v).resolve() for k, v in get_paths().items() if k in ('purelib', 'platlib', 'stdlib')]
    allowed += [Path(v).resolve() for v in site.getsitepackages()]
    allowed.append(Path(site.getusersitepackages()).resolve())
    result = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename or filename.startswith('<'):
            continue
        path = Path(filename).resolve()
        if not any(path.is_relative_to(parent) for parent in allowed):
            result[name] = str(path)
    return result

class Observer:
    def __init__(self, directory, interval=30):
        self.directory = Path(directory)
        self.process = psutil.Process()
        self.current = 'setup'
        self.stage_started = time.perf_counter()
        self.started = self.stage_started
        self.native_diagnostic_seconds = 0.0
        self.native_reader = None
        self.stages = []
        self.stop_event = threading.Event()
        self.monitor_cpu_seconds = 0.0
        self.interval = interval
        self.status = 'RUNNING'
        self.snapshot()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def snapshot(self):
        memory = self.process.memory_info()
        record = dict(status=self.status, pid=self.process.pid, create_time=self.process.create_time(),
                      stage=self.current, stage_elapsed_seconds=time.perf_counter()-self.stage_started,
                      elapsed_seconds=time.perf_counter()-self.started,
                      RSS_MiB=memory.rss/2**20,
                      process_peak_RSS_MiB=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024)
        temporary = self.directory/'status.pending.json'
        # Deliberately not Path.write_text: the monitor overlaps the algorithm, so
        # its thread CPU is reported separately, never subtracted from wall time.
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
        # Windows readers may briefly open the destination without delete sharing.
        # Never let a transient monitoring-file lock terminate the heartbeat.
        for attempt in range(20):
            try:
                temporary.replace(self.directory/'status.json')
                break
            except PermissionError:
                if attempt == 19:
                    return  # pending snapshot retained; next heartbeat retries
                time.sleep(0.05)

    def monitor(self):
        while not self.stop_event.wait(self.interval):
            began = time.thread_time()
            self.snapshot()
            self.monitor_cpu_seconds += time.thread_time()-began

    def excluded(self):
        native = self.native_reader() if self.native_reader else self.native_diagnostic_seconds
        return dict(TOTALS, native_diagnostic_io=float(native))

    @contextmanager
    def stage(self, name):
        previous, previous_start = self.current, self.stage_started
        self.current, self.stage_started = name, time.perf_counter()
        before = self.excluded()
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter()-started
            after = self.excluded()
            excluded = {k: after[k]-before[k] for k in before}
            self.stages.append(dict(stage=name, wall_seconds=elapsed, excluded_seconds=excluded,
                                    algorithm_seconds=elapsed-sum(excluded.values()),
                                    nested_within=previous))
            self.current, self.stage_started = previous, previous_start

    @contextmanager
    def pipeline_stages(self):
        import route_s_clean.pipeline as pipeline
        from route_s_clean.source_rh2b import load_rh2b_library
        library = load_rh2b_library()
        library.rh2b_diagnostic_io_seconds.restype = ctypes.c_double
        self.native_reader = library.rh2b_diagnostic_io_seconds
        names = {'build_source_order_tree':'S_tree', 'generate_source_edges':'S_sampling',
                 'build_unique_dyad_chart':'Q_unique_chart', 'build_descent_support_chart':'Q_descent_chart',
                 '_horizontal_endpoints':'Q_horizontal_law', '_vertical_endpoints':'Q_vertical_law',
                 'baseline_preserving_information_transport':'Q_transport', 'sample_backward':'Q_draw'}
        saved = {}
        for attr, label in names.items():
            original = getattr(pipeline, attr)
            saved[attr] = original
            def decorate(function, stage_name):
                @wraps(function)
                def wrapped(*args, **kwargs):
                    with self.stage(stage_name):
                        return function(*args, **kwargs)
                return wrapped
            setattr(pipeline, attr, decorate(original, label))
        try:
            yield
        finally:
            for attr, original in saved.items():
                setattr(pipeline, attr, original)

    def close(self, status):
        self.stop_event.set()
        self.thread.join()
        self.status = status
        self.snapshot()
