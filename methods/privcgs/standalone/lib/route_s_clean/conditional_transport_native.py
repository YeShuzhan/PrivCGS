"""Native packed loops for the exact conditional information transport."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source


SOURCE = native_source("conditional_transport", "packed_transport.cpp")
_LIBRARY = None
_MARGINAL_FUNCTION = None
_PROJECTION_FUNCTION = None
_SUFFICIENT_MARGINAL_FUNCTION = None
_ON_DEMAND_PROJECTION_FUNCTION = None


def _load_library():
    global _LIBRARY
    if _LIBRARY is None:
        path = build_shared_library(
            "packed-conditional-transport",
            SOURCE,
            compiler_flags=(
                "-O3",
                "-std=c++20",
                "-shared",
                "-static-libgcc",
                "-static-libstdc++",
                "-ffp-contract=off",
            ),
        )
        _LIBRARY = ctypes.CDLL(str(path))
    return _LIBRARY


def _load_marginal_function():
    global _MARGINAL_FUNCTION
    if _MARGINAL_FUNCTION is not None:
        return _MARGINAL_FUNCTION
    library = _load_library()
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_chain_marginals
    function.argtypes = [
        ctypes.c_int64,
        i64,
        i64,
        f64,
        f64,
        f64,
        f64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        f64,
        i64,
        f64,
        f64,
        f64,
        i64,
    ]
    function.restype = ctypes.c_int32
    _MARGINAL_FUNCTION = function
    return function


def _load_projection_function():
    global _PROJECTION_FUNCTION
    if _PROJECTION_FUNCTION is not None:
        return _PROJECTION_FUNCTION
    library = _load_library()
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_packed_entropic_projection
    function.argtypes = [
        ctypes.c_int64,
        i64,
        f64,
        i64,
        f64,
        f64,
        f64,
        i64,
        f64,
        f64,
    ]
    function.restype = ctypes.c_int32
    _PROJECTION_FUNCTION = function
    return function


def _load_sufficient_marginal_function():
    global _SUFFICIENT_MARGINAL_FUNCTION
    if _SUFFICIENT_MARGINAL_FUNCTION is not None:
        return _SUFFICIENT_MARGINAL_FUNCTION
    library = _load_library()
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_chain_marginals_sufficient
    function.argtypes = [
        ctypes.c_int64,
        i64,
        i64,
        f64,
        f64,
        f64,
        f64,
        ctypes.c_double,
        ctypes.c_double,
        f64,
        f64,
        f64,
        f64,
        i64,
    ]
    function.restype = ctypes.c_int32
    _SUFFICIENT_MARGINAL_FUNCTION = function
    return function


def _load_on_demand_projection_function():
    global _ON_DEMAND_PROJECTION_FUNCTION
    if _ON_DEMAND_PROJECTION_FUNCTION is not None:
        return _ON_DEMAND_PROJECTION_FUNCTION
    library = _load_library()
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_on_demand_entropic_projection_sufficient
    function.argtypes = [
        ctypes.c_int64,
        i64,
        i64,
        f64,
        f64,
        f64,
        f64,
        f64,
        ctypes.c_double,
        f64,
        f64,
        f64,
        i64,
        f64,
        f64,
        f64,
    ]
    function.restype = ctypes.c_int32
    _ON_DEMAND_PROJECTION_FUNCTION = function
    return function


def packed_chain_marginals(
    offsets,
    states,
    gain,
    forward,
    path_degree,
    path_target,
    quadratic: float,
    log_normalizer: float,
    *,
    materialize_transitions: bool,
):
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    states = np.ascontiguousarray(states, dtype=np.int64)
    gain = np.ascontiguousarray(gain, dtype=np.float64)
    forward = np.ascontiguousarray(forward, dtype=np.float64)
    degree = np.ascontiguousarray(path_degree, dtype=np.float64)
    target = np.ascontiguousarray(path_target, dtype=np.float64)
    position_count = int(offsets.size - 1)
    widths = np.diff(offsets)
    if (
        position_count <= 0
        or offsets[0] != 0
        or np.any(widths <= 0)
        or states.shape != (int(offsets[-1]),)
        or gain.shape != states.shape
        or forward.shape != states.shape
        or degree.shape != (position_count + 1,)
        or target.shape != degree.shape
    ):
        raise ValueError("invalid packed conditional-chain arrays")
    transition_count = np.asarray(widths[:-1] * widths[1:], dtype=np.int64)
    transition_offsets = np.zeros(position_count, dtype=np.int64)
    if transition_count.size:
        transition_offsets[1:] = np.cumsum(transition_count, dtype=np.int64)
    transition_size = int(transition_offsets[-1]) if materialize_transitions else 0
    transitions = np.empty(transition_size, dtype=np.float64)
    marginals = np.empty(states.size, dtype=np.float64)
    maximum_marginal_error = np.empty(1, dtype=np.float64)
    maximum_row_error = np.empty(1, dtype=np.float64)
    transition_work = np.empty(1, dtype=np.int64)
    status = int(
        _load_marginal_function()(
            position_count,
            offsets,
            states,
            gain,
            forward,
            degree,
            target,
            float(quadratic),
            float(log_normalizer),
            int(bool(materialize_transitions)),
            marginals,
            transition_offsets,
            transitions,
            maximum_marginal_error,
            maximum_row_error,
            transition_work,
        )
    )
    if status:
        raise ArithmeticError(f"packed conditional marginal kernel returned {status}")
    return (
        marginals,
        transition_offsets,
        transitions,
        int(transition_work[0]),
        float(maximum_marginal_error[0]),
        float(maximum_row_error[0]),
    )


def packed_chain_marginals_sufficient(
    offsets,
    states,
    gain,
    forward,
    path_degree,
    path_target,
    quadratic: float,
    log_normalizer: float,
):
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    states = np.ascontiguousarray(states, dtype=np.int64)
    gain = np.ascontiguousarray(gain, dtype=np.float64)
    forward = np.ascontiguousarray(forward, dtype=np.float64)
    degree = np.ascontiguousarray(path_degree, dtype=np.float64)
    target = np.ascontiguousarray(path_target, dtype=np.float64)
    position_count = int(offsets.size - 1)
    widths = np.diff(offsets)
    if (
        position_count <= 0
        or offsets[0] != 0
        or np.any(widths <= 0)
        or states.shape != (int(offsets[-1]),)
        or gain.shape != states.shape
        or forward.shape != states.shape
        or degree.shape != (position_count + 1,)
        or target.shape != degree.shape
    ):
        raise ValueError("invalid sufficient conditional-chain arrays")
    marginals = np.empty(states.size, dtype=np.float64)
    backward = np.empty(states.size, dtype=np.float64)
    row_correction = np.empty(states.size, dtype=np.float64)
    maximum_marginal_error = np.empty(1, dtype=np.float64)
    transition_work = np.empty(1, dtype=np.int64)
    status = int(
        _load_sufficient_marginal_function()(
            position_count,
            offsets,
            states,
            gain,
            forward,
            degree,
            target,
            float(quadratic),
            float(log_normalizer),
            marginals,
            backward,
            row_correction,
            maximum_marginal_error,
            transition_work,
        )
    )
    if status:
        raise ArithmeticError(f"sufficient conditional marginal kernel returned {status}")
    return (
        marginals,
        backward,
        row_correction,
        int(transition_work[0]),
        float(maximum_marginal_error[0]),
    )


def packed_entropic_projection(
    state_offsets,
    target_log_marginal_value,
    transition_offsets,
    reference_transition_value,
):
    offsets = np.ascontiguousarray(state_offsets, dtype=np.int64)
    targets = np.ascontiguousarray(target_log_marginal_value, dtype=np.float64)
    transition_offsets = np.ascontiguousarray(transition_offsets, dtype=np.int64)
    transitions = np.ascontiguousarray(reference_transition_value, dtype=np.float64)
    position_count = int(offsets.size - 1)
    if (
        position_count <= 0
        or offsets[0] != 0
        or np.any(offsets[1:] <= offsets[:-1])
        or targets.shape != (int(offsets[-1]),)
        or transition_offsets.shape != (position_count,)
        or transitions.shape != (int(transition_offsets[-1]),)
    ):
        raise ValueError("invalid packed entropic-projection arrays")
    left = np.empty(targets.size, dtype=np.float64)
    right = np.empty(targets.size, dtype=np.float64)
    total_iterations = np.empty(1, dtype=np.int64)
    maximum_error = np.empty(1, dtype=np.float64)
    maximum_bound = np.empty(1, dtype=np.float64)
    status = int(
        _load_projection_function()(
            position_count,
            offsets,
            targets,
            transition_offsets,
            transitions,
            left,
            right,
            total_iterations,
            maximum_error,
            maximum_bound,
        )
    )
    if status:
        raise ArithmeticError(f"packed entropic projection kernel returned {status}")
    return (
        left,
        right,
        int(total_iterations[0]),
        float(maximum_error[0]),
        float(maximum_bound[0]),
    )


def packed_on_demand_entropic_projection(
    state_offsets,
    states,
    gain,
    backward,
    row_correction,
    path_degree,
    path_target,
    quadratic: float,
    target_log_marginal_value,
):
    offsets = np.ascontiguousarray(state_offsets, dtype=np.int64)
    states = np.ascontiguousarray(states, dtype=np.int64)
    gain = np.ascontiguousarray(gain, dtype=np.float64)
    backward = np.ascontiguousarray(backward, dtype=np.float64)
    row_correction = np.ascontiguousarray(row_correction, dtype=np.float64)
    degree = np.ascontiguousarray(path_degree, dtype=np.float64)
    path_target = np.ascontiguousarray(path_target, dtype=np.float64)
    targets = np.ascontiguousarray(target_log_marginal_value, dtype=np.float64)
    position_count = int(offsets.size - 1)
    state_count = int(offsets[-1]) if position_count > 0 else 0
    if (
        position_count <= 0
        or offsets[0] != 0
        or np.any(offsets[1:] <= offsets[:-1])
        or states.shape != (state_count,)
        or gain.shape != (state_count,)
        or backward.shape != (state_count,)
        or row_correction.shape != (state_count,)
        or degree.shape != (position_count + 1,)
        or path_target.shape != degree.shape
        or targets.shape != (state_count,)
    ):
        raise ValueError("invalid on-demand entropic-projection arrays")
    left = np.empty(state_count, dtype=np.float64)
    right = np.empty(state_count, dtype=np.float64)
    total_iterations = np.empty(1, dtype=np.int64)
    maximum_error = np.empty(1, dtype=np.float64)
    maximum_bound = np.empty(1, dtype=np.float64)
    maximum_row_error = np.empty(1, dtype=np.float64)
    status = int(
        _load_on_demand_projection_function()(
            position_count,
            offsets,
            states,
            gain,
            backward,
            row_correction,
            degree,
            path_target,
            float(quadratic),
            targets,
            left,
            right,
            total_iterations,
            maximum_error,
            maximum_bound,
            maximum_row_error,
        )
    )
    if status:
        import os
        from pathlib import Path
        capture = os.environ.get('Q_PROJECTION_FAILURE_CAPTURE')
        if capture:
            destination = Path(capture)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination, state_offsets=offsets, states=states, gain=gain,
                     backward=backward, row_correction=row_correction,
                     path_degree=degree, path_target=path_target,
                     quadratic=float(quadratic), target_log_marginal_value=targets)
        raise ArithmeticError(f"on-demand entropic projection kernel returned {status}")
    return (
        left,
        right,
        int(total_iterations[0]),
        float(maximum_error[0]),
        float(maximum_bound[0]),
        float(maximum_row_error[0]),
    )


__all__ = (
    "packed_chain_marginals",
    "packed_chain_marginals_sufficient",
    "packed_entropic_projection",
    "packed_on_demand_entropic_projection",
)
