"""Native executor for the finite-support Gaussian degree chain."""

from __future__ import annotations

import ctypes

import numpy as np

from ._native_build import build_shared_library, native_source

SOURCE = native_source("degree_chain", "ragged_forward.cpp")
_LIBRARY = None
_FUNCTION = None
_MOMENT_FUNCTION = None
_FUSED_MOMENT_FUNCTION = None


def _load_library():
    global _LIBRARY
    if _LIBRARY is None:
        path = build_shared_library(
            "ragged-forward",
            SOURCE,
            compiler_flags=(
                "-O3", "-std=c++20", "-shared", "-static-libgcc",
                "-static-libstdc++", "-ffp-contract=off",
            ),
        )
        _LIBRARY = ctypes.CDLL(str(path))
    return _LIBRARY


def _load_function():
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    library = _load_library()
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_ragged_forward
    function.argtypes = [
        ctypes.c_int64,
        i64,
        i64,
        f64,
        f64,
        f64,
        ctypes.c_double,
        ctypes.c_double,
        f64,
        f64,
        i64,
        i64,
    ]
    function.restype = ctypes.c_int32
    _FUNCTION = function
    return function


def _load_moment_function():
    global _MOMENT_FUNCTION
    if _MOMENT_FUNCTION is not None:
        return _MOMENT_FUNCTION
    _load_function()
    library = _load_library()
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_scalar_chain_moments
    function.argtypes = [
        ctypes.c_int64, ctypes.c_int64, i64, i64, f64, f64, f64, f64, f64,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        f64, f64, f64, f64, f64, f64, f64, i64,
    ]
    function.restype = ctypes.c_int32
    _MOMENT_FUNCTION = function
    return function


def _load_fused_moment_function():
    global _FUSED_MOMENT_FUNCTION
    if _FUSED_MOMENT_FUNCTION is not None:
        return _FUSED_MOMENT_FUNCTION
    _load_function()
    library = _load_library()
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function = library.route_s_fused_one_channel
    function.argtypes = [
        ctypes.c_int64, i64, i64, f64, f64, f64, f64,
        ctypes.c_double, ctypes.c_double, ctypes.c_int32,
        f64, f64, f64, f64, f64, f64, f64, i64,
    ]
    function.restype = ctypes.c_int32
    _FUSED_MOMENT_FUNCTION = function
    return function


def ragged_forward(
    offsets,
    states,
    structural_gain,
    path_degree,
    path_target,
    quadratic: float,
    structural_temperature: float,
) -> tuple[np.ndarray, float, int, int]:
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    states = np.ascontiguousarray(states, dtype=np.int64)
    gain = np.ascontiguousarray(structural_gain, dtype=np.float64)
    degree = np.ascontiguousarray(path_degree, dtype=np.float64)
    target = np.ascontiguousarray(path_target, dtype=np.float64)
    count = int(offsets.size - 1)
    if (
        count <= 0
        or offsets[0] != 0
        or np.any(offsets[1:] <= offsets[:-1])
        or states.size != int(offsets[-1])
        or gain.shape != states.shape
        or degree.shape != (count + 1,)
        or target.shape != degree.shape
    ):
        raise ValueError("invalid flat-ragged Gaussian-chain arrays")
    forward = np.empty(states.size, dtype=np.float64)
    normalizer = np.empty(1, dtype=np.float64)
    pair_work = np.empty(1, dtype=np.int64)
    budget = np.empty(1, dtype=np.int64)
    status = int(
        _load_function()(
            count,
            offsets,
            states,
            gain,
            degree,
            target,
            float(quadratic),
            float(structural_temperature),
            forward,
            normalizer,
            pair_work,
            budget,
        )
    )
    if status:
        raise ArithmeticError(f"clean ragged-forward kernel returned {status}")
    return forward, float(normalizer[0]), int(pair_work[0]), int(budget[0])


def scalar_chain_moments(
    offsets,
    states,
    forward,
    structural_gain,
    additives,
    path_degree,
    path_target,
    inverse_two_variance: float,
    coefficient: float,
    expected_log_normalizer: float,
):
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    states = np.ascontiguousarray(states, dtype=np.int64)
    forward = np.ascontiguousarray(forward, dtype=np.float64)
    gain = np.ascontiguousarray(structural_gain, dtype=np.float64)
    degree = np.ascontiguousarray(path_degree, dtype=np.float64)
    target = np.ascontiguousarray(path_target, dtype=np.float64)
    channels = tuple(
        np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
        for value in additives
    )
    count = int(offsets.size - 1)
    total = int(offsets[-1])
    if (
        count <= 0 or not channels or states.shape != (total,)
        or forward.shape != (total,) or gain.shape != (total,)
        or degree.shape != (count + 1,) or target.shape != degree.shape
        or any(value.shape != (total,) for value in channels)
    ):
        raise ValueError("invalid scalar-chain moment arrays")
    additive = np.ascontiguousarray(np.concatenate(channels), dtype=np.float64)
    channel_count = len(channels)
    mean = np.empty(channel_count, dtype=np.float64)
    second = np.empty(channel_count, dtype=np.float64)
    cross = np.empty(channel_count, dtype=np.float64)
    degree_mean = np.empty(1, dtype=np.float64)
    degree_second = np.empty(1, dtype=np.float64)
    maximum_error = np.empty(1, dtype=np.float64)
    normalizer = np.empty(1, dtype=np.float64)
    work = np.empty(1, dtype=np.int64)
    status = int(_load_moment_function()(
        count, channel_count, offsets, states, forward, gain, additive,
        degree, target, float(inverse_two_variance), float(coefficient),
        float(expected_log_normalizer), mean, second, cross, degree_mean,
        degree_second, maximum_error, normalizer, work,
    ))
    if status:
        raise ArithmeticError(f"clean scalar-chain moment kernel returned {status}")
    variance = second - mean * mean
    variance_degree = float(degree_second[0] - degree_mean[0] * degree_mean[0])
    covariance = cross - mean * degree_mean[0]
    return {
        "means": mean,
        "variances": variance,
        "degree_covariances": covariance,
        "mean_degree": float(degree_mean[0]),
        "variance_degree": variance_degree,
        "maximum_forward_error": float(maximum_error[0]),
        "log_normalizer": float(normalizer[0]),
        "log_normalizer_error": float(abs(normalizer[0] - expected_log_normalizer)),
        "transition_work": int(work[0]),
    }


def fused_one_channel_moments(
    offsets,
    states,
    structural_gain,
    additive,
    path_degree,
    path_target,
    inverse_two_variance: float,
    coefficient: float,
):
    """Fuse the forward law and one-channel raw moments in one recurrence."""

    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    states = np.ascontiguousarray(states, dtype=np.int64)
    gain = np.ascontiguousarray(structural_gain, dtype=np.float64)
    score = np.ascontiguousarray(additive, dtype=np.float64)
    degree = np.ascontiguousarray(path_degree, dtype=np.float64)
    target = np.ascontiguousarray(path_target, dtype=np.float64)
    count = int(offsets.size - 1)
    total = int(offsets[-1])
    if (
        count <= 0
        or offsets[0] != 0
        or np.any(offsets[1:] <= offsets[:-1])
        or states.shape != (total,)
        or gain.shape != (total,)
        or score.shape != (total,)
        or degree.shape != (count + 1,)
        or target.shape != degree.shape
    ):
        raise ValueError("invalid fused one-channel moment arrays")
    # The base forward vector is an internal semiring coordinate and has no
    # downstream consumer in spectral calibration.  A one-element sentinel
    # satisfies the native ABI while preventing an O(total-state) allocation.
    forward_sentinel = np.empty(1, dtype=np.float64)
    outputs = [np.empty(1, dtype=np.float64) for _ in range(6)]
    work = np.empty(1, dtype=np.int64)
    status = int(
        _load_fused_moment_function()(
            count,
            offsets,
            states,
            gain,
            score,
            degree,
            target,
            float(inverse_two_variance),
            float(coefficient),
            0,
            forward_sentinel,
            *outputs,
            work,
        )
    )
    if status:
        raise ArithmeticError(f"clean fused one-channel kernel returned {status}")
    mean, second, cross, degree_mean, degree_second, normalizer = (
        float(value[0]) for value in outputs
    )
    return {
        "mean": mean,
        "second": second,
        "cross": cross,
        "degree_mean": degree_mean,
        "degree_second": degree_second,
        "log_normalizer": normalizer,
        "transition_work": int(work[0]),
    }


__all__ = (
    "fused_one_channel_moments",
    "ragged_forward",
    "scalar_chain_moments",
)
