"""Counting-measure retraction from a full packed chart to an admitted interval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape


@dataclass(frozen=True)
class RetractionShadow:
    state_offsets: np.ndarray
    additive: np.ndarray
    diagnostics: Mapping[str, object]


def _exact_segment_means(values, starts, stops):
    data = np.asarray(values, dtype=np.float64)
    begin = np.asarray(starts, dtype=np.int64)
    end = np.asarray(stops, dtype=np.int64)
    return np.fromiter(
        (
            float(np.mean(data[int(start) : int(stop)]))
            for start, stop in zip(begin, end)
        ),
        dtype=np.float64,
        count=begin.size,
    )


def counting_measure_retraction_shadow(
    full: PackedRelocationTape,
    admitted: PackedRelocationTape,
    full_spectral: PackedRelocationTape,
    admitted_spectral: PackedRelocationTape,
    full_scale: float,
    admitted_scale: float,
) -> RetractionShadow:
    """Conditional spectral residual after retracting omitted prefix states.

    Boundary states receive the exact counting-fibre mean of every full-chart
    state retracted to that boundary.  Interior states retain their direct score.
    The result is a sufficient-statistic column on the admitted support, not a
    post-generation correction.
    """

    full_negative = np.diff(full.negative_offsets).astype(np.int64)
    full_positive = np.diff(full.positive_offsets).astype(np.int64)
    admitted_negative = np.diff(admitted.negative_offsets).astype(np.int64)
    admitted_positive = np.diff(admitted.positive_offsets).astype(np.int64)
    full_lower, full_upper = -full_negative, full_positive
    admitted_lower, admitted_upper = -admitted_negative, admitted_positive
    bundle = np.repeat(
        np.arange(admitted.bundle_count, dtype=np.int64),
        np.diff(admitted.state_offsets),
    )
    direct_position = (
        full.state_offsets[bundle]
        + admitted.state_value
        - full_lower[bundle]
    ).astype(np.int64)
    mean_gain = np.asarray(
        full_spectral.structural_gain[direct_position], dtype=np.float64
    ).copy()
    lower_position = admitted.state_offsets[:-1]
    upper_position = admitted.state_offsets[1:] - 1
    lower_start = full.state_offsets[:-1]
    lower_stop = full.state_offsets[:-1] + admitted_lower - full_lower + 1
    upper_start = full.state_offsets[:-1] + admitted_upper - full_lower
    upper_stop = full.state_offsets[1:]
    singleton = admitted_lower == admitted_upper
    non_singleton = ~singleton
    if np.any(non_singleton):
        mean_gain[lower_position[non_singleton]] = _exact_segment_means(
            full_spectral.structural_gain,
            lower_start[non_singleton],
            lower_stop[non_singleton],
        )
        mean_gain[upper_position[non_singleton]] = _exact_segment_means(
            full_spectral.structural_gain,
            upper_start[non_singleton],
            upper_stop[non_singleton],
        )
    if np.any(singleton):
        mean_gain[lower_position[singleton]] = _exact_segment_means(
            full_spectral.structural_gain,
            full.state_offsets[:-1][singleton],
            full.state_offsets[1:][singleton],
        )
    shadow = (
        mean_gain / float(full_scale)
        - np.asarray(admitted_spectral.structural_gain, dtype=np.float64)
        / float(admitted_scale)
    )
    lower_fibre = admitted_lower - full_lower + 1
    upper_fibre = full_upper - admitted_upper + 1
    singleton_fibre = np.diff(full.state_offsets)
    fibre_maximum = np.where(
        singleton, singleton_fibre, np.maximum(lower_fibre, upper_fibre)
    )
    nontrivial = np.where(
        singleton,
        singleton_fibre > 1,
        (lower_fibre > 1).astype(np.int64) + (upper_fibre > 1).astype(np.int64),
    )
    distance = np.maximum(admitted_lower - full_lower, full_upper - admitted_upper)
    return RetractionShadow(
        state_offsets=np.ascontiguousarray(admitted.state_offsets, dtype=np.int64),
        additive=np.ascontiguousarray(shadow, dtype=np.float64),
        diagnostics={
            "full_state_count": int(full.state_value.size),
            "admitted_state_count": int(admitted.state_value.size),
            "omitted_state_count": int(full.state_value.size - admitted.state_value.size),
            "nontrivial_fibre_count": int(np.sum(nontrivial, dtype=np.int64)),
            "maximum_fibre_size": int(np.max(fibre_maximum, initial=0)),
            "maximum_retraction_distance": int(np.max(distance, initial=0)),
            "complete_dyad_enumeration": False,
            "complete_wedge_enumeration": False,
        },
    )


__all__ = ("RetractionShadow", "counting_measure_retraction_shadow")
