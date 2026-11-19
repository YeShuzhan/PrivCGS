"""RKSS1 with exact per-row power-path reuse."""

from __future__ import annotations

from pathlib import Path

import gfl_mpe_rkss1_v1 as implementation


SOURCE = Path(__file__).with_suffix(".cpp")
CANDIDATE_NAME = "exact-parallel-row-kernel-and-power-path-state-v1"
THREAD_COUNT = 8


def _activate() -> None:
    implementation.SOURCE = SOURCE
    implementation.CANDIDATE_NAME = CANDIDATE_NAME
    implementation.THREAD_COUNT = THREAD_COUNT


def reset_plan() -> None:
    _activate()
    implementation._LOADED = None
    implementation.reset_plan()


def plan_diagnostics():
    _activate()
    return implementation.plan_diagnostics()


def expected_log_counts(*args, **kwargs):
    _activate()
    implementation.THREAD_COUNT = THREAD_COUNT
    return implementation.expected_log_counts(*args, **kwargs)


__all__ = ("expected_log_counts", "plan_diagnostics", "reset_plan")
