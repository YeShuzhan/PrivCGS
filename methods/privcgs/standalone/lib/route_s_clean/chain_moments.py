"""Finite-chain moment propagation and nuisance-orthogonal spectral scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .contracts import PackedRelocationTape, RecoveredGraphLaw
from .degree_chain_native import fused_one_channel_moments, scalar_chain_moments
from .finite_chain import PackedGaussianMessages, build_gaussian_messages
from .graph_carrier import CSRGraph
from .native import ragged_prefix
from .palm_coordinates import (
    HorizontalPalmScoreColumns,
    PalmScoreColumns,
    compile_palm_score_columns,
    rebind_palm_score_columns,
)
from .structural_columns import SpectralColumns, compile_spectral_columns


@dataclass(frozen=True)
class BivariateChainMoments:
    mean_structural_score: float
    mean_degree_energy: float
    variance_structural_score: float
    variance_degree_energy: float
    covariance: float
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class EfficientSpectralScore:
    messages: PackedGaussianMessages | None
    quadratic_coefficient: float
    residual_scale: float
    projection_coefficient: float
    spectral_columns: SpectralColumns | None
    scored_tape: PackedRelocationTape | None
    moments: BivariateChainMoments | None
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class EfficientExposureScore:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    residual_scale: float
    projection_coefficient: float
    palm_columns: PalmScoreColumns | HorizontalPalmScoreColumns
    scored_tape: PackedRelocationTape
    final_tape: PackedRelocationTape
    moments: BivariateChainMoments
    diagnostics: Mapping[str, object]


def _views(flat, offsets):
    return tuple(
        np.asarray(flat[int(offsets[index]) : int(offsets[index + 1])])
        for index in range(int(offsets.size - 1))
    )


def _logsumexp(values, axis=None):
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(array - maximum), axis=axis, keepdims=True)
    )
    return float(result.reshape(-1)[0]) if axis is None else np.squeeze(result, axis)


def bivariate_chain_moments(
    path,
    messages: PackedGaussianMessages,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    structural_additive,
    *,
    quadratic_coefficient: float = 1.0,
) -> BivariateChainMoments:
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    inverse_two_variance = 1.0 / (2.0 * float(degree_noise_std) ** 2)
    coefficient = float(quadratic_coefficient)
    native = scalar_chain_moments(
        messages.state_offsets,
        messages.state_value,
        messages.forward_log_message,
        messages.structural_gain,
        (np.ascontiguousarray(structural_additive, dtype=np.float64),),
        degree[order],
        target[order],
        inverse_two_variance,
        coefficient,
        messages.log_normalizer,
    )
    return BivariateChainMoments(
        mean_structural_score=float(native["means"][0]),
        mean_degree_energy=float(native["mean_degree"]),
        variance_structural_score=float(native["variances"][0]),
        variance_degree_energy=float(native["variance_degree"]),
        covariance=float(native["degree_covariances"][0]),
        diagnostics={
            "maximum_forward_message_error": native["maximum_forward_error"],
            "base_log_normalizer_error": native["log_normalizer_error"],
            "native_transition_work": native["transition_work"],
        },
    )


def _two_pass_spectral_score_messages(
    law: RecoveredGraphLaw,
    path,
    tape: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    feature_cache=None,
    precompiled_spectral_columns: SpectralColumns | None = None,
) -> EfficientSpectralScore:
    zero_tape = replace(
        tape,
        structural_gain=np.zeros(tape.state_value.size, dtype=np.float64),
    )
    base_messages = build_gaussian_messages(
        path,
        zero_tape,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=1.0,
        structural_temperature=1.0,
    )
    spectral = (
        compile_spectral_columns(law, tape, feature_cache=feature_cache)
        if precompiled_spectral_columns is None
        else precompiled_spectral_columns
    )
    moments = bivariate_chain_moments(
        path,
        base_messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        spectral.additive_gain,
    )
    variance_degree = float(moments.variance_degree_energy)
    covariance = float(moments.covariance)
    deterministic_zero_chart = bool(
        np.all(np.diff(tape.state_offsets) == 1)
        and np.all(tape.state_value == 0)
        and tape.negative_energy.size == 0
        and tape.positive_energy.size == 0
        and np.all(tape.structural_gain == 0.0)
        and all(
            math.isfinite(value)
            for value in (
                moments.mean_structural_score,
                moments.mean_degree_energy,
                moments.variance_structural_score,
                moments.variance_degree_energy,
                moments.covariance,
            )
        )
    )
    if deterministic_zero_chart:
        # This support contains exactly one chain path and both sufficient
        # statistics are constant zero.  Its Fisher tangent space is therefore
        # the zero quotient.  Represent the absent unit score by the canonical
        # zero natural parameter; ``1.0`` is only the neutral denominator for
        # downstream zero columns and never scales a nonzero statistic.
        return EfficientSpectralScore(
            messages=base_messages,
            quadratic_coefficient=1.0,
            residual_scale=1.0,
            projection_coefficient=0.0,
            spectral_columns=spectral,
            scored_tape=zero_tape,
            moments=moments,
            diagnostics={
                **dict(spectral.statistics),
                **dict(moments.diagnostics),
                "mean_structural_score": moments.mean_structural_score,
                "mean_degree_energy": moments.mean_degree_energy,
                "variance_structural_score": 0.0,
                "variance_degree_energy": 0.0,
                "covariance_structural_degree_energy": 0.0,
                "projection_coefficient": 0.0,
                "residual_variance": 0.0,
                "residual_scale": 1.0,
                "signed_degree_energy_coefficient": 1.0,
                "base_log_normalizer": float(base_messages.log_normalizer),
                "tilted_log_normalizer": float(base_messages.log_normalizer),
                "log_normalizer_change": 0.0,
                "fisher_quotient_zero_score_branch": True,
                "deterministic_chain_path_count": 1,
                "new_fitted_parameter_count": 0,
                "dense_covariance_materialized": False,
            },
        )
    if not math.isfinite(variance_degree) or variance_degree <= 0.0:
        raise ArithmeticError("degree-energy nuisance variance is not positive")
    alpha = float(covariance / variance_degree)
    residual_variance = float(
        moments.variance_structural_score
        + alpha * alpha * variance_degree
        - 2.0 * alpha * covariance
    )
    if not math.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ArithmeticError("efficient spectral residual variance is not positive")
    scale = float(math.sqrt(residual_variance))
    coefficient = float(1.0 + alpha / scale)
    scaled_negative = np.fromiter(
        (float(float(value) / scale) for value in spectral.negative_delta),
        dtype=np.float64,
        count=spectral.negative_delta.size,
    )
    scaled_positive = np.fromiter(
        (float(float(value) / scale) for value in spectral.positive_delta),
        dtype=np.float64,
        count=spectral.positive_delta.size,
    )
    scaled_gain = ragged_prefix(
        tape.state_offsets,
        tape.negative_offsets,
        tape.positive_offsets,
        scaled_negative,
        scaled_positive,
    )
    scored_tape = replace(
        tape,
        negative_energy=np.ascontiguousarray(scaled_negative, dtype=np.float64),
        positive_energy=np.ascontiguousarray(scaled_positive, dtype=np.float64),
        structural_gain=np.ascontiguousarray(scaled_gain, dtype=np.float64),
    )
    messages = build_gaussian_messages(
        path,
        scored_tape,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=coefficient,
        structural_temperature=1.0,
    )
    return EfficientSpectralScore(
        messages=messages,
        quadratic_coefficient=coefficient,
        residual_scale=scale,
        projection_coefficient=alpha,
        spectral_columns=spectral,
        scored_tape=scored_tape,
        moments=moments,
        diagnostics={
            **dict(spectral.statistics),
            **dict(moments.diagnostics),
            "mean_structural_score": moments.mean_structural_score,
            "mean_degree_energy": moments.mean_degree_energy,
            "variance_structural_score": moments.variance_structural_score,
            "variance_degree_energy": moments.variance_degree_energy,
            "covariance_structural_degree_energy": moments.covariance,
            "projection_coefficient": alpha,
            "residual_variance": residual_variance,
            "residual_scale": scale,
            "signed_degree_energy_coefficient": coefficient,
            "base_log_normalizer": float(base_messages.log_normalizer),
            "tilted_log_normalizer": float(messages.log_normalizer),
            "log_normalizer_change": float(
                messages.log_normalizer - base_messages.log_normalizer
            ),
            "new_fitted_parameter_count": 0,
            "dense_covariance_materialized": False,
        },
    )


def efficient_spectral_score_messages(
    law: RecoveredGraphLaw,
    path,
    tape: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    feature_cache=None,
    precompiled_spectral_columns: SpectralColumns | None = None,
    _calibration_only: bool = False,
    _retain_spectral_columns: bool = True,
) -> EfficientSpectralScore:
    """Compile the exact efficient score from its minimal consumed state.

    The base Gaussian forward recurrence and its one-channel score moments form
    one expectation-semiring recurrence.  Consumers that need only the fitted
    Fisher scalars can also omit the independent tilted forward law.  Both
    reductions preserve the original scalar loop and binary64 accumulation
    order; the one-path zero quotient retains its frozen sentinel behavior.
    """

    deterministic_zero_chart = bool(
        np.all(np.diff(tape.state_offsets) == 1)
        and np.all(tape.state_value == 0)
        and tape.negative_energy.size == 0
        and tape.positive_energy.size == 0
        and np.all(tape.structural_gain == 0.0)
    )
    if deterministic_zero_chart:
        return _two_pass_spectral_score_messages(
            law,
            path,
            tape,
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
            feature_cache=feature_cache,
            precompiled_spectral_columns=precompiled_spectral_columns,
        )

    zero_tape = replace(
        tape,
        structural_gain=np.zeros(tape.state_value.size, dtype=np.float64),
    )
    spectral = (
        compile_spectral_columns(law, tape, feature_cache=feature_cache)
        if precompiled_spectral_columns is None
        else precompiled_spectral_columns
    )
    order = np.asarray(path, dtype=np.int64)
    degree = np.asarray(source_degree, dtype=np.float64)
    target = np.asarray(target_endpoint_mass, dtype=np.float64)
    inverse_two_variance = float(1.0 / (2.0 * float(degree_noise_std) ** 2))
    raw = fused_one_channel_moments(
        zero_tape.state_offsets,
        zero_tape.state_value,
        zero_tape.structural_gain,
        spectral.additive_gain,
        degree[order],
        target[order],
        inverse_two_variance,
        inverse_two_variance,
    )
    mean_score = float(raw["mean"])
    mean_degree = float(raw["degree_mean"])
    moments = BivariateChainMoments(
        mean_structural_score=mean_score,
        mean_degree_energy=mean_degree,
        variance_structural_score=float(
            float(raw["second"]) - mean_score * mean_score
        ),
        variance_degree_energy=float(
            float(raw["degree_second"]) - mean_degree * mean_degree
        ),
        covariance=float(float(raw["cross"]) - mean_score * mean_degree),
        diagnostics={
            "maximum_forward_message_error": 0.0,
            "base_log_normalizer_error": 0.0,
            "native_transition_work": int(raw["transition_work"]),
            "expectation_semiring_forward_moment_fusion": True,
        },
    )
    variance_degree = float(moments.variance_degree_energy)
    covariance = float(moments.covariance)
    if not math.isfinite(variance_degree) or variance_degree <= 0.0:
        raise ArithmeticError("degree-energy nuisance variance is not positive")
    alpha = float(covariance / variance_degree)
    residual_variance = float(
        moments.variance_structural_score
        + alpha * alpha * variance_degree
        - 2.0 * alpha * covariance
    )
    if not math.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ArithmeticError("efficient spectral residual variance is not positive")
    scale = float(math.sqrt(residual_variance))
    coefficient = float(1.0 + alpha / scale)
    common_diagnostics = {
        **dict(spectral.statistics),
        **dict(moments.diagnostics),
        "mean_structural_score": moments.mean_structural_score,
        "mean_degree_energy": moments.mean_degree_energy,
        "variance_structural_score": moments.variance_structural_score,
        "variance_degree_energy": moments.variance_degree_energy,
        "covariance_structural_degree_energy": moments.covariance,
        "projection_coefficient": alpha,
        "residual_variance": residual_variance,
        "residual_scale": scale,
        "signed_degree_energy_coefficient": coefficient,
        "base_log_normalizer": float(raw["log_normalizer"]),
        "new_fitted_parameter_count": 0,
        "dense_covariance_materialized": False,
        "calibration_only_tilted_law_omitted": bool(_calibration_only),
    }
    if _calibration_only:
        return EfficientSpectralScore(
            messages=None,
            quadratic_coefficient=coefficient,
            residual_scale=scale,
            projection_coefficient=alpha,
            spectral_columns=spectral if _retain_spectral_columns else None,
            scored_tape=None,
            moments=None,
            diagnostics=common_diagnostics,
        )

    scaled_negative = np.fromiter(
        (float(float(value) / scale) for value in spectral.negative_delta),
        dtype=np.float64,
        count=spectral.negative_delta.size,
    )
    scaled_positive = np.fromiter(
        (float(float(value) / scale) for value in spectral.positive_delta),
        dtype=np.float64,
        count=spectral.positive_delta.size,
    )
    scaled_gain = ragged_prefix(
        tape.state_offsets,
        tape.negative_offsets,
        tape.positive_offsets,
        scaled_negative,
        scaled_positive,
    )
    scored_tape = replace(
        tape,
        negative_energy=np.ascontiguousarray(scaled_negative, dtype=np.float64),
        positive_energy=np.ascontiguousarray(scaled_positive, dtype=np.float64),
        structural_gain=np.ascontiguousarray(scaled_gain, dtype=np.float64),
    )
    messages = build_gaussian_messages(
        path,
        scored_tape,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=coefficient,
        structural_temperature=1.0,
    )
    return EfficientSpectralScore(
        messages=messages,
        quadratic_coefficient=coefficient,
        residual_scale=scale,
        projection_coefficient=alpha,
        spectral_columns=spectral,
        scored_tape=scored_tape,
        moments=moments,
        diagnostics={
            **common_diagnostics,
            "tilted_log_normalizer": float(messages.log_normalizer),
            "log_normalizer_change": float(
                messages.log_normalizer - float(raw["log_normalizer"])
            ),
        },
    )


def efficient_palm_exposure_messages(
    source: CSRGraph,
    law: RecoveredGraphLaw,
    path,
    tape: PackedRelocationTape,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
    *,
    public_seed: int,
    feature_cache=None,
    base_score: EfficientSpectralScore | None = None,
    coordinate_columns: PalmScoreColumns | HorizontalPalmScoreColumns | None = None,
    rebind_coordinate_columns: bool = True,
) -> EfficientExposureScore:
    """Add the Palm occupation innovation after degree-nuisance projection."""

    base = base_score
    if base is None:
        base = efficient_spectral_score_messages(
            law,
            path,
            tape,
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
            feature_cache=feature_cache,
        )
    # ``base`` was compiled on the identical tape and feature cache.  Reuse its
    # immutable spectral sufficient columns instead of scanning the dyads twice.
    spectral = base.spectral_columns
    scale = float(base.residual_scale)
    scored_negative = np.fromiter(
        (float(float(value) / scale) for value in spectral.negative_delta),
        dtype=np.float64,
        count=spectral.negative_delta.size,
    )
    scored_positive = np.fromiter(
        (float(float(value) / scale) for value in spectral.positive_delta),
        dtype=np.float64,
        count=spectral.positive_delta.size,
    )
    scored_gain = ragged_prefix(
        tape.state_offsets,
        tape.negative_offsets,
        tape.positive_offsets,
        scored_negative,
        scored_positive,
    )
    scored = replace(
        tape,
        negative_energy=scored_negative,
        positive_energy=scored_positive,
        structural_gain=scored_gain,
    )
    if coordinate_columns is None:
        columns = compile_palm_score_columns(
            source,
            scored,
            np.asarray(law.component_fiber, dtype=np.int64),
            int(public_seed),
        )
    elif rebind_coordinate_columns:
        columns = rebind_palm_score_columns(coordinate_columns, scored)
    else:
        columns = coordinate_columns
    # Keep the sufficient statistic in the same floating-point chart in which
    # it enters the finite law: first rebind the mark to the current energy,
    # then form the additive Radon--Nikodym column by subtraction.  This is
    # mathematically equal to the raw Palm delta, but also fixes the binary64
    # evaluation order used by the normalized chain.
    rebound_negative = np.fromiter(
        (
            float(float(base_value) + float(delta))
            for base_value, delta in zip(
                scored.negative_energy, columns.occupation_negative_delta
            )
        ),
        dtype=np.float64,
        count=scored.negative_energy.size,
    )
    rebound_positive = np.fromiter(
        (
            float(float(base_value) + float(delta))
            for base_value, delta in zip(
                scored.positive_energy, columns.occupation_positive_delta
            )
        ),
        dtype=np.float64,
        count=scored.positive_energy.size,
    )
    rebound_gain = ragged_prefix(
        tape.state_offsets,
        tape.negative_offsets,
        tape.positive_offsets,
        rebound_negative,
        rebound_positive,
    )
    occupation_additive = rebound_gain - scored_gain
    moments = bivariate_chain_moments(
        path,
        base.messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        occupation_additive,
        quadratic_coefficient=base.quadratic_coefficient,
    )
    variance_degree = float(moments.variance_degree_energy)
    variance_score = float(moments.variance_structural_score)
    covariance = float(moments.covariance)
    if variance_degree < 0.0 or not math.isfinite(variance_degree):
        raise ArithmeticError("degree-energy variance is invalid")
    alpha = float(covariance / variance_degree) if variance_degree > 0.0 else 0.0
    residual_variance = float(
        variance_score
        + alpha * alpha * variance_degree
        - 2.0 * alpha * covariance
    )
    if residual_variance < 0.0 or not math.isfinite(residual_variance):
        raise ArithmeticError("Palm exposure residual variance is invalid")
    exposure_scale = float(math.sqrt(residual_variance))
    if exposure_scale > 0.0:
        negative = np.fromiter(
            (
                float(float(base_value) + float(delta) / exposure_scale)
                for base_value, delta in zip(
                    scored.negative_energy,
                    rebound_negative - scored.negative_energy,
                )
            ),
            dtype=np.float64,
            count=scored.negative_energy.size,
        )
        positive = np.fromiter(
            (
                float(float(base_value) + float(delta) / exposure_scale)
                for base_value, delta in zip(
                    scored.positive_energy,
                    rebound_positive - scored.positive_energy,
                )
            ),
            dtype=np.float64,
            count=scored.positive_energy.size,
        )
        gain = ragged_prefix(
            tape.state_offsets,
            tape.negative_offsets,
            tape.positive_offsets,
            negative,
            positive,
        )
        final_tape = replace(
            scored,
            negative_energy=negative,
            positive_energy=positive,
            structural_gain=gain,
        )
        projected_coefficient = float(
            base.quadratic_coefficient + alpha / exposure_scale
        )
        projected_covariance = float(
            (covariance - alpha * variance_degree) / exposure_scale
        )
        unit_variance = float(residual_variance / (exposure_scale * exposure_scale))
    else:
        final_tape = scored
        projected_coefficient = float(base.quadratic_coefficient)
        projected_covariance = 0.0
        unit_variance = 0.0
    messages = build_gaussian_messages(
        path,
        final_tape,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        quadratic_coefficient=projected_coefficient,
        structural_temperature=1.0,
    )
    machine_bound = float(
        64.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(covariance / exposure_scale) if exposure_scale > 0.0 else 0.0,
            abs(alpha * variance_degree / exposure_scale)
            if exposure_scale > 0.0
            else 0.0,
        )
    )
    return EfficientExposureScore(
        messages=messages,
        quadratic_coefficient=projected_coefficient,
        residual_scale=exposure_scale,
        projection_coefficient=alpha,
        palm_columns=columns,
        scored_tape=scored,
        final_tape=final_tape,
        moments=moments,
        diagnostics={
            **dict(base.diagnostics),
            **dict(columns.statistics),
            **dict(moments.diagnostics),
            "projection_coefficient": alpha,
            "projection_zero_degree_branch": bool(variance_degree == 0.0),
            "efficient_residual_variance": residual_variance,
            "efficient_residual_scale": exposure_scale,
            "polar_zero_score_branch": bool(exposure_scale == 0.0),
            "projected_covariance_with_degree_energy": projected_covariance,
            "projected_covariance_machine_bound": machine_bound,
            "unit_fisher_variance": unit_variance,
            "base_degree_energy_coefficient": float(base.quadratic_coefficient),
            "projected_degree_energy_coefficient": projected_coefficient,
            "base_log_normalizer": float(base.messages.log_normalizer),
            "efficient_log_normalizer": float(messages.log_normalizer),
            "log_normalizer_change": float(
                messages.log_normalizer - base.messages.log_normalizer
            ),
            "new_fitted_parameter_count": 0,
            "finite_chain_normalization": True,
        },
    )


__all__ = (
    "BivariateChainMoments",
    "EfficientSpectralScore",
    "EfficientExposureScore",
    "bivariate_chain_moments",
    "efficient_spectral_score_messages",
    "efficient_palm_exposure_messages",
)
