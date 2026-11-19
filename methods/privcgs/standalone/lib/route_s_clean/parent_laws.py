"""Shared-parent exponential-family projections on a packed degree chain."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .chain_moments import BivariateChainMoments, bivariate_chain_moments
from .contracts import PackedRelocationTape
from .finite_chain import PackedGaussianMessages, build_gaussian_messages
from .endpoint_laws import fisher_covariance


@dataclass(frozen=True)
class MomentSubstitutionParent:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    closure_parameter: float
    degree_parameter: float
    moments: BivariateChainMoments
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class HorizontalAntiClosureParent:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    covariance: np.ndarray
    natural_parameter: np.ndarray
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class HamiltonianShadowParent:
    messages: PackedGaussianMessages
    quadratic_coefficient: float
    covariance: np.ndarray
    natural_parameter: np.ndarray
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _NewtonResult:
    point: np.ndarray
    success: bool
    function_evaluations: int
    iterations: int
    backtracking_evaluations: int
    termination_reason: str


def certified_fisher_newton(evaluate, target_closure: float, target_degree: float):
    """Parameter-free dyadic Newton solver with a floating-point certificate."""

    machine = np.finfo(np.float64)
    maximum_iterations = int(machine.nmant)
    maximum_backtracking = int(machine.nmant)
    point = np.zeros(2, dtype=np.float64)
    function_evaluations = 0
    backtracking_evaluations = 0

    def call(value):
        nonlocal function_evaluations
        function_evaluations += 1
        return evaluate(value)

    current = call(point)
    for iteration in range(maximum_iterations + 1):
        residual, jacobian, _messages, _coefficient, moments = current
        condition = float(np.linalg.cond(jacobian))
        moment_scale = float(
            max(
                1.0,
                abs(float(target_closure)),
                abs(float(target_degree)),
                abs(float(moments.mean_structural_score)),
                abs(float(moments.mean_degree_energy)),
            )
        )
        backward_bound = float(
            1024.0 * machine.eps * max(1.0, condition) * moment_scale
        )
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm == 0.0:
            return _NewtonResult(
                point.copy(), True, function_evaluations, iteration,
                backtracking_evaluations, "exact_zero_residual",
            )
        step = np.linalg.solve(jacobian, -np.asarray(residual, dtype=np.float64))
        accepted = False
        step_scale = 1.0
        for _ in range(maximum_backtracking + 1):
            candidate_point = point + step_scale * step
            if np.array_equal(candidate_point, point):
                if residual_norm <= backward_bound:
                    return _NewtonResult(
                        point.copy(), True, function_evaluations, iteration,
                        backtracking_evaluations,
                        "representable_parameter_step_stagnation",
                    )
                break
            candidate = call(candidate_point)
            candidate_norm = float(np.max(np.abs(candidate[0])))
            if math.isfinite(candidate_norm) and candidate_norm < residual_norm:
                point = np.asarray(candidate_point, dtype=np.float64)
                current = candidate
                accepted = True
                break
            if step_scale == 1.0 and residual_norm <= backward_bound:
                return _NewtonResult(
                    point.copy(), True, function_evaluations, iteration,
                    backtracking_evaluations,
                    "representable_full_newton_residual_stagnation",
                )
            backtracking_evaluations += 1
            step_scale *= 0.5
        if not accepted:
            break
    return _NewtonResult(
        point.copy(), False, function_evaluations, maximum_iterations,
        backtracking_evaluations, "iteration_or_globalization_exhaustion",
    )


def closure_degree_moment_substitution_parent(
    path,
    tape: PackedRelocationTape,
    base_messages: PackedGaussianMessages,
    base_coefficient: float,
    reference_messages: PackedGaussianMessages,
    reference_coefficient: float,
    closure_additive,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
) -> MomentSubstitutionParent:
    """Match reference closure drift while preserving base degree energy.

    In full Fisher rank this is the unique two-coordinate exponential-family
    information projection certified by its Jacobian.  If a statistic is exactly
    constant on the admitted support and the zero parameter already matches both
    targets exactly, the canonical zero parameter represents the same unique law
    in the Fisher quotient even though the redundant parameter is unidentified.
    The support and all sufficient statistics are fixed before solving; no graph
    metric or post-generation state is read.
    """

    closure = np.ascontiguousarray(closure_additive, dtype=np.float64)
    reference_moments = bivariate_chain_moments(
        path,
        reference_messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        closure,
        quadratic_coefficient=reference_coefficient,
    )
    base_moments = bivariate_chain_moments(
        path,
        base_messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        closure,
        quadratic_coefficient=base_coefficient,
    )
    target_closure = float(reference_moments.mean_structural_score)
    target_degree = float(base_moments.mean_degree_energy)
    cache = {}

    def evaluate(parameters):
        key = tuple(float(value) for value in parameters)
        if key not in cache:
            gain = np.ascontiguousarray(
                base_messages.structural_gain + key[0] * closure,
                dtype=np.float64,
            )
            coefficient = float(base_coefficient + key[1])
            messages = build_gaussian_messages(
                path,
                replace(tape, structural_gain=gain),
                source_degree,
                target_endpoint_mass,
                degree_noise_std,
                quadratic_coefficient=coefficient,
                structural_temperature=1.0,
            )
            moments = bivariate_chain_moments(
                path,
                messages,
                source_degree,
                target_endpoint_mass,
                degree_noise_std,
                closure,
                quadratic_coefficient=coefficient,
            )
            covariance = float(moments.covariance)
            residual = np.asarray(
                [
                    moments.mean_structural_score - target_closure,
                    moments.mean_degree_energy - target_degree,
                ],
                dtype=np.float64,
            )
            jacobian = np.asarray(
                [
                    [moments.variance_structural_score, -covariance],
                    [covariance, -moments.variance_degree_energy],
                ],
                dtype=np.float64,
            )
            cache[key] = residual, jacobian, messages, coefficient, moments
        return cache[key]

    solution = certified_fisher_newton(evaluate, target_closure, target_degree)
    residual, jacobian, messages, coefficient, moments = evaluate(solution.point)
    covariance_matrix = np.asarray(
        [
            [moments.variance_structural_score, moments.covariance],
            [moments.covariance, moments.variance_degree_energy],
        ],
        dtype=np.float64,
    )
    condition = float(np.linalg.cond(jacobian))
    moment_scale = float(
        max(
            1.0,
            abs(target_closure),
            abs(target_degree),
            abs(moments.mean_structural_score),
            abs(moments.mean_degree_energy),
        )
    )
    backward_bound = float(
        1024.0 * np.finfo(np.float64).eps * max(1.0, condition) * moment_scale
    )
    maximum_residual = float(np.max(np.abs(residual)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance_matrix)))
    rank_deficient_exact_solution = bool(
        solution.success
        and minimum_eigenvalue == 0.0
        and np.array_equal(residual, np.zeros(2, dtype=np.float64))
        and np.array_equal(solution.point, np.zeros(2, dtype=np.float64))
    )
    if (
        not solution.success
        or (minimum_eigenvalue <= 0.0 and not rank_deficient_exact_solution)
        or (
            not rank_deficient_exact_solution
            and maximum_residual > backward_bound
        )
    ):
        raise ArithmeticError("closure-degree dual moment certificate failed")
    diagnostics = {
        "solver_success": True,
        "solver_method": "certified_fisher_newton_dyadic_residual_globalization",
        "solver_function_evaluations": int(solution.function_evaluations),
        "solver_iterations": int(solution.iterations),
        "solver_backtracking_evaluations": int(
            solution.backtracking_evaluations
        ),
        "solver_termination_reason": solution.termination_reason,
        "target_closure_drift": target_closure,
        "target_degree_energy": target_degree,
        "closure_moment_residual": float(residual[0]),
        "degree_energy_residual": float(residual[1]),
        "maximum_absolute_moment_residual": maximum_residual,
        "moment_backward_error_bound": backward_bound,
        "fisher_minimum_eigenvalue": minimum_eigenvalue,
        "jacobian_condition_number": condition,
        "new_fitted_parameter_count": 0,
        "complete_dyad_enumeration": False,
        "complete_wedge_enumeration": False,
    }
    if rank_deficient_exact_solution:
        diagnostics.update(
            {
                "fisher_quotient_zero_parameter_branch": True,
                "redundant_parameter_direction_count": int(
                    2 - np.linalg.matrix_rank(covariance_matrix)
                ),
            }
        )
    return MomentSubstitutionParent(
        messages=messages,
        quadratic_coefficient=float(coefficient),
        closure_parameter=float(solution.point[0]),
        degree_parameter=float(solution.point[1]),
        moments=moments,
        diagnostics=diagnostics,
    )


def horizontal_anti_closure_parent(
    path,
    tape: PackedRelocationTape,
    parent: MomentSubstitutionParent,
    palm_additive,
    closure_additive,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
) -> HorizontalAntiClosureParent:
    """Unit-Fisher Palm innovation plus the opposing closure innovation."""

    palm = np.ascontiguousarray(palm_additive, dtype=np.float64)
    closure = np.ascontiguousarray(closure_additive, dtype=np.float64)
    covariance, _ = fisher_covariance(
        path,
        parent.messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        parent.quadratic_coefficient,
        (palm, closure),
    )
    covariance = 0.5 * (covariance + covariance.T)
    covariance_scale = float(max(1.0, np.max(np.abs(covariance))))
    tolerance = float(1024.0 * np.finfo(np.float64).eps * covariance_scale)
    minimum = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum < -tolerance:
        raise ArithmeticError("horizontal Fisher covariance is not PSD")
    nuisance = np.asarray(covariance[1:, 1:], dtype=np.float64)
    nuisance_eigenvalues = np.linalg.eigvalsh(nuisance)
    nuisance_minimum = float(np.min(nuisance_eigenvalues))
    rank_adaptive = bool(nuisance_minimum <= tolerance)
    if nuisance_minimum < -tolerance:
        raise ArithmeticError("horizontal closure-degree nuisance is not PSD")
    if not rank_adaptive:
        beta = np.linalg.solve(nuisance, covariance[1:, 0])
    else:
        singular = np.linalg.svd(nuisance, compute_uv=False)
        maximum_singular = float(np.max(singular, initial=0.0))
        inverse = np.linalg.pinv(
            nuisance,
            rcond=(
                tolerance / maximum_singular
                if maximum_singular > 0.0
                else np.finfo(np.float64).eps
            ),
        )
        beta = inverse @ covariance[1:, 0]
    palm_vector = np.asarray([1.0, -beta[0], -beta[1]], dtype=np.float64)
    palm_variance = float(palm_vector @ covariance @ palm_vector)
    if palm_variance < -tolerance:
        raise ArithmeticError("horizontal Palm efficient variance is negative")
    palm_rank_positive = bool(palm_variance > tolerance)
    if not rank_adaptive and not palm_rank_positive:
        raise ArithmeticError("horizontal Palm efficient rank is zero")
    degree_variance = float(covariance[2, 2])
    if degree_variance < -tolerance:
        raise ArithmeticError("horizontal degree nuisance variance is negative")
    if not rank_adaptive and not degree_variance > tolerance:
        raise ArithmeticError("horizontal degree nuisance has zero rank")
    closure_alpha = (
        float(covariance[1, 2] / degree_variance)
        if degree_variance > tolerance
        else 0.0
    )
    closure_vector = np.asarray([0.0, 1.0, -closure_alpha], dtype=np.float64)
    closure_variance = float(closure_vector @ covariance @ closure_vector)
    if closure_variance < -tolerance:
        raise ArithmeticError("horizontal closure residual variance is negative")
    closure_rank_positive = bool(closure_variance > tolerance)
    if not rank_adaptive and not closure_rank_positive:
        raise ArithmeticError("horizontal closure residual rank is zero")
    palm_score = (
        palm_vector / float(math.sqrt(palm_variance))
        if palm_rank_positive
        else np.zeros(3, dtype=np.float64)
    )
    anti_closure_score = (
        -closure_vector / float(math.sqrt(closure_variance))
        if closure_rank_positive
        else np.zeros(3, dtype=np.float64)
    )
    natural = np.asarray(palm_score + anti_closure_score, dtype=np.float64)
    additive = (
        float(natural[0]) * palm
        + float(natural[1]) * closure
    )
    gain = np.ascontiguousarray(
        parent.messages.structural_gain + additive, dtype=np.float64
    )
    coefficient = float(parent.quadratic_coefficient - natural[2])
    if palm_rank_positive or closure_rank_positive:
        messages = build_gaussian_messages(
            path,
            replace(tape, structural_gain=gain),
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
            quadratic_coefficient=coefficient,
            structural_temperature=1.0,
        )
    else:
        messages = parent.messages
    return HorizontalAntiClosureParent(
        messages=messages,
        quadratic_coefficient=coefficient,
        covariance=np.ascontiguousarray(covariance),
        natural_parameter=natural,
        diagnostics={
            "fisher_minimum_eigenvalue": minimum,
            "fisher_machine_tolerance": tolerance,
            "palm_closure_projection_coefficient": float(beta[0]),
            "palm_degree_projection_coefficient": float(beta[1]),
            "closure_degree_projection_coefficient": closure_alpha,
            "palm_residual_variance": palm_variance,
            "closure_residual_variance": closure_variance,
            "palm_unit_fisher_variance": float(palm_score @ covariance @ palm_score),
            "closure_unit_fisher_variance": float(
                (-anti_closure_score) @ covariance @ (-anti_closure_score)
            ),
            "anti_closure_orientation_is_negative": True,
            "new_fitted_parameter_count": 0,
            **(
                {
                    "fisher_quotient_rank_adaptive_branch": True,
                    "palm_score_rank_positive": palm_rank_positive,
                    "closure_score_rank_positive": closure_rank_positive,
                    "nuisance_rank": int(
                        np.count_nonzero(nuisance_eigenvalues > tolerance)
                    ),
                }
                if rank_adaptive
                else {}
            ),
        },
    )


def hamiltonian_shadow_parent(
    path,
    tape: PackedRelocationTape,
    parent: HorizontalAntiClosureParent,
    palm_additive,
    closure_additive,
    shadow_additive,
    source_degree,
    target_endpoint_mass,
    degree_noise_std: float,
) -> HamiltonianShadowParent:
    """Natural-amplitude Rao--Blackwell shadow of omitted completion states."""

    palm = np.ascontiguousarray(palm_additive, dtype=np.float64)
    closure = np.ascontiguousarray(closure_additive, dtype=np.float64)
    shadow = np.ascontiguousarray(shadow_additive, dtype=np.float64)
    covariance, _ = fisher_covariance(
        path,
        parent.messages,
        source_degree,
        target_endpoint_mass,
        degree_noise_std,
        parent.quadratic_coefficient,
        (palm, closure, shadow),
    )
    nuisance_indices = np.asarray([0, 1, 3], dtype=np.int64)
    nuisance = covariance[np.ix_(nuisance_indices, nuisance_indices)]
    cross = covariance[nuisance_indices, 2]
    singular = np.linalg.svd(nuisance, compute_uv=False)
    maximum = float(np.max(singular, initial=0.0))
    tolerance = float(
        np.finfo(np.float64).eps
        * max(int(nuisance.shape[0]), 2)
        * max(maximum, 1.0)
    )
    nuisance_rank = int(np.sum(singular > tolerance))
    inverse = np.linalg.pinv(
        nuisance,
        rcond=tolerance / maximum if maximum > 0.0 else np.finfo(np.float64).eps,
    )
    beta = inverse @ cross
    residual = np.asarray([-beta[0], -beta[1], 1.0, -beta[2]], dtype=np.float64)
    residual_variance = float(residual @ covariance @ residual)
    joint_singular = np.linalg.svd(covariance, compute_uv=False)
    joint_maximum = float(np.max(joint_singular, initial=0.0))
    joint_tolerance = float(
        np.finfo(np.float64).eps
        * max(int(covariance.shape[0]), 2)
        * max(joint_maximum, 1.0)
    )
    joint_rank = int(np.sum(joint_singular > joint_tolerance))
    positive = bool(joint_rank > nuisance_rank and residual_variance > joint_tolerance)
    if positive:
        natural = residual
        additive = (
            float(natural[0]) * palm
            + float(natural[1]) * closure
            + float(natural[2]) * shadow
        )
        gain = np.ascontiguousarray(
            parent.messages.structural_gain + additive, dtype=np.float64
        )
        coefficient = float(parent.quadratic_coefficient - natural[3])
        messages = build_gaussian_messages(
            path,
            replace(tape, structural_gain=gain),
            source_degree,
            target_endpoint_mass,
            degree_noise_std,
            quadratic_coefficient=coefficient,
            structural_temperature=1.0,
        )
    else:
        natural = np.zeros(4, dtype=np.float64)
        coefficient = float(parent.quadratic_coefficient)
        messages = parent.messages
    return HamiltonianShadowParent(
        messages=messages,
        quadratic_coefficient=coefficient,
        covariance=np.ascontiguousarray(covariance),
        natural_parameter=np.ascontiguousarray(natural),
        diagnostics={
            "nuisance_rank": nuisance_rank,
            "joint_rank": joint_rank,
            "rank_positive": positive,
            "residual_variance": residual_variance if positive else 0.0,
            "projection_coefficients": beta.tolist(),
            "natural_amplitude_preserved": True,
            "unit_fisher_renormalization_applied": False,
            "new_fitted_parameter_count": 0,
        },
    )


__all__ = (
    "HamiltonianShadowParent",
    "HorizontalAntiClosureParent",
    "MomentSubstitutionParent",
    "certified_fisher_newton",
    "closure_degree_moment_substitution_parent",
    "hamiltonian_shadow_parent",
    "horizontal_anti_closure_parent",
)
