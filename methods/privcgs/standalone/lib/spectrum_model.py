from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


def _maximum_entropy_initialization(
    support: np.ndarray, target_mean: float
) -> tuple[np.ndarray, float]:
    """Return the unique truncated exponential law with the required mean."""

    def law(parameter: float) -> tuple[np.ndarray, float]:
        exponent = parameter * support
        exponent -= float(np.max(exponent))
        weight = np.exp(exponent)
        probability = weight / float(np.sum(weight))
        return probability, float(np.dot(probability, support))

    lower = -100.0
    upper = 100.0
    for _ in range(256):
        middle = 0.5 * (lower + upper)
        _, mean = law(middle)
        if mean < target_mean:
            lower = middle
        else:
            upper = middle
    parameter = 0.5 * (lower + upper)
    probability, _ = law(parameter)
    return probability, parameter


def _mean_constrained_m_step(
    expected_probability: np.ndarray,
    support: np.ndarray,
    target_mean: float,
) -> tuple[np.ndarray, float]:
    """Exact M-step for sum(p)=1 and dot(k,p)=target_mean.

    If q is the normalized expected component count, the constrained maximizer
    has p_k=q_k/(1+t*(k-mu)).  Its mean equation is strictly decreasing in t.
    """

    q = np.asarray(expected_probability, dtype=np.float64)
    q /= float(np.sum(q))
    centered = support - float(target_mean)
    active = q > 0.0
    positive = active & (centered > 0.0)
    negative = active & (centered < 0.0)
    if not np.any(positive) or not np.any(negative):
        raise RuntimeError("released mean is not in the interior of the support")
    # Keep a scale-aware open-interval margin.  ``nextafter`` alone can round
    # 1+t*x back to zero when the extremal active coordinate is multiplied.
    open_margin = 64.0 * float(np.finfo(np.float64).eps)
    lower = (
        -1.0 / float(np.max(centered[positive])) * (1.0 - open_margin)
    )
    upper = (
        1.0 / float(np.max(-centered[negative])) * (1.0 - open_margin)
    )

    def equation(value: float) -> float:
        return float(np.sum(centered * q / (1.0 + value * centered)))

    lower_value = equation(lower)
    upper_value = equation(upper)
    if not (lower_value > 0.0 and upper_value < 0.0):
        raise RuntimeError(
            "active EM support does not bracket the released mean: "
            f"f(lower)={lower_value}, f(upper)={upper_value}"
        )
    for _ in range(128):
        middle = 0.5 * (lower + upper)
        if equation(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    tilt = 0.5 * (lower + upper)
    probability = q / (1.0 + tilt * centered)
    probability /= float(np.sum(probability))
    return probability, tilt


class ReleasedMixture:
    def __init__(
        self,
        observation: np.ndarray,
        sigma: float,
        support_maximum: int,
        half_bandwidth: int,
        target_mean: float,
    ) -> None:
        self.observation = np.asarray(observation, dtype=np.float64)
        self.sigma = float(sigma)
        self.support = np.arange(int(support_maximum) + 1, dtype=np.float64)
        self.target_mean = float(target_mean)
        self.center = np.floor(self.observation).astype(np.int64)
        self.offset = np.arange(
            -int(half_bandwidth), int(half_bandwidth) + 1, dtype=np.int64
        )

    def em_step(
        self, probability: np.ndarray, need_gradient: bool
    ) -> tuple[np.ndarray, float, np.ndarray | None, float]:
        component_count = self.support.size
        expected = np.zeros(component_count, dtype=np.float64)
        gradient = (
            np.zeros(component_count, dtype=np.float64) if need_gradient else None
        )
        log_probability = np.full(component_count, -np.inf, dtype=np.float64)
        positive = probability > 0.0
        log_probability[positive] = np.log(probability[positive])
        log_likelihood_without_constant = 0.0
        chunk_size = 2048 if need_gradient else 4096
        for start in range(0, self.observation.size, chunk_size):
            observed = self.observation[start : start + chunk_size]
            index = self.center[start : start + chunk_size, None] + self.offset
            valid = (index >= 0) & (index < component_count)
            safe_index = np.clip(index, 0, component_count - 1)
            log_kernel = -0.5 * np.square(
                (observed[:, None] - safe_index.astype(np.float64)) / self.sigma
            )
            log_weight = log_kernel + log_probability[safe_index]
            log_weight[~valid] = -np.inf
            row_maximum = np.max(log_weight, axis=1)
            if np.any(~np.isfinite(row_maximum)):
                raise RuntimeError("numerical support was lost during constrained EM")
            normalized_weight = np.exp(log_weight - row_maximum[:, None])
            row_log_density = row_maximum + np.log(
                np.sum(normalized_weight, axis=1)
            )
            log_likelihood_without_constant += float(np.sum(row_log_density))
            responsibility = np.exp(log_weight - row_log_density[:, None])
            responsibility[~valid] = 0.0
            expected += np.bincount(
                safe_index[valid],
                weights=responsibility[valid],
                minlength=component_count,
            )
            if gradient is not None:
                gradient_term = np.exp(log_kernel - row_log_density[:, None])
                gradient_term[~valid] = 0.0
                gradient += np.bincount(
                    safe_index[valid],
                    weights=gradient_term[valid],
                    minlength=component_count,
                )
        next_probability, tilt = _mean_constrained_m_step(
            expected / float(self.observation.size),
            self.support,
            self.target_mean,
        )
        normal_constant = -float(self.observation.size) * math.log(
            self.sigma * math.sqrt(2.0 * math.pi)
        )
        return (
            next_probability,
            log_likelihood_without_constant + normal_constant,
            gradient,
            tilt,
        )

    def linear_oracle(
        self, gradient: np.ndarray
    ) -> tuple[float, tuple[int, int], tuple[float, float]]:
        """Maximize a linear functional over fixed-mean lattice laws."""

        mu = self.target_mean
        nearest = int(round(mu))
        if abs(mu - nearest) <= 32.0 * np.finfo(np.float64).eps:
            return float(gradient[nearest]), (nearest, nearest), (1.0, 0.0)
        lower_maximum = int(math.floor(mu))
        upper_minimum = int(math.ceil(mu))
        best_value = -np.inf
        best_pair = (-1, -1)
        best_weight = (0.0, 0.0)
        upper_index = np.arange(
            upper_minimum, gradient.size, dtype=np.int64
        )
        for lower in range(lower_maximum + 1):
            denominator = upper_index.astype(np.float64) - float(lower)
            lower_weight = (upper_index.astype(np.float64) - mu) / denominator
            upper_weight = (mu - float(lower)) / denominator
            value = (
                lower_weight * float(gradient[lower])
                + upper_weight * gradient[upper_index]
            )
            position = int(np.argmax(value))
            if float(value[position]) > best_value:
                best_value = float(value[position])
                best_pair = (lower, int(upper_index[position]))
                best_weight = (
                    float(lower_weight[position]),
                    float(upper_weight[position]),
                )
        return best_value, best_pair, best_weight


def _largest_remainder_counts(expected_count: np.ndarray) -> np.ndarray:
    lower = np.floor(expected_count).astype(np.int64)
    remainder = int(round(float(np.sum(expected_count)))) - int(np.sum(lower))
    if remainder < 0 or remainder > expected_count.size:
        raise RuntimeError("invalid largest-remainder count")
    fraction = expected_count - lower.astype(np.float64)
    support = np.arange(expected_count.size, dtype=np.int64)
    order = np.lexsort((support, -fraction))
    lower[order[:remainder]] += 1
    return lower


def _adjacent_mass_correction(
    histogram: np.ndarray,
    expected_count: np.ndarray,
    endpoint_mass: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reach exact endpoint mass by deterministic least-cost adjacent moves."""

    output = np.asarray(histogram, dtype=np.int64).copy()
    support = np.arange(output.size, dtype=np.int64)
    initial_mass = int(np.dot(support, output))
    difference = int(endpoint_mass) - initial_mass
    if difference == 0:
        return output, {
            "initial_endpoint_mass": initial_mass,
            "required_unit_moves": 0,
            "direction": "none",
        }
    direction = 1 if difference > 0 else -1
    remaining = abs(difference)
    version = np.zeros(output.size, dtype=np.int64)
    heap: list[tuple[float, int, int, int]] = []

    def push(source: int) -> None:
        destination = source + direction
        if source < 0 or source >= output.size:
            return
        if destination < 0 or destination >= output.size or output[source] <= 0:
            return
        source_error = float(output[source]) - float(expected_count[source])
        destination_error = float(output[destination]) - float(
            expected_count[destination]
        )
        cost = (
            (source_error - 1.0) ** 2
            - source_error**2
            + (destination_error + 1.0) ** 2
            - destination_error**2
        )
        heapq.heappush(
            heap,
            (cost, source, int(version[source]), int(version[destination])),
        )

    for source in range(output.size):
        push(source)
    completed = 0
    while completed < remaining:
        if not heap:
            raise RuntimeError("integer histogram has no legal endpoint-mass correction")
        _, source, source_version, destination_version = heapq.heappop(heap)
        destination = source + direction
        if (
            source_version != int(version[source])
            or destination_version != int(version[destination])
            or output[source] <= 0
        ):
            continue
        output[source] -= 1
        output[destination] += 1
        version[source] += 1
        version[destination] += 1
        completed += 1
        for affected in (source - 1, source, source + 1):
            push(affected)
    return output, {
        "initial_endpoint_mass": initial_mass,
        "required_unit_moves": remaining,
        "direction": "up" if direction > 0 else "down",
    }


def _erdos_gallai(histogram: np.ndarray) -> dict[str, object]:
    support = np.arange(histogram.size, dtype=np.int64)
    degree = np.repeat(support, histogram.astype(np.int64))[::-1]
    node_count = int(degree.size)
    endpoint_mass = int(np.sum(degree, dtype=np.int64))
    if endpoint_mass % 2:
        return {"graphical": False, "reason": "odd endpoint mass"}
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(degree, dtype=np.int64))
    )
    # Evaluate exactly the same inequalities as the former Python rank loop,
    # but perform all monotone searches and integer arithmetic in vectorized
    # int64 arrays.  This is certificate-only: it does not change the released
    # histogram, the probability vector, or any random-number consumption.
    ranks = np.arange(1, node_count + 1, dtype=np.int64)
    count_ge_rank = np.searchsorted(-degree, -ranks, side="right")
    split = np.maximum(ranks, count_ge_rank)
    right = (
        ranks * (ranks - 1)
        + ranks * (split - ranks)
        + prefix[node_count]
        - prefix[split]
    )
    violations = prefix[1:] - right
    violating = np.flatnonzero(violations > 0)
    first_violation = int(violating[0] + 1) if violating.size else None
    maximum_violation = (
        int(np.max(violations[violating])) if violating.size else 0
    )
    return {
        "graphical": first_violation is None,
        "reason": (
            "all Erdos-Gallai inequalities hold"
            if first_violation is None
            else "Erdos-Gallai violation"
        ),
        "first_violating_rank": first_violation,
        "maximum_violation": maximum_violation,
    }
