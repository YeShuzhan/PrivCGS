from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


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
            raise RuntimeError("no legal adjacent endpoint-mass correction")
        _cost, source, source_version, destination_version = heapq.heappop(heap)
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
    degrees = np.repeat(support, histogram.astype(np.int64))[::-1]
    node_count = int(degrees.size)
    endpoint_mass = int(np.sum(degrees, dtype=np.int64))
    if endpoint_mass % 2:
        return {"graphical": False, "reason": "odd endpoint mass"}
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(degrees, dtype=np.int64))
    )
    ranks = np.arange(1, node_count + 1, dtype=np.int64)
    count_ge_rank = np.searchsorted(-degrees, -ranks, side="right")
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
    maximum_violation = int(np.max(violations, initial=0))
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


def _graphicalize_by_balancing(
    histogram: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the first graphical state on the canonical balancing path."""

    output = np.asarray(histogram, dtype=np.int64).copy()
    support = np.arange(output.size, dtype=np.int64)
    initial_certificate = _erdos_gallai(output)
    initial_potential = int(np.dot(support * support, output))
    if bool(initial_certificate["graphical"]):
        return output, {
            "required": False,
            "unit_transfer_count": 0,
            "selection": "current maximum degree to current minimum degree",
            "first_graphical_state_returned": True,
            "initial_graphicality": initial_certificate,
            "final_graphicality": initial_certificate,
            "initial_squared_degree_potential": initial_potential,
            "final_squared_degree_potential": initial_potential,
            "strict_potential_descent_per_transfer": True,
        }

    transfer_count = 0
    certificate = initial_certificate
    while not bool(certificate["graphical"]):
        occupied = np.flatnonzero(output)
        if occupied.size == 0:
            raise RuntimeError("positive endpoint mass lost all degree support")
        low = int(occupied[0])
        high = int(occupied[-1])
        if high - low < 2:
            raise RuntimeError(
                "almost-regular even degree sequence unexpectedly failed graphicality"
            )
        output[high] -= 1
        output[high - 1] += 1
        output[low] -= 1
        output[low + 1] += 1
        transfer_count += 1
        certificate = _erdos_gallai(output)

    final_potential = int(np.dot(support * support, output))
    if not final_potential < initial_potential:
        raise RuntimeError("graphicality repair did not decrease its exact potential")
    return output, {
        "required": True,
        "unit_transfer_count": transfer_count,
        "selection": "current maximum degree to current minimum degree",
        "first_graphical_state_returned": True,
        "initial_graphicality": initial_certificate,
        "final_graphicality": certificate,
        "initial_squared_degree_potential": initial_potential,
        "final_squared_degree_potential": final_potential,
        "strict_potential_descent_per_transfer": True,
        "finite_stop": (
            "integer squared-degree potential strictly decreases until the "
            "even-sum almost-regular graphical state"
        ),
    }
