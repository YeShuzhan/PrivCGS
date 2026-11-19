"""Memory-capped low-rank row dots for the frozen GCFR-Scale executor."""

from __future__ import annotations

import numpy as np


TEMPORARY_OPERAND_BYTE_CAP = 16 * 1024 * 1024


def batched_binary64_row_dot(
    left_factor: np.ndarray,
    right_factor: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Evaluate selected row pairs with the promoted batched NumPy arithmetic.

    This batched arithmetic is part of the frozen GCFR-Scale execution contract.
    Depending on NumPy, BLAS, alignment and operand structure, it may differ by
    final-bit roundoff from a one-dimensional ``row @ row`` loop.  Batching
    removes Python iteration while a fixed temporary-byte cap bounds gather
    memory.  The cap is an executor property, not a graph-law parameter.
    """

    left_factor = np.asarray(left_factor, dtype=np.float64)
    right_factor = np.asarray(right_factor, dtype=np.float64)
    left_index = np.asarray(left_index, dtype=np.int64)
    right_index = np.asarray(right_index, dtype=np.int64)
    rank = int(left_factor.shape[1])
    bytes_per_row = max(1, 2 * rank * np.dtype(np.float64).itemsize)
    batch_size = max(1, TEMPORARY_OPERAND_BYTE_CAP // bytes_per_row)
    output = np.empty(left_index.size, dtype=np.float64)
    maximum_batch = 0
    for start in range(0, left_index.size, batch_size):
        stop = min(left_index.size, start + batch_size)
        left = np.ascontiguousarray(left_factor[left_index[start:stop]])
        right = np.ascontiguousarray(right_factor[right_index[start:stop]])
        output[start:stop] = np.matmul(
            left[:, None, :], right[:, :, None]
        )[:, 0, 0]
        maximum_batch = max(maximum_batch, stop - start)
    return output, {
        "row_dot_count": int(output.size),
        "maximum_batch_size": int(maximum_batch),
        "temporary_operand_byte_cap": int(TEMPORARY_OPERAND_BYTE_CAP),
        "gcfr_scale_batched_binary64_arithmetic": True,
    }
