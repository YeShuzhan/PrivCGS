#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <cstdio>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

double logsumexp(const double* values, std::int64_t count) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int64_t index = 0; index < count; ++index) {
        maximum = std::max(maximum, values[index]);
    }
    double total = 0.0;
    for (std::int64_t index = 0; index < count; ++index) {
        total += std::exp(values[index] - maximum);
    }
    return maximum + std::log(total);
}

}  // namespace

// Direct finalization of an actual capped iterate, without another Sinkhorn
// iteration. It is exported also for qualification on captured formal inputs.
ROUTE_S_EXPORT std::int32_t route_s_direct_cap_residual_balance(
    std::int64_t rows, std::int64_t columns, const double* base,
    const double* target_rows, const double* target_columns,
    double* left, const double* right, double tolerance,
    double* out_before, double* out_after) {
    if (rows <= 0 || columns <= 0 || !base || !target_rows ||
        !target_columns || !left || !right || !out_before || !out_after ||
        !std::isfinite(tolerance) || tolerance <= 0.0) return 1;
    std::vector<double> scratch(static_cast<std::size_t>(std::max(rows, columns)));
    std::vector<double> residual(static_cast<std::size_t>(rows));
    std::vector<double> original_left(left, left + rows);
    auto marginal_error = [&]() {
        double maximum = 0.0;
        for (std::int64_t i = 0; i < rows; ++i) {
            for (std::int64_t j = 0; j < columns; ++j) {
                const double first = base[i * columns + j] + left[i];
                scratch[j] = first + right[j];
            }
            const double value = logsumexp(scratch.data(), columns) - target_rows[i];
            if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
            residual[i] = value;
            maximum = std::max(maximum, std::abs(value));
        }
        for (std::int64_t j = 0; j < columns; ++j) {
            for (std::int64_t i = 0; i < rows; ++i) {
                const double first = base[i * columns + j] + left[i];
                scratch[i] = first + right[j];
            }
            const double value = logsumexp(scratch.data(), rows) - target_columns[j];
            if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
            maximum = std::max(maximum, std::abs(value));
        }
        return maximum;
    };
    const double before = marginal_error();
    *out_before = before;
    *out_after = before;
    // Formerly accepted iterates are never changed. Large/nonfinite failures
    // retain their original failure; this is not a relaxed tolerance.
    if (!std::isfinite(before) || before <= tolerance || before > 2.0 * tolerance)
        return 3;
    for (std::int64_t i = 0; i < rows; ++i)
        left[i] = original_left[i] - 0.5 * residual[i];
    const double after = marginal_error();
    *out_after = after;
    if (!std::isfinite(after) || after > tolerance) {
        std::copy(original_left.begin(), original_left.end(), left);
        return 3;
    }
    return 0;
}

// Explicitly authorized only after the entire legacy path has failed.
// Continue the same two marginal updates on the same kernel and targets.
ROUTE_S_EXPORT std::int32_t route_s_authorized_failed_pair_continuation(
    std::int64_t rows, std::int64_t columns, const double* base,
    const double* target_rows, const double* target_columns,
    double* left, double* right, double tolerance,
    std::int64_t maximum_extra_iterations, std::int64_t* out_extra_iterations,
    double* out_before, double* out_after) {
    if (rows <= 0 || columns <= 0 || !base || !target_rows ||
        !target_columns || !left || !right || !out_extra_iterations ||
        !out_before || !out_after || maximum_extra_iterations < 0 ||
        !std::isfinite(tolerance) || tolerance <= 0.0) return 1;
    std::vector<double> scratch(static_cast<std::size_t>(std::max(rows, columns)));
    auto marginal_error = [&]() {
        double maximum = 0.0;
        for (std::int64_t i = 0; i < rows; ++i) {
            for (std::int64_t j = 0; j < columns; ++j)
                scratch[j] = (base[i * columns + j] + left[i]) + right[j];
            const double residual = logsumexp(scratch.data(), columns) - target_rows[i];
            if (!std::isfinite(residual)) return std::numeric_limits<double>::infinity();
            maximum = std::max(maximum, std::abs(residual));
        }
        for (std::int64_t j = 0; j < columns; ++j) {
            for (std::int64_t i = 0; i < rows; ++i)
                scratch[i] = (base[i * columns + j] + left[i]) + right[j];
            const double residual = logsumexp(scratch.data(), rows) - target_columns[j];
            if (!std::isfinite(residual)) return std::numeric_limits<double>::infinity();
            maximum = std::max(maximum, std::abs(residual));
        }
        return maximum;
    };
    *out_extra_iterations = 0;
    *out_before = marginal_error();
    *out_after = *out_before;
    if (!std::isfinite(*out_before)) return 3;
    if (*out_before <= tolerance) return 0;
    for (std::int64_t step = 1; step <= maximum_extra_iterations; ++step) {
        for (std::int64_t i = 0; i < rows; ++i) {
            for (std::int64_t j = 0; j < columns; ++j)
                scratch[j] = base[i * columns + j] + right[j];
            left[i] = target_rows[i] - logsumexp(scratch.data(), columns);
        }
        for (std::int64_t j = 0; j < columns; ++j) {
            for (std::int64_t i = 0; i < rows; ++i)
                scratch[i] = base[i * columns + j] + left[i];
            right[j] = target_columns[j] - logsumexp(scratch.data(), rows);
        }
        *out_extra_iterations = step;
        const double error = marginal_error();
        *out_after = error;
        if (!std::isfinite(error)) return 3;
        if (error <= tolerance) return 0;
        if (error <= 2.0 * tolerance) {
            double before_balance = error, after_balance = error;
            const auto balanced = route_s_direct_cap_residual_balance(
                rows, columns, base, target_rows, target_columns,
                left, right, tolerance, &before_balance, &after_balance);
            if (balanced == 0) {
                *out_after = after_balance;
                return 0;
            }
        }
    }
    return 3;
}

ROUTE_S_EXPORT std::int32_t route_s_chain_marginals(
    std::int64_t position_count,
    const std::int64_t* offsets,
    const std::int64_t* states,
    const double* gain,
    const double* forward,
    const double* path_degree,
    const double* path_target,
    double quadratic,
    double log_normalizer,
    std::int32_t materialize_transitions,
    double* out_marginals,
    std::int64_t* out_transition_offsets,
    double* out_transitions,
    double* out_maximum_marginal_error,
    double* out_maximum_row_error,
    std::int64_t* out_transition_work) {
    if (position_count <= 0 || offsets == nullptr || states == nullptr ||
        gain == nullptr || forward == nullptr || path_degree == nullptr ||
        path_target == nullptr || out_marginals == nullptr ||
        out_transition_offsets == nullptr ||
        out_maximum_marginal_error == nullptr ||
        out_maximum_row_error == nullptr || out_transition_work == nullptr ||
        !std::isfinite(quadratic) || !std::isfinite(log_normalizer) ||
        offsets[0] != 0 || (materialize_transitions && out_transitions == nullptr)) {
        return 1;
    }
    std::int64_t maximum_width = 0;
    std::int64_t transition_total = 0;
    out_transition_offsets[0] = 0;
    for (std::int64_t index = 0; index < position_count; ++index) {
        const std::int64_t width = offsets[index + 1] - offsets[index];
        if (width <= 0) {
            return 2;
        }
        maximum_width = std::max(maximum_width, width);
        if (index + 1 < position_count) {
            const std::int64_t next_width =
                offsets[index + 2] - offsets[index + 1];
            if (next_width <= 0) {
                return 2;
            }
            transition_total += width * next_width;
            out_transition_offsets[index + 1] = transition_total;
        }
    }

    const std::int64_t total_states = offsets[position_count];
    std::vector<double> backward(static_cast<std::size_t>(total_states));
    std::vector<double> scratch(static_cast<std::size_t>(maximum_width));
    const std::int64_t last_start = offsets[position_count - 1];
    const double last_degree = path_degree[position_count];
    const double last_target = path_target[position_count];
    for (std::int64_t state = last_start; state < total_states; ++state) {
        const double residual =
            last_degree + static_cast<double>(states[state]) - last_target;
        const double square = residual * residual;
        backward[static_cast<std::size_t>(state)] = (-square) * quadratic;
    }

    std::int64_t transition_work = 0;
    for (std::int64_t index = position_count - 2; index >= 0; --index) {
        const std::int64_t previous_start = offsets[index];
        const std::int64_t previous_stop = offsets[index + 1];
        const std::int64_t current_start = previous_stop;
        const std::int64_t current_stop = offsets[index + 2];
        const std::int64_t current_width = current_stop - current_start;
        transition_work += (previous_stop - previous_start) * current_width;
        const double node_degree = path_degree[index + 1];
        const double node_target = path_target[index + 1];
        for (std::int64_t previous = previous_start;
             previous < previous_stop; ++previous) {
            for (std::int64_t current = current_start;
                 current < current_stop; ++current) {
                const double realized =
                    node_degree + static_cast<double>(states[previous]) -
                    static_cast<double>(states[current]);
                const double residual = realized - node_target;
                const double square = residual * residual;
                const double penalty = square * quadratic;
                const double local = gain[current] - penalty;
                scratch[static_cast<std::size_t>(current - current_start)] =
                    local + backward[static_cast<std::size_t>(current)];
            }
            backward[static_cast<std::size_t>(previous)] =
                logsumexp(scratch.data(), current_width);
        }
    }

    double maximum_marginal_error = 0.0;
    for (std::int64_t index = 0; index < position_count; ++index) {
        const std::int64_t start = offsets[index];
        const std::int64_t stop = offsets[index + 1];
        const std::int64_t width = stop - start;
        for (std::int64_t state = start; state < stop; ++state) {
            const double combined =
                forward[state] + backward[static_cast<std::size_t>(state)];
            const double raw = combined - log_normalizer;
            out_marginals[state] = raw;
            scratch[static_cast<std::size_t>(state - start)] = raw;
        }
        const double normalizer = logsumexp(scratch.data(), width);
        maximum_marginal_error = std::max(
            maximum_marginal_error, std::abs(normalizer)
        );
        for (std::int64_t state = start; state < stop; ++state) {
            out_marginals[state] = out_marginals[state] - normalizer;
        }
    }

    double maximum_row_error = 0.0;
    if (materialize_transitions) {
        for (std::int64_t index = 0; index + 1 < position_count; ++index) {
            const std::int64_t previous_start = offsets[index];
            const std::int64_t previous_stop = offsets[index + 1];
            const std::int64_t current_start = previous_stop;
            const std::int64_t current_stop = offsets[index + 2];
            const std::int64_t current_width = current_stop - current_start;
            const double node_degree = path_degree[index + 1];
            const double node_target = path_target[index + 1];
            const std::int64_t transition_start = out_transition_offsets[index];
            for (std::int64_t previous = previous_start;
                 previous < previous_stop; ++previous) {
                const std::int64_t row = previous - previous_start;
                for (std::int64_t current = current_start;
                     current < current_stop; ++current) {
                    const double realized =
                        node_degree + static_cast<double>(states[previous]) -
                        static_cast<double>(states[current]);
                    const double residual = realized - node_target;
                    const double square = residual * residual;
                    const double penalty = square * quadratic;
                    const double first = gain[current] - penalty;
                    const double second =
                        first + backward[static_cast<std::size_t>(current)];
                    const double value =
                        second - backward[static_cast<std::size_t>(previous)];
                    scratch[static_cast<std::size_t>(current - current_start)] = value;
                    out_transitions[
                        transition_start + row * current_width +
                        (current - current_start)
                    ] = value;
                }
                const double row_normalizer =
                    logsumexp(scratch.data(), current_width);
                maximum_row_error = std::max(
                    maximum_row_error, std::abs(row_normalizer)
                );
                for (std::int64_t current = 0; current < current_width; ++current) {
                    const std::int64_t output =
                        transition_start + row * current_width + current;
                    out_transitions[output] =
                        out_transitions[output] - row_normalizer;
                }
            }
        }
    }
    *out_maximum_marginal_error = maximum_marginal_error;
    *out_maximum_row_error = maximum_row_error;
    *out_transition_work = transition_work;
    return 0;
}

// Exact sufficient-state form of the reference conditional law.  Every row's
// binary64 normalization correction is computed while the backward row is hot;
// the O(sum width_i*width_{i+1}) transition table is never materialized.
ROUTE_S_EXPORT std::int32_t route_s_chain_marginals_sufficient(
    std::int64_t position_count,
    const std::int64_t* offsets,
    const std::int64_t* states,
    const double* gain,
    const double* forward,
    const double* path_degree,
    const double* path_target,
    double quadratic,
    double log_normalizer,
    double* out_marginals,
    double* out_backward,
    double* out_row_correction,
    double* out_maximum_marginal_error,
    std::int64_t* out_transition_work) {
    if (position_count <= 0 || offsets == nullptr || states == nullptr ||
        gain == nullptr || forward == nullptr || path_degree == nullptr ||
        path_target == nullptr || out_marginals == nullptr ||
        out_backward == nullptr || out_row_correction == nullptr ||
        out_maximum_marginal_error == nullptr || out_transition_work == nullptr ||
        !std::isfinite(quadratic) || !std::isfinite(log_normalizer) ||
        offsets[0] != 0) {
        return 1;
    }
    std::int64_t maximum_width = 0;
    std::int64_t transition_work = 0;
    for (std::int64_t index = 0; index < position_count; ++index) {
        const std::int64_t width = offsets[index + 1] - offsets[index];
        if (width <= 0) {
            return 2;
        }
        maximum_width = std::max(maximum_width, width);
        if (index + 1 < position_count) {
            const std::int64_t next_width =
                offsets[index + 2] - offsets[index + 1];
            if (next_width <= 0 ||
                width > std::numeric_limits<std::int64_t>::max() / next_width) {
                return 2;
            }
            transition_work += width * next_width;
        }
    }

    const std::int64_t total_states = offsets[position_count];
    std::vector<double> scratch(static_cast<std::size_t>(maximum_width));
    const std::int64_t last_start = offsets[position_count - 1];
    const double last_degree = path_degree[position_count];
    const double last_target = path_target[position_count];
    for (std::int64_t state = last_start; state < total_states; ++state) {
        const double residual =
            last_degree + static_cast<double>(states[state]) - last_target;
        const double square = residual * residual;
        out_backward[state] = (-square) * quadratic;
        out_row_correction[state] = 0.0;
    }

    for (std::int64_t index = position_count - 2; index >= 0; --index) {
        const std::int64_t previous_start = offsets[index];
        const std::int64_t previous_stop = offsets[index + 1];
        const std::int64_t current_start = previous_stop;
        const std::int64_t current_stop = offsets[index + 2];
        const std::int64_t current_width = current_stop - current_start;
        const double node_degree = path_degree[index + 1];
        const double node_target = path_target[index + 1];
        for (std::int64_t previous = previous_start;
             previous < previous_stop; ++previous) {
            for (std::int64_t current = current_start;
                 current < current_stop; ++current) {
                const double realized =
                    node_degree + static_cast<double>(states[previous]) -
                    static_cast<double>(states[current]);
                const double residual = realized - node_target;
                const double square = residual * residual;
                const double penalty = square * quadratic;
                const double local = gain[current] - penalty;
                scratch[static_cast<std::size_t>(current - current_start)] =
                    local + out_backward[current];
            }
            out_backward[previous] = logsumexp(scratch.data(), current_width);
            for (std::int64_t current = 0; current < current_width; ++current) {
                scratch[static_cast<std::size_t>(current)] =
                    scratch[static_cast<std::size_t>(current)] -
                    out_backward[previous];
            }
            out_row_correction[previous] =
                logsumexp(scratch.data(), current_width);
        }
    }

    double maximum_marginal_error = 0.0;
    for (std::int64_t index = 0; index < position_count; ++index) {
        const std::int64_t start = offsets[index];
        const std::int64_t stop = offsets[index + 1];
        const std::int64_t width = stop - start;
        for (std::int64_t state = start; state < stop; ++state) {
            const double combined = forward[state] + out_backward[state];
            const double raw = combined - log_normalizer;
            out_marginals[state] = raw;
            scratch[static_cast<std::size_t>(state - start)] = raw;
        }
        const double normalizer = logsumexp(scratch.data(), width);
        maximum_marginal_error = std::max(
            maximum_marginal_error, std::abs(normalizer)
        );
        for (std::int64_t state = start; state < stop; ++state) {
            out_marginals[state] = out_marginals[state] - normalizer;
        }
    }
    *out_maximum_marginal_error = maximum_marginal_error;
    *out_transition_work = transition_work;
    return 0;
}

ROUTE_S_EXPORT std::int32_t route_s_packed_entropic_projection(
    std::int64_t position_count,
    const std::int64_t* state_offsets,
    const double* targets,
    const std::int64_t* transition_offsets,
    const double* reference_transitions,
    double* out_left_scales,
    double* out_right_scales,
    std::int64_t* out_total_iterations,
    double* out_maximum_error,
    double* out_maximum_bound) {
    if (position_count <= 0 || state_offsets == nullptr || targets == nullptr ||
        transition_offsets == nullptr || reference_transitions == nullptr ||
        out_left_scales == nullptr || out_right_scales == nullptr ||
        out_total_iterations == nullptr || out_maximum_error == nullptr ||
        out_maximum_bound == nullptr || state_offsets[0] != 0 ||
        transition_offsets[0] != 0) {
        return 1;
    }
    const std::int64_t total_states = state_offsets[position_count];
    std::fill(out_left_scales, out_left_scales + total_states, 0.0);
    std::fill(out_right_scales, out_right_scales + total_states, 0.0);
    if (position_count == 1) {
        *out_total_iterations = 0;
        *out_maximum_error = 0.0;
        *out_maximum_bound = 0.0;
        return 0;
    }
    std::int64_t maximum_width = 0;
    std::int64_t maximum_pairs = 0;
    for (std::int64_t index = 0; index < position_count; ++index) {
        const std::int64_t width =
            state_offsets[index + 1] - state_offsets[index];
        if (width <= 0) {
            return 2;
        }
        maximum_width = std::max(maximum_width, width);
        if (index + 1 < position_count) {
            const std::int64_t next_width =
                state_offsets[index + 2] - state_offsets[index + 1];
            maximum_pairs = std::max(maximum_pairs, width * next_width);
        }
    }
    std::vector<double> base(static_cast<std::size_t>(maximum_pairs));
    std::vector<double> scratch(static_cast<std::size_t>(maximum_width));
    std::int64_t total_iterations = 0;
    double maximum_error = 0.0;
    double maximum_bound = 0.0;
    constexpr std::int64_t mantissa_bits = 52;
    constexpr std::int64_t maximum_iterations = mantissa_bits * mantissa_bits;
    const double machine_epsilon = std::numeric_limits<double>::epsilon();

    for (std::int64_t index = 0; index + 1 < position_count; ++index) {
        const std::int64_t left_start = state_offsets[index];
        const std::int64_t left_stop = state_offsets[index + 1];
        const std::int64_t right_start = left_stop;
        const std::int64_t right_stop = state_offsets[index + 2];
        const std::int64_t left_width = left_stop - left_start;
        const std::int64_t right_width = right_stop - right_start;
        const std::int64_t transition_start = transition_offsets[index];
        double operation_scale = std::max(
            1.0, std::max(static_cast<double>(left_width),
                          static_cast<double>(right_width))
        );
        for (std::int64_t left = 0; left < left_width; ++left) {
            operation_scale = std::max(
                operation_scale, std::abs(targets[left_start + left])
            );
            for (std::int64_t right = 0; right < right_width; ++right) {
                base[static_cast<std::size_t>(left * right_width + right)] =
                    targets[left_start + left] +
                    reference_transitions[
                        transition_start + left * right_width + right
                    ];
            }
        }
        for (std::int64_t right = 0; right < right_width; ++right) {
            operation_scale = std::max(
                operation_scale, std::abs(targets[right_start + right])
            );
        }
        const double bound =
            static_cast<double>(maximum_iterations) * machine_epsilon *
            operation_scale;
        double error = std::numeric_limits<double>::infinity();
        std::int64_t iteration = 0;
        for (iteration = 1; iteration <= maximum_iterations; ++iteration) {
            for (std::int64_t left = 0; left < left_width; ++left) {
                for (std::int64_t right = 0; right < right_width; ++right) {
                    scratch[static_cast<std::size_t>(right)] =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_right_scales[right_start + right];
                }
                out_left_scales[left_start + left] =
                    targets[left_start + left] -
                    logsumexp(scratch.data(), right_width);
            }
            for (std::int64_t right = 0; right < right_width; ++right) {
                for (std::int64_t left = 0; left < left_width; ++left) {
                    scratch[static_cast<std::size_t>(left)] =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_left_scales[left_start + left];
                }
                out_right_scales[right_start + right] =
                    targets[right_start + right] -
                    logsumexp(scratch.data(), left_width);
            }
            double row_error = 0.0;
            for (std::int64_t left = 0; left < left_width; ++left) {
                for (std::int64_t right = 0; right < right_width; ++right) {
                    const double first =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_left_scales[left_start + left];
                    scratch[static_cast<std::size_t>(right)] =
                        first + out_right_scales[right_start + right];
                }
                row_error = std::max(
                    row_error,
                    std::abs(
                        logsumexp(scratch.data(), right_width) -
                        targets[left_start + left]
                    )
                );
            }
            double column_error = 0.0;
            for (std::int64_t right = 0; right < right_width; ++right) {
                for (std::int64_t left = 0; left < left_width; ++left) {
                    const double first =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_left_scales[left_start + left];
                    scratch[static_cast<std::size_t>(left)] =
                        first + out_right_scales[right_start + right];
                }
                column_error = std::max(
                    column_error,
                    std::abs(
                        logsumexp(scratch.data(), left_width) -
                        targets[right_start + right]
                    )
                );
            }
            error = std::max(row_error, column_error);
            if (error <= bound) {
                break;
            }
        }
        if (iteration > maximum_iterations) {
            return 3;
        }
        total_iterations += iteration;
        maximum_error = std::max(maximum_error, error);
        maximum_bound = std::max(maximum_bound, bound);
    }
    *out_total_iterations = total_iterations;
    *out_maximum_error = maximum_error;
    *out_maximum_bound = maximum_bound;
    return 0;
}

// Reconstruct one exact normalized transition block from its sufficient state,
// project it immediately, and release the block before advancing.
ROUTE_S_EXPORT std::int32_t route_s_on_demand_entropic_projection_sufficient(
    std::int64_t position_count,
    const std::int64_t* state_offsets,
    const std::int64_t* states,
    const double* gain,
    const double* backward,
    const double* row_correction,
    const double* path_degree,
    const double* path_target,
    double quadratic,
    const double* targets,
    double* out_left_scales,
    double* out_right_scales,
    std::int64_t* out_total_iterations,
    double* out_maximum_error,
    double* out_maximum_bound,
    double* out_maximum_row_error) {
    if (position_count <= 0 || state_offsets == nullptr || states == nullptr ||
        gain == nullptr || backward == nullptr || row_correction == nullptr ||
        path_degree == nullptr || path_target == nullptr || targets == nullptr ||
        out_left_scales == nullptr || out_right_scales == nullptr ||
        out_total_iterations == nullptr || out_maximum_error == nullptr ||
        out_maximum_bound == nullptr || out_maximum_row_error == nullptr ||
        !std::isfinite(quadratic) || state_offsets[0] != 0) {
        return 1;
    }
    const std::int64_t total_states = state_offsets[position_count];
    std::fill(out_left_scales, out_left_scales + total_states, 0.0);
    std::fill(out_right_scales, out_right_scales + total_states, 0.0);
    std::int64_t maximum_width = 0;
    std::int64_t maximum_pairs = 0;
    for (std::int64_t index = 0; index < position_count; ++index) {
        const std::int64_t width =
            state_offsets[index + 1] - state_offsets[index];
        if (width <= 0) {
            return 2;
        }
        maximum_width = std::max(maximum_width, width);
        if (index + 1 < position_count) {
            const std::int64_t next_width =
                state_offsets[index + 2] - state_offsets[index + 1];
            if (next_width <= 0 ||
                width > std::numeric_limits<std::int64_t>::max() / next_width) {
                return 2;
            }
            maximum_pairs = std::max(maximum_pairs, width * next_width);
        }
    }
    if (position_count == 1) {
        *out_total_iterations = 0;
        *out_maximum_error = 0.0;
        *out_maximum_bound = 0.0;
        *out_maximum_row_error = 0.0;
        return 0;
    }

    std::vector<double> base(static_cast<std::size_t>(maximum_pairs));
    std::vector<double> scratch(static_cast<std::size_t>(maximum_width));
    std::int64_t total_iterations = 0;
    double maximum_error = 0.0;
    double maximum_bound = 0.0;
    double maximum_row_error = 0.0;
    constexpr std::int64_t mantissa_bits = 52;
    constexpr std::int64_t maximum_iterations = mantissa_bits * mantissa_bits;
    // Keep the historical accuracy bound and every successful-path operation.
    // Only the resource cap is extended: ill-conditioned positive kernels can
    // require more than 52^2 Sinkhorn steps. Never accept a larger residual.
    constexpr std::int64_t continuation_maximum_iterations = maximum_iterations * 100;
    const double machine_epsilon = std::numeric_limits<double>::epsilon();

    for (std::int64_t index = 0; index + 1 < position_count; ++index) {
        const std::int64_t left_start = state_offsets[index];
        const std::int64_t left_stop = state_offsets[index + 1];
        const std::int64_t right_start = left_stop;
        const std::int64_t right_stop = state_offsets[index + 2];
        const std::int64_t left_width = left_stop - left_start;
        const std::int64_t right_width = right_stop - right_start;
        const double node_degree = path_degree[index + 1];
        const double node_target = path_target[index + 1];
        double operation_scale = std::max(
            1.0, std::max(static_cast<double>(left_width),
                          static_cast<double>(right_width))
        );
        for (std::int64_t left = 0; left < left_width; ++left) {
            const std::int64_t previous = left_start + left;
            const double row_normalizer = row_correction[previous];
            maximum_row_error = std::max(
                maximum_row_error, std::abs(row_normalizer)
            );
            operation_scale = std::max(
                operation_scale, std::abs(targets[left_start + left])
            );
            for (std::int64_t right = 0; right < right_width; ++right) {
                const std::int64_t current = right_start + right;
                const double realized =
                    node_degree + static_cast<double>(states[previous]) -
                    static_cast<double>(states[current]);
                const double residual = realized - node_target;
                const double square = residual * residual;
                const double penalty = square * quadratic;
                const double first = gain[current] - penalty;
                const double second = first + backward[current];
                const double value = second - backward[previous];
                const double normalized = value - row_normalizer;
                base[static_cast<std::size_t>(left * right_width + right)] =
                    targets[left_start + left] + normalized;
            }
        }
        for (std::int64_t right = 0; right < right_width; ++right) {
            operation_scale = std::max(
                operation_scale, std::abs(targets[right_start + right])
            );
        }
        const double bound =
            static_cast<double>(maximum_iterations) * machine_epsilon *
            operation_scale;
        double error = std::numeric_limits<double>::infinity();
        std::int64_t iteration = 0;
        for (iteration = 1; iteration <= continuation_maximum_iterations; ++iteration) {
            for (std::int64_t left = 0; left < left_width; ++left) {
                for (std::int64_t right = 0; right < right_width; ++right) {
                    scratch[static_cast<std::size_t>(right)] =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_right_scales[right_start + right];
                }
                out_left_scales[left_start + left] =
                    targets[left_start + left] -
                    logsumexp(scratch.data(), right_width);
            }
            for (std::int64_t right = 0; right < right_width; ++right) {
                for (std::int64_t left = 0; left < left_width; ++left) {
                    scratch[static_cast<std::size_t>(left)] =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_left_scales[left_start + left];
                }
                out_right_scales[right_start + right] =
                    targets[right_start + right] -
                    logsumexp(scratch.data(), left_width);
            }
            double row_error = 0.0;
            for (std::int64_t left = 0; left < left_width; ++left) {
                for (std::int64_t right = 0; right < right_width; ++right) {
                    const double first =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_left_scales[left_start + left];
                    scratch[static_cast<std::size_t>(right)] =
                        first + out_right_scales[right_start + right];
                }
                row_error = std::max(
                    row_error,
                    std::abs(
                        logsumexp(scratch.data(), right_width) -
                        targets[left_start + left]
                    )
                );
            }
            double column_error = 0.0;
            for (std::int64_t right = 0; right < right_width; ++right) {
                for (std::int64_t left = 0; left < left_width; ++left) {
                    const double first =
                        base[static_cast<std::size_t>(left * right_width + right)] +
                        out_left_scales[left_start + left];
                    scratch[static_cast<std::size_t>(left)] =
                        first + out_right_scales[right_start + right];
                }
                column_error = std::max(
                    column_error,
                    std::abs(
                        logsumexp(scratch.data(), left_width) -
                        targets[right_start + right]
                    )
                );
            }
            error = std::max(row_error, column_error);
            if (error <= bound) {
                break;
            }
        }
        std::int64_t additional_cap_iterations = 0;
        if (iteration > continuation_maximum_iterations) {
            // User-authorized numerical tolerance, only after the historical
            // continuation cap. Previously successful paths remain unchanged.
            constexpr double cap_tolerance = 1e-6;
            if (!std::isfinite(error) || error > cap_tolerance) {
                double before = error;
                double after = error;
                const auto balanced = std::isfinite(error) ?
                    route_s_direct_cap_residual_balance(
                        left_width, right_width, base.data(),
                        targets + left_start, targets + right_start,
                        out_left_scales + left_start,
                        out_right_scales + right_start,
                        cap_tolerance, &before, &after) : 3;
                if (balanced != 0) {
                    constexpr std::int64_t maximum_extra_iterations =
                        continuation_maximum_iterations * 3;
                    const auto continued = route_s_authorized_failed_pair_continuation(
                        left_width, right_width, base.data(),
                        targets + left_start, targets + right_start,
                        out_left_scales + left_start, out_right_scales + right_start,
                        cap_tolerance, maximum_extra_iterations,
                        &additional_cap_iterations, &before, &after);
                    std::fprintf(stderr, "Q projection user-authorized failure continuation: index=%lld widths=%lld,%lld original_iterations=%lld additional_iterations=%lld maximum_total_iterations=%lld before=%.17g after=%.17g tolerance=%.17g status=%d\n", (long long)index, (long long)left_width, (long long)right_width, (long long)continuation_maximum_iterations, (long long)additional_cap_iterations, (long long)(continuation_maximum_iterations + maximum_extra_iterations), before, after, cap_tolerance, (int)continued);
                    if (continued != 0) return 3;
                } else {
                    std::fprintf(stderr, "Q projection direct residual balance: index=%lld widths=%lld,%lld before=%.17g after=%.17g tolerance=%.17g additional_sinkhorn_iterations=0\n", (long long)index, (long long)left_width, (long long)right_width, before, after, cap_tolerance);
                }
                error = after;
            }
            iteration = continuation_maximum_iterations + additional_cap_iterations;
            maximum_bound = std::max(maximum_bound, cap_tolerance);
            std::fprintf(stderr, "Q projection tolerance fallback: index=%lld widths=%lld,%lld iterations=%lld error=%.17g original_bound=%.17g accepted_tolerance=%.17g\n", (long long)index, (long long)left_width, (long long)right_width, (long long)iteration, error, bound, cap_tolerance);
        }
        if (iteration > maximum_iterations) std::fprintf(stderr, "Q projection continuation: index=%lld iterations=%lld error=%.17g bound=%.17g\n", (long long)index, (long long)iteration, error, bound);
        total_iterations += iteration;
        maximum_error = std::max(maximum_error, error);
        maximum_bound = std::max(maximum_bound, bound);
    }
    *out_total_iterations = total_iterations;
    *out_maximum_error = maximum_error;
    *out_maximum_bound = maximum_bound;
    *out_maximum_row_error = maximum_row_error;
    return 0;
}
