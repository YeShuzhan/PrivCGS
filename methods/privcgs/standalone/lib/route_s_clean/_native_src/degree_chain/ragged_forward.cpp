#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

std::int64_t transition_budget(std::int64_t left, std::int64_t right) {
    const std::int64_t length = left + right - 1;
    std::int64_t logarithm = 1;
    std::int64_t power = 2;
    while (power < length + 1) {
        power <<= 1;
        ++logarithm;
    }
    return length * logarithm * logarithm;
}

}  // namespace

// Exact finite-support Gaussian-chain forward propagation on a flat ragged tape.
// No dyad, wedge, graph adjacency, or private input is visible to this kernel.
ROUTE_S_EXPORT std::int32_t route_s_ragged_forward(
    std::int64_t state_count,
    const std::int64_t* offsets,
    const std::int64_t* states,
    const double* structural_gain,
    const double* path_degree,
    const double* path_target,
    double quadratic,
    double structural_temperature,
    double* forward,
    double* log_normalizer,
    std::int64_t* pair_work,
    std::int64_t* linear_log_squared_budget) {
    if (state_count <= 0 || offsets == nullptr || states == nullptr ||
        structural_gain == nullptr || path_degree == nullptr ||
        path_target == nullptr || forward == nullptr ||
        log_normalizer == nullptr || pair_work == nullptr ||
        linear_log_squared_budget == nullptr || !std::isfinite(quadratic) ||
        !std::isfinite(structural_temperature) || structural_temperature <= 0.0 ||
        offsets[0] != 0) {
        return 1;
    }
    std::int64_t maximum_width = 0;
    for (std::int64_t index = 0; index < state_count; ++index) {
        const std::int64_t width = offsets[index + 1] - offsets[index];
        if (width <= 0) {
            return 2;
        }
        maximum_width = std::max(maximum_width, width);
    }

    std::int64_t total_pair_work = 0;
    std::int64_t total_budget = 0;
    for (std::int64_t index = 1; index < state_count; ++index) {
        const std::int64_t previous_width = offsets[index] - offsets[index - 1];
        const std::int64_t current_width = offsets[index + 1] - offsets[index];
        if (previous_width >
            std::numeric_limits<std::int64_t>::max() / current_width) {
            return 3;
        }
        const std::int64_t local_pair_work = previous_width * current_width;
        const std::int64_t local_budget =
            transition_budget(previous_width, current_width);
        if (local_pair_work >
                std::numeric_limits<std::int64_t>::max() - total_pair_work ||
            local_budget >
                std::numeric_limits<std::int64_t>::max() - total_budget) {
            return 3;
        }
        total_pair_work += local_pair_work;
        total_budget += local_budget;
    }
    if (total_pair_work > total_budget) {
        return 3;
    }

    const std::int64_t first_stop = offsets[1];
    const double first_offset = path_degree[0] - path_target[0];
    for (std::int64_t state = 0; state < first_stop; ++state) {
        const double residual = first_offset - static_cast<double>(states[state]);
        forward[state] = structural_gain[state] / structural_temperature -
                         residual * residual * quadratic;
    }

    std::vector<double> log_weight(static_cast<std::size_t>(maximum_width));
    for (std::int64_t index = 1; index < state_count; ++index) {
        const std::int64_t previous_start = offsets[index - 1];
        const std::int64_t previous_stop = offsets[index];
        const std::int64_t current_start = offsets[index];
        const std::int64_t current_stop = offsets[index + 1];
        const std::int64_t previous_width = previous_stop - previous_start;
        const std::int64_t current_width = current_stop - current_start;
        const double node_offset = path_degree[index] - path_target[index];
        for (std::int64_t current = 0; current < current_width; ++current) {
            const double current_state =
                static_cast<double>(states[current_start + current]);
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                const double residual =
                    node_offset + static_cast<double>(states[previous_start + previous]) -
                    current_state;
                const double value = forward[previous_start + previous] -
                                     residual * residual * quadratic;
                log_weight[static_cast<std::size_t>(previous)] = value;
                maximum = std::max(maximum, value);
            }
            double exponential_sum = 0.0;
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                exponential_sum += std::exp(
                    log_weight[static_cast<std::size_t>(previous)] - maximum);
            }
            const double transformed = maximum + std::log(exponential_sum);
            forward[current_start + current] =
                structural_gain[current_start + current] / structural_temperature +
                transformed;
        }
    }

    const std::int64_t last_start = offsets[state_count - 1];
    const std::int64_t last_stop = offsets[state_count];
    const double terminal_offset = path_degree[state_count] - path_target[state_count];
    double terminal_maximum = -std::numeric_limits<double>::infinity();
    for (std::int64_t state = last_start; state < last_stop; ++state) {
        const double residual = terminal_offset + static_cast<double>(states[state]);
        const double value = forward[state] - residual * residual * quadratic;
        log_weight[static_cast<std::size_t>(state - last_start)] = value;
        terminal_maximum = std::max(terminal_maximum, value);
    }
    double terminal_sum = 0.0;
    for (std::int64_t state = last_start; state < last_stop; ++state) {
        terminal_sum += std::exp(
            log_weight[static_cast<std::size_t>(state - last_start)] - terminal_maximum);
    }
    const double normalizer = terminal_maximum + std::log(terminal_sum);
    if (!std::isfinite(normalizer)) {
        return 4;
    }
    *log_normalizer = normalizer;
    *pair_work = total_pair_work;
    *linear_log_squared_budget = total_budget;
    return 0;
}

// Sequential scalar semiring moments on an already admitted ragged chain.
// Additives are channel-major.  The loop order is part of the published
// binary64 law and therefore deliberately remains scalar and deterministic.
ROUTE_S_EXPORT std::int32_t route_s_scalar_chain_moments(
    std::int64_t state_count,
    std::int64_t channel_count,
    const std::int64_t* offsets,
    const std::int64_t* states,
    const double* forward,
    const double* structural_gain,
    const double* additive,
    const double* path_degree,
    const double* path_target,
    double inverse_two_variance,
    double coefficient,
    double expected_log_normalizer,
    double* out_mean,
    double* out_second,
    double* out_degree_cross,
    double* out_degree_mean,
    double* out_degree_second,
    double* out_maximum_forward_error,
    double* out_log_normalizer,
    std::int64_t* out_transition_work) {
    if (state_count <= 0 || channel_count <= 0 || offsets == nullptr ||
        states == nullptr || forward == nullptr || structural_gain == nullptr ||
        additive == nullptr || path_degree == nullptr || path_target == nullptr ||
        out_mean == nullptr || out_second == nullptr ||
        out_degree_cross == nullptr || out_degree_mean == nullptr ||
        out_degree_second == nullptr || out_maximum_forward_error == nullptr ||
        out_log_normalizer == nullptr || out_transition_work == nullptr ||
        !std::isfinite(inverse_two_variance) || inverse_two_variance <= 0.0 ||
        !std::isfinite(coefficient) || !std::isfinite(expected_log_normalizer) ||
        offsets[0] != 0) {
        return 1;
    }
    const std::int64_t total_states = offsets[state_count];
    std::int64_t maximum_width = 0;
    for (std::int64_t index = 0; index < state_count; ++index) {
        const std::int64_t width = offsets[index + 1] - offsets[index];
        if (width <= 0) {
            return 2;
        }
        maximum_width = std::max(maximum_width, width);
    }
    const std::size_t scalar_width = static_cast<std::size_t>(maximum_width);
    const std::size_t channel_width =
        static_cast<std::size_t>(channel_count) * scalar_width;
    std::vector<double> h_mean_a(channel_width);
    std::vector<double> h_mean_b(channel_width);
    std::vector<double> h_second_a(channel_width);
    std::vector<double> h_second_b(channel_width);
    std::vector<double> hd_a(channel_width);
    std::vector<double> hd_b(channel_width);
    std::vector<double> d_mean_a(scalar_width);
    std::vector<double> d_mean_b(scalar_width);
    std::vector<double> d_second_a(scalar_width);
    std::vector<double> d_second_b(scalar_width);
    std::vector<double> probability(scalar_width);
    std::vector<double> degree_added(scalar_width);

    const std::int64_t first_width = offsets[1];
    const double first_offset = path_degree[0] - path_target[0];
    for (std::int64_t state = 0; state < first_width; ++state) {
        const double residual = first_offset - static_cast<double>(states[state]);
        const double degree_energy =
            residual * residual * inverse_two_variance;
        d_mean_a[static_cast<std::size_t>(state)] = degree_energy;
        d_second_a[static_cast<std::size_t>(state)] =
            degree_energy * degree_energy;
        for (std::int64_t channel = 0; channel < channel_count; ++channel) {
            const std::size_t location =
                static_cast<std::size_t>(channel) * scalar_width +
                static_cast<std::size_t>(state);
            const double score = additive[
                static_cast<std::size_t>(channel) *
                    static_cast<std::size_t>(total_states) +
                static_cast<std::size_t>(state)
            ];
            h_mean_a[location] = score;
            h_second_a[location] = score * score;
            hd_a[location] = score * degree_energy;
        }
    }

    double maximum_forward_error = 0.0;
    std::int64_t transition_work = 0;
    std::int64_t previous_width = first_width;
    for (std::int64_t index = 1; index < state_count; ++index) {
        const std::int64_t previous_offset = offsets[index - 1];
        const std::int64_t current_offset = offsets[index];
        const std::int64_t current_width = offsets[index + 1] - current_offset;
        const double node_offset = path_degree[index] - path_target[index];
        transition_work += previous_width * current_width;
        for (std::int64_t current = 0; current < current_width; ++current) {
            const double current_state =
                static_cast<double>(states[current_offset + current]);
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                const double residual =
                    node_offset +
                    static_cast<double>(states[previous_offset + previous]) -
                    current_state;
                const double added =
                    residual * residual * inverse_two_variance;
                degree_added[static_cast<std::size_t>(previous)] = added;
                const double value = forward[previous_offset + previous] -
                                     coefficient * added;
                maximum = std::max(maximum, value);
            }
            if (!std::isfinite(maximum)) {
                return 3;
            }
            double exponential_sum = 0.0;
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                const double value = forward[previous_offset + previous] -
                    coefficient * degree_added[static_cast<std::size_t>(previous)];
                const double mass = std::exp(value - maximum);
                probability[static_cast<std::size_t>(previous)] = mass;
                exponential_sum += mass;
            }
            const double normalizer = maximum + std::log(exponential_sum);
            if (!std::isfinite(normalizer)) {
                return 3;
            }
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                probability[static_cast<std::size_t>(previous)] /= exponential_sum;
            }
            const double reconstructed =
                normalizer + structural_gain[current_offset + current];
            maximum_forward_error = std::max(
                maximum_forward_error,
                std::abs(reconstructed - forward[current_offset + current])
            );
            double degree_mean_sum = 0.0;
            double degree_second_sum = 0.0;
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                const std::size_t position = static_cast<std::size_t>(previous);
                const double mass = probability[position];
                const double added = degree_added[position];
                const double old_mean = d_mean_a[position];
                degree_mean_sum += mass * (old_mean + added);
                degree_second_sum += mass * (
                    d_second_a[position] + 2.0 * old_mean * added + added * added
                );
            }
            d_mean_b[static_cast<std::size_t>(current)] = degree_mean_sum;
            d_second_b[static_cast<std::size_t>(current)] = degree_second_sum;
            for (std::int64_t channel = 0; channel < channel_count; ++channel) {
                const std::size_t channel_base =
                    static_cast<std::size_t>(channel) * scalar_width;
                const double added_score = additive[
                    static_cast<std::size_t>(channel) *
                        static_cast<std::size_t>(total_states) +
                    static_cast<std::size_t>(current_offset + current)
                ];
                double score_mean_sum = 0.0;
                double score_second_sum = 0.0;
                double score_degree_sum = 0.0;
                for (std::int64_t previous = 0;
                     previous < previous_width; ++previous) {
                    const std::size_t position = static_cast<std::size_t>(previous);
                    const double mass = probability[position];
                    const double old_score = h_mean_a[channel_base + position];
                    const double old_second = h_second_a[channel_base + position];
                    const double old_cross = hd_a[channel_base + position];
                    const double old_degree = d_mean_a[position];
                    const double added_degree = degree_added[position];
                    score_mean_sum += mass * (old_score + added_score);
                    score_second_sum += mass * (
                        old_second + 2.0 * old_score * added_score +
                        added_score * added_score
                    );
                    score_degree_sum += mass * (
                        old_cross + added_degree * old_score +
                        added_score * old_degree + added_score * added_degree
                    );
                }
                const std::size_t output =
                    channel_base + static_cast<std::size_t>(current);
                h_mean_b[output] = score_mean_sum;
                h_second_b[output] = score_second_sum;
                hd_b[output] = score_degree_sum;
            }
        }
        h_mean_a.swap(h_mean_b);
        h_second_a.swap(h_second_b);
        hd_a.swap(hd_b);
        d_mean_a.swap(d_mean_b);
        d_second_a.swap(d_second_b);
        previous_width = current_width;
    }

    const std::int64_t last_offset = offsets[state_count - 1];
    const std::int64_t last_width = offsets[state_count] - last_offset;
    const double terminal_offset =
        path_degree[state_count] - path_target[state_count];
    double terminal_maximum = -std::numeric_limits<double>::infinity();
    for (std::int64_t state = 0; state < last_width; ++state) {
        const double residual = terminal_offset +
            static_cast<double>(states[last_offset + state]);
        const double terminal_degree =
            residual * residual * inverse_two_variance;
        degree_added[static_cast<std::size_t>(state)] = terminal_degree;
        terminal_maximum = std::max(
            terminal_maximum,
            forward[last_offset + state] - coefficient * terminal_degree
        );
    }
    if (!std::isfinite(terminal_maximum)) {
        return 3;
    }
    double terminal_sum = 0.0;
    for (std::int64_t state = 0; state < last_width; ++state) {
        const double value = forward[last_offset + state] -
            coefficient * degree_added[static_cast<std::size_t>(state)];
        const double mass = std::exp(value - terminal_maximum);
        probability[static_cast<std::size_t>(state)] = mass;
        terminal_sum += mass;
    }
    const double log_normalizer = terminal_maximum + std::log(terminal_sum);
    for (std::int64_t state = 0; state < last_width; ++state) {
        probability[static_cast<std::size_t>(state)] /= terminal_sum;
    }
    double degree_mean = 0.0;
    double degree_second = 0.0;
    for (std::int64_t state = 0; state < last_width; ++state) {
        const std::size_t position = static_cast<std::size_t>(state);
        const double mass = probability[position];
        const double terminal_degree = degree_added[position];
        degree_mean += mass * (d_mean_a[position] + terminal_degree);
        degree_second += mass * (
            d_second_a[position] + 2.0 * d_mean_a[position] * terminal_degree +
            terminal_degree * terminal_degree
        );
    }
    for (std::int64_t channel = 0; channel < channel_count; ++channel) {
        const std::size_t channel_base =
            static_cast<std::size_t>(channel) * scalar_width;
        double score_mean = 0.0;
        double score_second = 0.0;
        double score_degree = 0.0;
        for (std::int64_t state = 0; state < last_width; ++state) {
            const std::size_t position = static_cast<std::size_t>(state);
            const double mass = probability[position];
            const double mean_value = h_mean_a[channel_base + position];
            score_mean += mass * mean_value;
            score_second += mass * h_second_a[channel_base + position];
            score_degree += mass * (
                hd_a[channel_base + position] + mean_value * degree_added[position]
            );
        }
        out_mean[channel] = score_mean;
        out_second[channel] = score_second;
        out_degree_cross[channel] = score_degree;
    }
    *out_degree_mean = degree_mean;
    *out_degree_second = degree_second;
    *out_maximum_forward_error = maximum_forward_error;
    *out_log_normalizer = log_normalizer;
    *out_transition_work = transition_work;
    return 0;
}

// Exact one-channel expectation-semiring recurrence.  It forms the same base
// Gaussian forward law and the score/degree raw moments in one traversal.  The
// optional forward output exists for mathematical verification; production
// spectral calibration consumes only the normalizer and raw moments.
ROUTE_S_EXPORT std::int32_t route_s_fused_one_channel(
    std::int64_t state_count,
    const std::int64_t* offsets,
    const std::int64_t* states,
    const double* structural_gain,
    const double* additive,
    const double* path_degree,
    const double* path_target,
    double inverse_two_variance,
    double coefficient,
    std::int32_t store_forward,
    double* forward,
    double* out_score_mean,
    double* out_score_second,
    double* out_score_degree,
    double* out_degree_mean,
    double* out_degree_second,
    double* out_log_normalizer,
    std::int64_t* out_transition_work) {
    if (state_count <= 0 || offsets == nullptr || states == nullptr ||
        structural_gain == nullptr || additive == nullptr || path_degree == nullptr ||
        path_target == nullptr || forward == nullptr || out_score_mean == nullptr ||
        out_score_second == nullptr || out_score_degree == nullptr ||
        out_degree_mean == nullptr || out_degree_second == nullptr ||
        out_log_normalizer == nullptr || out_transition_work == nullptr ||
        !std::isfinite(inverse_two_variance) || inverse_two_variance <= 0.0 ||
        !std::isfinite(coefficient) || coefficient <= 0.0 ||
        (store_forward != 0 && store_forward != 1) || offsets[0] != 0) {
        return 1;
    }

    std::int64_t maximum_width = 0;
    for (std::int64_t index = 0; index < state_count; ++index) {
        const std::int64_t width = offsets[index + 1] - offsets[index];
        if (width <= 0) return 2;
        maximum_width = std::max(maximum_width, width);
    }
    const std::size_t width = static_cast<std::size_t>(maximum_width);
    std::vector<double> score_mean_a(width);
    std::vector<double> score_mean_b(width);
    std::vector<double> score_second_a(width);
    std::vector<double> score_second_b(width);
    std::vector<double> score_degree_a(width);
    std::vector<double> score_degree_b(width);
    std::vector<double> degree_mean_a(width);
    std::vector<double> degree_mean_b(width);
    std::vector<double> degree_second_a(width);
    std::vector<double> degree_second_b(width);
    std::vector<double> probability(width);
    std::vector<double> degree_added(width);
    std::vector<double> forward_a(width);
    std::vector<double> forward_b(width);

    const std::int64_t first_width = offsets[1];
    const double first_offset = path_degree[0] - path_target[0];
    for (std::int64_t state = 0; state < first_width; ++state) {
        const double residual = first_offset - static_cast<double>(states[state]);
        const double degree_energy =
            residual * residual * inverse_two_variance;
        const double score = additive[state];
        const double message =
            structural_gain[state] - residual * residual * coefficient;
        forward_a[static_cast<std::size_t>(state)] = message;
        if (store_forward != 0) forward[state] = message;
        score_mean_a[static_cast<std::size_t>(state)] = score;
        score_second_a[static_cast<std::size_t>(state)] = score * score;
        score_degree_a[static_cast<std::size_t>(state)] = score * degree_energy;
        degree_mean_a[static_cast<std::size_t>(state)] = degree_energy;
        degree_second_a[static_cast<std::size_t>(state)] =
            degree_energy * degree_energy;
    }

    std::int64_t transition_work = 0;
    std::int64_t previous_width = first_width;
    for (std::int64_t index = 1; index < state_count; ++index) {
        const std::int64_t previous_offset = offsets[index - 1];
        const std::int64_t current_offset = offsets[index];
        const std::int64_t current_width = offsets[index + 1] - current_offset;
        const double node_offset = path_degree[index] - path_target[index];
        if (previous_width >
            std::numeric_limits<std::int64_t>::max() / current_width) {
            return 3;
        }
        transition_work += previous_width * current_width;
        for (std::int64_t current = 0; current < current_width; ++current) {
            const double current_state =
                static_cast<double>(states[current_offset + current]);
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                const double residual =
                    node_offset +
                    static_cast<double>(states[previous_offset + previous]) -
                    current_state;
                const double degree_energy =
                    residual * residual * inverse_two_variance;
                degree_added[static_cast<std::size_t>(previous)] = degree_energy;
                const double value = forward_a[static_cast<std::size_t>(previous)] -
                                     residual * residual * coefficient;
                maximum = std::max(maximum, value);
            }
            if (!std::isfinite(maximum)) return 4;
            double exponential_sum = 0.0;
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                const double residual =
                    node_offset +
                    static_cast<double>(states[previous_offset + previous]) -
                    current_state;
                const double value = forward_a[static_cast<std::size_t>(previous)] -
                                     residual * residual * coefficient;
                const double mass = std::exp(value - maximum);
                probability[static_cast<std::size_t>(previous)] = mass;
                exponential_sum += mass;
            }
            const double normalizer = maximum + std::log(exponential_sum);
            if (!std::isfinite(normalizer)) return 4;
            const double current_message =
                structural_gain[current_offset + current] + normalizer;
            forward_b[static_cast<std::size_t>(current)] = current_message;
            if (store_forward != 0) {
                forward[current_offset + current] = current_message;
            }
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                probability[static_cast<std::size_t>(previous)] /= exponential_sum;
            }

            const double added_score = additive[current_offset + current];
            double score_mean_sum = 0.0;
            double score_second_sum = 0.0;
            double score_degree_sum = 0.0;
            double degree_mean_sum = 0.0;
            double degree_second_sum = 0.0;
            for (std::int64_t previous = 0; previous < previous_width; ++previous) {
                const std::size_t position = static_cast<std::size_t>(previous);
                const double mass = probability[position];
                const double added_degree = degree_added[position];
                const double old_score = score_mean_a[position];
                const double old_degree = degree_mean_a[position];
                score_mean_sum += mass * (old_score + added_score);
                score_second_sum += mass * (
                    score_second_a[position] + 2.0 * old_score * added_score +
                    added_score * added_score
                );
                score_degree_sum += mass * (
                    score_degree_a[position] + added_degree * old_score +
                    added_score * old_degree + added_score * added_degree
                );
                degree_mean_sum += mass * (old_degree + added_degree);
                degree_second_sum += mass * (
                    degree_second_a[position] + 2.0 * old_degree * added_degree +
                    added_degree * added_degree
                );
            }
            const std::size_t output = static_cast<std::size_t>(current);
            score_mean_b[output] = score_mean_sum;
            score_second_b[output] = score_second_sum;
            score_degree_b[output] = score_degree_sum;
            degree_mean_b[output] = degree_mean_sum;
            degree_second_b[output] = degree_second_sum;
        }
        score_mean_a.swap(score_mean_b);
        score_second_a.swap(score_second_b);
        score_degree_a.swap(score_degree_b);
        degree_mean_a.swap(degree_mean_b);
        degree_second_a.swap(degree_second_b);
        forward_a.swap(forward_b);
        previous_width = current_width;
    }

    const std::int64_t last_offset = offsets[state_count - 1];
    const std::int64_t last_width = offsets[state_count] - last_offset;
    const double terminal_offset =
        path_degree[state_count] - path_target[state_count];
    double terminal_maximum = -std::numeric_limits<double>::infinity();
    for (std::int64_t state = 0; state < last_width; ++state) {
        const double residual =
            terminal_offset + static_cast<double>(states[last_offset + state]);
        const double degree_energy =
            residual * residual * inverse_two_variance;
        degree_added[static_cast<std::size_t>(state)] = degree_energy;
        terminal_maximum = std::max(
            terminal_maximum,
            forward_a[static_cast<std::size_t>(state)] -
            residual * residual * coefficient
        );
    }
    if (!std::isfinite(terminal_maximum)) return 4;
    double terminal_sum = 0.0;
    for (std::int64_t state = 0; state < last_width; ++state) {
        const double residual =
            terminal_offset + static_cast<double>(states[last_offset + state]);
        const double value = forward_a[static_cast<std::size_t>(state)] -
                             residual * residual * coefficient;
        const double mass = std::exp(value - terminal_maximum);
        probability[static_cast<std::size_t>(state)] = mass;
        terminal_sum += mass;
    }
    const double log_normalizer = terminal_maximum + std::log(terminal_sum);
    if (!std::isfinite(log_normalizer)) return 4;
    for (std::int64_t state = 0; state < last_width; ++state) {
        probability[static_cast<std::size_t>(state)] /= terminal_sum;
    }

    double score_mean = 0.0;
    double score_second = 0.0;
    double score_degree = 0.0;
    double degree_mean = 0.0;
    double degree_second = 0.0;
    for (std::int64_t state = 0; state < last_width; ++state) {
        const std::size_t position = static_cast<std::size_t>(state);
        const double mass = probability[position];
        const double terminal_degree = degree_added[position];
        const double mean_score = score_mean_a[position];
        score_mean += mass * mean_score;
        score_second += mass * score_second_a[position];
        score_degree += mass * (
            score_degree_a[position] + mean_score * terminal_degree
        );
        degree_mean += mass * (degree_mean_a[position] + terminal_degree);
        degree_second += mass * (
            degree_second_a[position] +
            2.0 * degree_mean_a[position] * terminal_degree +
            terminal_degree * terminal_degree
        );
    }
    *out_score_mean = score_mean;
    *out_score_second = score_second;
    *out_score_degree = score_degree;
    *out_degree_mean = degree_mean;
    *out_degree_second = degree_second;
    *out_log_normalizer = log_normalizer;
    *out_transition_work = transition_work;
    return 0;
}
