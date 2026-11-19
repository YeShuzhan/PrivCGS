#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

namespace {

struct HeapItem {
    double negative_score;
    std::int64_t index;
    std::int64_t version;
};

struct PythonTupleGreater {
    bool operator()(const HeapItem& left, const HeapItem& right) const noexcept {
        if (left.negative_score != right.negative_score) {
            return left.negative_score > right.negative_score;
        }
        if (left.index != right.index) {
            return left.index > right.index;
        }
        return left.version > right.version;
    }
};

inline std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// CPython's float power operator reaches the platform pow(double,double)
// implementation.  Calling through a volatile function pointer prevents GCC
// from strength-reducing exponent 2.0 to multiplication, which can change the
// cancellation bits in the subsequent four-term degree-energy expression.
inline double python_float_square(double value) noexcept {
    using PowFunction = double (*)(double, double);
    static PowFunction volatile runtime_pow = static_cast<PowFunction>(std::pow);
    return runtime_pow(value, 2.0);
}

}  // namespace

extern "C" __declspec(dllexport) std::int32_t
route_s_positive_hamiltonian_heap_v1(
    std::int64_t bundle_count,
    std::int64_t node_count,
    const std::int64_t* order,
    const double* degree,
    const double* target,
    double denominator,
    double residual_scale,
    double coefficient,
    const std::int64_t* sign,
    const std::int64_t* core_capacity,
    const std::int64_t* negative_offsets,
    const double* negative_spectral,
    const std::int64_t* positive_offsets,
    const double* positive_spectral,
    std::int64_t* accepted,
    double* output_statistics,
    std::int64_t* output_diagnostics,
    std::int64_t audit_capacity,
    std::int64_t* audit_accepted_index,
    std::uint64_t* audit_accepted_gain_bits) {
    if (bundle_count < 0 || node_count < 0 || denominator <= 0.0 ||
        residual_scale <= 0.0) {
        return 1;
    }
    try {
        std::vector<double> realized(degree, degree + node_count);
        std::vector<std::int64_t> flow(static_cast<std::size_t>(bundle_count));
        std::vector<std::int64_t> versions(
            static_cast<std::size_t>(bundle_count), 0);
        std::fill(accepted, accepted + bundle_count, std::int64_t{0});
        for (std::int64_t index = 0; index < bundle_count; ++index) {
            const std::int64_t value = sign[index] * core_capacity[index];
            flow[static_cast<std::size_t>(index)] = value;
            const std::int64_t left = order[index];
            const std::int64_t right = order[index + 1];
            if (left < 0 || left >= node_count || right < 0 || right >= node_count) {
                return 2;
            }
            realized[static_cast<std::size_t>(left)] -= static_cast<double>(value);
            realized[static_cast<std::size_t>(right)] += static_cast<double>(value);
        }

        auto score = [&](std::int64_t index, double& output) -> bool {
            ++output_diagnostics[4];
            const std::int64_t position = accepted[index];
            double spectral = 0.0;
            if (sign[index] > 0) {
                const std::int64_t start = positive_offsets[index];
                const std::int64_t stop = positive_offsets[index + 1];
                if (start + position >= stop) {
                    return false;
                }
                spectral = positive_spectral[start + position];
            } else if (sign[index] < 0) {
                const std::int64_t start = negative_offsets[index];
                const std::int64_t stop = negative_offsets[index + 1];
                if (start + position >= stop) {
                    return false;
                }
                spectral = negative_spectral[start + position];
            } else {
                return false;
            }
            const std::int64_t direction = sign[index];
            const std::int64_t left = order[index];
            const std::int64_t right = order[index + 1];
            const double left_residual =
                realized[static_cast<std::size_t>(left)] - target[left];
            const double right_residual =
                realized[static_cast<std::size_t>(right)] - target[right];
            const double left_after = left_residual - static_cast<double>(direction);
            const double right_after = right_residual + static_cast<double>(direction);
            const double left_after_square = python_float_square(left_after);
            const double left_before_square = python_float_square(left_residual);
            const double right_after_square = python_float_square(right_after);
            const double right_before_square = python_float_square(right_residual);
            double delta_degree = left_after_square - left_before_square;
            delta_degree = delta_degree + right_after_square;
            delta_degree = delta_degree - right_before_square;
            delta_degree = delta_degree / denominator;
            const double normalized_spectral = spectral / residual_scale;
            const double degree_penalty = coefficient * delta_degree;
            output = normalized_spectral - degree_penalty;
            return true;
        };

        std::priority_queue<
            HeapItem, std::vector<HeapItem>, PythonTupleGreater> heap;
        auto push = [&](std::int64_t index) {
            if (index < 0 || index >= bundle_count) {
                return;
            }
            double value = 0.0;
            if (score(index, value)) {
                heap.push(HeapItem{-value, index, versions[index]});
                ++output_diagnostics[0];
                output_diagnostics[6] = std::max<std::int64_t>(
                    output_diagnostics[6], static_cast<std::int64_t>(heap.size()));
            }
        };
        for (std::int64_t index = 0; index < bundle_count; ++index) {
            push(index);
        }

        double minimum_accepted = std::numeric_limits<double>::infinity();
        double terminal_maximum = -std::numeric_limits<double>::infinity();
        std::int64_t accepted_total = 0;
        while (!heap.empty()) {
            const HeapItem item = heap.top();
            heap.pop();
            ++output_diagnostics[1];
            if (item.version != versions[item.index]) {
                ++output_diagnostics[2];
                continue;
            }
            double value = 0.0;
            if (!score(item.index, value)) {
                ++output_diagnostics[3];
                continue;
            }
            if (value <= 0.0) {
                terminal_maximum = value;
                break;
            }
            if (accepted_total >= audit_capacity && audit_capacity > 0) {
                return 3;
            }
            if (audit_capacity > 0) {
                audit_accepted_index[accepted_total] = item.index;
                audit_accepted_gain_bits[accepted_total] = double_bits(value);
            }
            minimum_accepted = std::min(minimum_accepted, value);
            ++accepted_total;
            ++accepted[item.index];
            const std::int64_t direction = sign[item.index];
            flow[static_cast<std::size_t>(item.index)] += direction;
            const std::int64_t left = order[item.index];
            const std::int64_t right = order[item.index + 1];
            realized[static_cast<std::size_t>(left)] -= static_cast<double>(direction);
            realized[static_cast<std::size_t>(right)] += static_cast<double>(direction);
            for (std::int64_t affected = item.index - 1;
                 affected <= item.index + 1; ++affected) {
                if (0 <= affected && affected < bundle_count) {
                    ++versions[affected];
                    push(affected);
                }
            }
        }
        output_statistics[0] =
            accepted_total == 0 ? 0.0 : minimum_accepted;
        output_statistics[1] = terminal_maximum;
        output_diagnostics[5] = accepted_total;
        output_diagnostics[7] = static_cast<std::int64_t>(heap.size());
        return 0;
    } catch (const std::bad_alloc&) {
        return 4;
    } catch (...) {
        return 5;
    }
}
