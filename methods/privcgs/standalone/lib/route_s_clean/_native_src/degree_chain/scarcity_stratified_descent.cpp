#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Exact descent admission factored by the frozen primary scarcity key.  Equal-
// scarcity strata are independent except for reserved dyads and bundle capacity,
// so the global candidate table is never materialized.

namespace {

struct LawView {
    std::int64_t nodes;
    std::int64_t rank;
    std::int64_t parent_count;
    const double* sociability;
    const double* star;
    const double* interaction;
    const double* weights;
    double dust;
    const std::int32_t* parent_fiber;
    const std::int32_t* component_fiber;
    const double* marked_shift;
    const double* component_shift;
    const double* parent_flux;
};

struct ActiveBundle {
    std::int64_t position;
    std::int64_t candidate_count;
    double scarcity;
};

struct Candidate {
    std::int64_t path_index;
    std::int64_t old_left;
    std::int64_t old_right;
    std::int64_t new_left;
    std::int64_t new_right;
    double gain;
};

inline std::uint64_t edge_key(
    std::int64_t n, std::int64_t left, std::int64_t right
) {
    const std::int64_t low = std::min(left, right);
    const std::int64_t high = std::max(left, right);
    return static_cast<std::uint64_t>(low) * static_cast<std::uint64_t>(n) +
           static_cast<std::uint64_t>(high);
}

inline bool contains_source_key(
    const std::uint64_t* keys, std::int64_t count, std::uint64_t value
) {
    return std::binary_search(keys, keys + count, value);
}

inline double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

inline double log_expm1_intensity(double value) {
    const double clipped = std::min(std::max(value, 1.0e-12), 20.0);
    return std::log(std::max(std::expm1(clipped), 1.0e-12));
}

double bilinear_eta(const LawView& law, std::int64_t left, std::int64_t right) {
    const double* x = law.interaction + left * law.rank;
    const double* y = law.interaction + right * law.rank;
    double value = 0.0;
    for (std::int64_t index = 0; index < law.rank; ++index) {
        value += law.weights[index] * x[index] * y[index];
    }
    return value;
}

std::int32_t dyad_component(
    const LawView& law, std::int64_t left, std::int64_t right
) {
    const std::int32_t first = law.component_fiber[left];
    const std::int32_t second = law.component_fiber[right];
    double similarity = 0.0;
    const double* x = law.interaction + left * law.rank;
    const double* y = law.interaction + right * law.rank;
    for (std::int64_t index = 0; index < law.rank; ++index) {
        similarity += x[index] * y[index];
    }
    if (first == second && similarity >= 0.0) return 1;
    if (law.star[left] + law.star[right] >= 2.0) return 0;
    if (law.sociability[left] + law.sociability[right] <= 2.0) return 3;
    return 2;
}

double edge_eta(const LawView& law, std::int64_t left, std::int64_t right) {
    const double pair = softplus(bilinear_eta(law, left, right)) + 1.0e-6;
    const double intensity =
        law.sociability[left] * law.sociability[right] * pair +
        law.sociability[left] * law.star[right] +
        law.sociability[right] * law.star[left] + law.dust;
    const std::int32_t parent_left = law.parent_fiber[left];
    const std::int32_t parent_right = law.parent_fiber[right];
    const std::int32_t component = dyad_component(law, left, right);
    return log_expm1_intensity(intensity) +
           law.marked_shift[parent_left] + law.marked_shift[parent_right] +
           law.parent_flux[
               static_cast<std::int64_t>(parent_left) * law.parent_count + parent_right
           ] +
           law.component_shift[component];
}

bool candidate_less(const Candidate& first, const Candidate& second) {
    if (first.gain != second.gain) return first.gain > second.gain;
    if (first.path_index != second.path_index) {
        return first.path_index < second.path_index;
    }
    if (first.old_left != second.old_left) return first.old_left < second.old_left;
    if (first.old_right != second.old_right) return first.old_right < second.old_right;
    if (first.new_left != second.new_left) return first.new_left < second.new_left;
    return first.new_right < second.new_right;
}

}  // namespace

ROUTE_S_EXPORT std::int32_t route_s_scarcity_stratified_descent(
    std::int64_t n,
    std::int64_t rank,
    std::int64_t parent_count,
    std::int64_t component_count,
    const double* sociability,
    const double* star,
    const double* interaction,
    const double* weights,
    double dust,
    const std::int32_t* parent_fiber,
    const std::int32_t* component_fiber,
    const double* marked_shift,
    const double* component_shift,
    const double* parent_flux,
    const std::int64_t* path,
    const std::int8_t* direction,
    const std::int64_t* requested,
    const std::int64_t* offsets,
    const std::int64_t* neighbors,
    std::int64_t source_edge_count,
    const std::uint64_t* source_edge_keys,
    std::int64_t output_capacity,
    std::int64_t* out_path_index,
    std::int64_t* out_old_left,
    std::int64_t* out_old_right,
    std::int64_t* out_new_left,
    std::int64_t* out_new_right,
    double* out_gain,
    std::int64_t* out_admitted_by_bundle,
    std::int64_t* out_selected_count,
    std::int64_t* diagnostics
) {
    if (n <= 0 || rank <= 0 || parent_count <= 0 || component_count <= 0 ||
        sociability == nullptr || star == nullptr || interaction == nullptr ||
        weights == nullptr || parent_fiber == nullptr || component_fiber == nullptr ||
        marked_shift == nullptr || component_shift == nullptr || parent_flux == nullptr ||
        path == nullptr || direction == nullptr || requested == nullptr ||
        offsets == nullptr || neighbors == nullptr || source_edge_count < 0 ||
        source_edge_keys == nullptr || output_capacity < 0 || out_path_index == nullptr ||
        out_old_left == nullptr || out_old_right == nullptr || out_new_left == nullptr ||
        out_new_right == nullptr || out_gain == nullptr ||
        out_admitted_by_bundle == nullptr || out_selected_count == nullptr ||
        diagnostics == nullptr) {
        return 1;
    }

    try {
        const LawView law{
            n, rank, parent_count, sociability, star, interaction, weights, dust,
            parent_fiber, component_fiber, marked_shift, component_shift, parent_flux,
        };
        const std::int64_t bundle_count = std::max<std::int64_t>(n - 1, 0);
        std::fill(
            out_admitted_by_bundle,
            out_admitted_by_bundle + bundle_count,
            std::int64_t{0}
        );
        std::fill(diagnostics, diagnostics + 10, std::int64_t{0});

        std::vector<ActiveBundle> active;
        active.reserve(static_cast<std::size_t>(bundle_count));
        std::int64_t total_candidate_count = 0;
        for (std::int64_t position = 0; position < bundle_count; ++position) {
            if (requested[position] <= 0 || direction[position] == 0) continue;
            const std::int64_t left = path[position];
            const std::int64_t right = path[position + 1];
            if (left < 0 || left >= n || right < 0 || right >= n) return 2;
            const std::int64_t donor = direction[position] > 0 ? left : right;
            const std::int64_t receiver = direction[position] > 0 ? right : left;
            std::int64_t local_count = 0;
            for (
                std::int64_t cursor = offsets[donor];
                cursor < offsets[donor + 1];
                ++cursor
            ) {
                const std::int64_t neighbor = neighbors[cursor];
                if (neighbor == receiver) continue;
                if (!contains_source_key(
                        source_edge_keys,
                        source_edge_count,
                        edge_key(n, receiver, neighbor)
                    )) {
                    ++local_count;
                }
            }
            if (local_count > 0) {
                total_candidate_count += local_count;
                active.push_back(ActiveBundle{
                    position,
                    local_count,
                    static_cast<double>(local_count) /
                        static_cast<double>(requested[position]),
                });
            }
        }

        std::sort(active.begin(), active.end(), [](const ActiveBundle& first,
                                                   const ActiveBundle& second) {
            if (first.scarcity != second.scarcity) {
                return first.scarcity < second.scarcity;
            }
            return first.position < second.position;
        });

        std::int64_t maximum_stratum_count = 0;
        for (std::size_t begin = 0; begin < active.size();) {
            std::size_t end = begin + 1;
            while (end < active.size() && active[end].scarcity == active[begin].scarcity) {
                ++end;
            }
            std::int64_t stratum_count = 0;
            for (std::size_t index = begin; index < end; ++index) {
                stratum_count += active[index].candidate_count;
            }
            maximum_stratum_count = std::max(maximum_stratum_count, stratum_count);
            begin = end;
        }

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<std::size_t>(maximum_stratum_count));
        std::unordered_set<std::uint64_t> reserved;
        const std::int64_t reserve_hint = std::min(
            total_candidate_count,
            output_capacity > std::numeric_limits<std::int64_t>::max() / 2
                ? total_candidate_count
                : output_capacity * 2
        );
        reserved.reserve(
            static_cast<std::size_t>(std::max<std::int64_t>(reserve_hint, 1))
        );

        std::int64_t selected_count = 0;
        std::int64_t stratum_total = 0;
        std::int64_t examined_total = 0;
        std::int64_t unread_total = 0;
        std::int64_t direct_eta_calls = 0;
        for (std::size_t begin = 0; begin < active.size();) {
            std::size_t end = begin + 1;
            while (end < active.size() && active[end].scarcity == active[begin].scarcity) {
                ++end;
            }
            ++stratum_total;
            candidates.clear();
            std::int64_t remaining_capacity = 0;
            std::int64_t expected_stratum_count = 0;
            for (std::size_t bundle_index = begin; bundle_index < end; ++bundle_index) {
                const std::int64_t position = active[bundle_index].position;
                remaining_capacity += requested[position];
                expected_stratum_count += active[bundle_index].candidate_count;
                const std::int64_t left = path[position];
                const std::int64_t right = path[position + 1];
                const std::int64_t donor = direction[position] > 0 ? left : right;
                const std::int64_t receiver = direction[position] > 0 ? right : left;
                for (
                    std::int64_t cursor = offsets[donor];
                    cursor < offsets[donor + 1];
                    ++cursor
                ) {
                    const std::int64_t neighbor = neighbors[cursor];
                    if (neighbor == receiver) continue;
                    if (contains_source_key(
                            source_edge_keys,
                            source_edge_count,
                            edge_key(n, receiver, neighbor)
                        )) {
                        continue;
                    }
                    const std::int64_t old_left = std::min(donor, neighbor);
                    const std::int64_t old_right = std::max(donor, neighbor);
                    const std::int64_t new_left = std::min(receiver, neighbor);
                    const std::int64_t new_right = std::max(receiver, neighbor);
                    const double gain =
                        edge_eta(law, new_left, new_right) -
                        edge_eta(law, old_left, old_right);
                    if (!std::isfinite(gain)) return 3;
                    direct_eta_calls += 2;
                    candidates.push_back(Candidate{
                        position, old_left, old_right, new_left, new_right, gain,
                    });
                }
            }
            if (static_cast<std::int64_t>(candidates.size()) != expected_stratum_count) {
                return 4;
            }
            std::sort(candidates.begin(), candidates.end(), candidate_less);
            std::int64_t examined_in_stratum = 0;
            for (const Candidate& candidate : candidates) {
                ++examined_in_stratum;
                const std::int64_t bundle = candidate.path_index;
                if (out_admitted_by_bundle[bundle] >= requested[bundle]) continue;
                const std::uint64_t old_key =
                    edge_key(n, candidate.old_left, candidate.old_right);
                const std::uint64_t new_key =
                    edge_key(n, candidate.new_left, candidate.new_right);
                if (reserved.find(old_key) != reserved.end() ||
                    reserved.find(new_key) != reserved.end()) {
                    continue;
                }
                if (selected_count >= output_capacity) return 5;
                reserved.insert(old_key);
                reserved.insert(new_key);
                ++out_admitted_by_bundle[bundle];
                --remaining_capacity;
                out_path_index[selected_count] = candidate.path_index;
                out_old_left[selected_count] = candidate.old_left;
                out_old_right[selected_count] = candidate.old_right;
                out_new_left[selected_count] = candidate.new_left;
                out_new_right[selected_count] = candidate.new_right;
                out_gain[selected_count] = candidate.gain;
                ++selected_count;
                if (remaining_capacity == 0) break;
            }
            examined_total += examined_in_stratum;
            unread_total +=
                static_cast<std::int64_t>(candidates.size()) - examined_in_stratum;
            begin = end;
        }

        *out_selected_count = selected_count;
        diagnostics[0] = total_candidate_count;
        diagnostics[1] = static_cast<std::int64_t>(active.size());
        diagnostics[2] = stratum_total;
        diagnostics[3] = maximum_stratum_count;
        diagnostics[4] = selected_count;
        diagnostics[5] = static_cast<std::int64_t>(reserved.size());
        diagnostics[6] = examined_total;
        diagnostics[7] = unread_total;
        diagnostics[8] =
            maximum_stratum_count * static_cast<std::int64_t>(sizeof(Candidate));
        diagnostics[9] = direct_eta_calls;
        return 0;
    } catch (...) {
        return 99;
    }
}
