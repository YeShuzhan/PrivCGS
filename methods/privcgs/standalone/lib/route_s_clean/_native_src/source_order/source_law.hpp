#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C"
#endif

namespace {

constexpr std::uint64_t GOLDEN64 = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t PATH_MIX = 0xD2B74407B1CE6E93ULL;
constexpr std::uint64_t DRAW_MIX = 0xCA5A826395121157ULL;
constexpr std::uint64_t GUMBEL_DOMAIN = 0x56313032ULL;
constexpr std::uint64_t MARK_DOMAIN = 0x4D41524BULL;

// Semantic diagnostics for the clean RouteS source-order executor.  The frozen
// ABI only exposes five integer slots, one of which used to mean the maximum
// open priority queue.  Frontier-free Top-K has no such queue, so keep its
// distinct runtime quantities here instead of overloading that old label.
thread_local std::int64_t last_candidate_prefix_limit = 0;
thread_local std::int64_t last_maximum_materialized_segment_size = 0;
thread_local std::int64_t last_maximum_dfs_stack_size = 0;
thread_local std::int64_t last_topk_round_count = 0;
thread_local std::int64_t last_used_frontier_free_topk = 0;
thread_local std::int64_t last_used_monolithic_fallback = 0;
thread_local std::int64_t last_binary64_serial_tie_comparison_count = 0;
thread_local std::int64_t last_tie_cache_certificate_build_count = 0;
thread_local std::int64_t last_tie_cache_certificate_reuse_count = 0;
thread_local std::int64_t last_tie_cache_maximum_ready_pool_size = 0;

inline std::uint64_t splitmix64(std::uint64_t value) {
    std::uint64_t z = value + GOLDEN64;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline double uniform53(
    std::uint64_t seed, std::uint64_t path_hash, std::uint64_t draw_index
) {
    const std::uint64_t identity = seed ^ path_hash * PATH_MIX ^ draw_index * DRAW_MIX;
    const std::uint64_t mantissa = splitmix64(identity) >> 11;
    return (static_cast<double>(mantissa) + 0.5) /
           static_cast<double>(std::uint64_t(1) << 53);
}

inline std::uint32_t extend_path(std::uint32_t value, std::uint32_t item) {
    std::uint32_t output = value;
    output ^= item + 0x9E3779B9U + (output << 6) + (output >> 2);
    return output;
}

inline double gumbel(double uniform, double location) {
    return location - std::log(-std::log(uniform));
}

inline double truncated_gumbel(double uniform, double location, double ceiling) {
    const double log_cdf_ceiling = -std::exp(-(ceiling - location));
    const double log_cdf_draw = std::log(uniform) + log_cdf_ceiling;
    return location - std::log(-log_cdf_draw);
}

inline double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

inline double log_expm1_intensity(double value) {
    const double clipped = std::min(std::max(value, 1.0e-12), 20.0);
    return std::log(std::max(std::expm1(clipped), 1.0e-12));
}

struct TreeView {
    int count;
    int rank;
    const int* child0;
    const int* child1;
    const int* size;
    const int* singleton;
    const double* low;
    const double* high;
    const double* sociability_max;
    const double* star_max;
    const std::uint64_t* parent_mask;
};

struct LawView {
    int nodes;
    int rank;
    int parent_count;
    int component_count;
    const double* sociability;
    const double* star;
    const double* interaction;
    const double* weights;
    double dust;
    const int* parent_fiber;
    const int* component_fiber;
    const double* marked_shift;
    const double* component_shift;
    const double* parent_flux;
    double component_upper;
};

struct Cell {
    int left;
    int right;
    std::uint32_t path_hash;
    double maximum;
    double priority;
    std::uint64_t serial;
};

struct CellLess {
    bool operator()(const Cell& first, const Cell& second) const {
        if (first.priority != second.priority) {
            return first.priority < second.priority;
        }
        return first.serial > second.serial;
    }
};

// A segmented binary heap avoids std::vector's geometric spare capacity and
// reallocation peak.  The strict order is identical to std::priority_queue:
// larger priority first, then smaller serial first for an exact tie.
inline bool cell_higher(const Cell& first, const Cell& second) {
    if (first.priority != second.priority) {
        return first.priority > second.priority;
    }
    return first.serial < second.serial;
}

// Exact external-merge priority queue for the million-node regime.  Each full
// insertion buffer becomes one descending immutable run.  The next cell is the
// maximum of the in-memory buffer head and all run heads, so the priority/serial
// order is identical to one monolithic heap.  Runs are sequential temporary
// files and are deleted as soon as they are exhausted.
inline bool singleton(const TreeView& tree, int node) {
    return tree.size[node] == 1;
}

double bilinear_eta(const LawView& law, int left, int right) {
    const double* x = law.interaction + static_cast<std::int64_t>(left) * law.rank;
    const double* y = law.interaction + static_cast<std::int64_t>(right) * law.rank;
    double value = 0.0;
    for (int index = 0; index < law.rank; ++index) {
        value += law.weights[index] * x[index] * y[index];
    }
    return value;
}

int dyad_component(const LawView& law, int left, int right) {
    const int first = law.component_fiber[left];
    const int second = law.component_fiber[right];
    double similarity = 0.0;
    const double* x = law.interaction + static_cast<std::int64_t>(left) * law.rank;
    const double* y = law.interaction + static_cast<std::int64_t>(right) * law.rank;
    for (int index = 0; index < law.rank; ++index) {
        similarity += x[index] * y[index];
    }
    if (first == second && similarity >= 0.0) {
        return 1;
    }
    if (law.star[left] + law.star[right] >= 2.0) {
        return 0;
    }
    if (law.sociability[left] + law.sociability[right] <= 2.0) {
        return 3;
    }
    return 2;
}

double edge_eta(const LawView& law, int left, int right) {
    ObserveSample sample(4,(observe_eta_calls&4095)==0);++observe_eta_calls;
    const double pair = softplus(bilinear_eta(law, left, right)) + 1.0e-6;
    const double intensity =
        law.sociability[left] * law.sociability[right] * pair +
        law.sociability[left] * law.star[right] +
        law.sociability[right] * law.star[left] + law.dust;
    const int parent_left = law.parent_fiber[left];
    const int parent_right = law.parent_fiber[right];
    const int component = dyad_component(law, left, right);
    if(intensity>=20.0)++observe_eta_saturated;
    return log_expm1_intensity(intensity) +
           law.marked_shift[parent_left] + law.marked_shift[parent_right] +
           law.parent_flux[parent_left * law.parent_count + parent_right] +
           law.component_shift[component];
}

double marked_upper(const LawView& law, std::uint64_t mask) {
    double value = -std::numeric_limits<double>::infinity();
    for (int parent = 0; parent < law.parent_count; ++parent) {
        if ((mask >> parent) & 1ULL) {
            value = std::max(value, law.marked_shift[parent]);
        }
    }
    return value;
}

double parent_upper(
    const LawView& law, std::uint64_t left_mask, std::uint64_t right_mask
) {
    double value = -std::numeric_limits<double>::infinity();
    for (int left = 0; left < law.parent_count; ++left) {
        if (!((left_mask >> left) & 1ULL)) {
            continue;
        }
        for (int right = 0; right < law.parent_count; ++right) {
            if ((right_mask >> right) & 1ULL) {
                value = std::max(
                    value, law.parent_flux[left * law.parent_count + right]
                );
            }
        }
    }
    return value;
}

#include "phase_query_cache.inc"

double cell_upper(
    const LawView& law, const TreeView& tree, int left, int right
) {
    if (singleton(tree, left) && singleton(tree, right)) {
        return edge_eta(law, tree.singleton[left], tree.singleton[right]);
    }
    const double* left_low = tree.low + static_cast<std::int64_t>(left) * law.rank;
    const double* left_high = tree.high + static_cast<std::int64_t>(left) * law.rank;
    const double* right_low = tree.low + static_cast<std::int64_t>(right) * law.rank;
    const double* right_high = tree.high + static_cast<std::int64_t>(right) * law.rank;
    double bilinear = 0.0;
    for (int index = 0; index < law.rank; ++index) {
        const double first = left_low[index] * right_low[index];
        const double second = left_low[index] * right_high[index];
        const double third = left_high[index] * right_low[index];
        const double fourth = left_high[index] * right_high[index];
        const double lower = std::min(std::min(first, second), std::min(third, fourth));
        const double upper = std::max(std::max(first, second), std::max(third, fourth));
        bilinear += law.weights[index] * (law.weights[index] >= 0.0 ? upper : lower);
    }
    const double pair = softplus(bilinear) + 1.0e-6;
    const double intensity =
        tree.sociability_max[left] * tree.sociability_max[right] * pair +
        tree.sociability_max[left] * tree.star_max[right] +
        tree.sociability_max[right] * tree.star_max[left] + law.dust;
    return log_expm1_intensity(intensity) +
           cached_marked_upper(law, tree, left) +
           cached_marked_upper(law, tree, right) +
           cached_parent_upper(law, tree, left, right) +
           law.component_upper + 1.0e-12;
}

std::int64_t cell_count(const TreeView& tree, int left, int right, bool same) {
    const std::int64_t first = tree.size[left];
    return same ? first * (first - 1) / 2 : first * tree.size[right];
}

struct ChildSpec {
    int left;
    int right;
    bool same;
    int raw_index;
    std::int64_t count;
};

void append_child(
    std::vector<ChildSpec>& output, const TreeView& tree, int left, int right,
    bool same, int raw_index
) {
    const std::int64_t count = cell_count(tree, left, right, same);
    if (count > 0) {
        output.push_back(ChildSpec{left, right, same, raw_index, count});
    }
}

void split_specs(const TreeView& tree, const Cell& parent, std::vector<ChildSpec>& out) {
    out.clear();
    const bool same = parent.left == parent.right;
    if (same) {
        const int first = tree.child0[parent.left];
        const int second = tree.child1[parent.left];
        append_child(out, tree, first, first, true, 0);
        append_child(out, tree, first, second, false, 1);
        append_child(out, tree, second, second, true, 2);
    } else if (!singleton(tree, parent.left) &&
               (singleton(tree, parent.right) ||
                tree.size[parent.left] >= tree.size[parent.right])) {
        append_child(out, tree, tree.child0[parent.left], parent.right, false, 0);
        append_child(out, tree, tree.child1[parent.left], parent.right, false, 1);
    } else {
        append_child(out, tree, parent.left, tree.child0[parent.right], false, 0);
        append_child(out, tree, parent.left, tree.child1[parent.right], false, 1);
    }
}

inline std::uint64_t edge_code(int left, int right) {
    const std::uint32_t first = static_cast<std::uint32_t>(std::min(left, right));
    const std::uint32_t second = static_cast<std::uint32_t>(std::max(left, right));
    return (static_cast<std::uint64_t>(first) << 32) | second;
}

struct SelectedLeaf {
    int left;
    int right;
    double utility;
};

inline bool leaf_better(const SelectedLeaf& first, const SelectedLeaf& second) {
    if (first.utility != second.utility) {
        return first.utility > second.utility;
    }
    return edge_code(first.left, first.right) < edge_code(second.left, second.right);
}

struct LeafBetter {
    bool operator()(const SelectedLeaf& first, const SelectedLeaf& second) const {
        return leaf_better(first, second);
    }
};

void topk_child_cells(
    const LawView& law,
    const TreeView& tree,
    const Cell& parent,
    double temperature,
    std::uint64_t process_seed,
    std::vector<ChildSpec>& specs,
    std::vector<Cell>& children
) {
    split_specs(tree, parent, specs);
    const std::int64_t parent_count = cell_count(
        tree, parent.left, parent.right, parent.left == parent.right
    );
    std::int64_t count_sum = 0;
    for (const ChildSpec& child : specs) {
        count_sum += child.count;
    }
    if (count_sum != parent_count) {
        throw std::logic_error("RouteS top-k cell partition lost dyad mass");
    }
    const std::uint32_t split_hash = extend_path(parent.path_hash, 0x7FFFFFFFU);
    const double winner_threshold = uniform53(process_seed, split_hash, 0) *
        static_cast<double>(parent_count);
    std::int64_t cumulative = 0;
    int winner = static_cast<int>(specs.size()) - 1;
    for (int index = 0; index < static_cast<int>(specs.size()); ++index) {
        cumulative += specs[index].count;
        if (winner_threshold < static_cast<double>(cumulative)) {
            winner = index;
            break;
        }
    }
    children.clear();
    std::uint64_t draw_index = 1;
    for (int index = 0; index < static_cast<int>(specs.size()); ++index) {
        const ChildSpec& spec = specs[index];
        const double maximum = index == winner
            ? parent.maximum
            : truncated_gumbel(
                  uniform53(process_seed, split_hash, draw_index++),
                  std::log(static_cast<double>(spec.count)), parent.maximum
              );
        const double upper = cell_upper(law, tree, spec.left, spec.right);
        children.push_back(Cell{
            spec.left,
            spec.right,
            extend_path(parent.path_hash, static_cast<std::uint32_t>(spec.raw_index)),
            maximum,
            upper + temperature * maximum,
            0,
        });
    }
    std::sort(children.begin(), children.end(), cell_higher);
}

struct SerialPathStep {
    Cell cell;
    int raw_child_index;
};

struct CompactSerialPathCertificate {
    // When ready is false, priority stores the original maximum, not a bound.
    std::vector<double> priority;
    std::vector<int> raw_child_index;
    std::vector<int> cell_left, cell_right;
    std::vector<unsigned char> priority_ready;
};

class HistoricalLeafOrder {
public:
    HistoricalLeafOrder(
        const LawView& law,
        const TreeView& tree,
        double temperature,
        std::uint64_t process_seed,
        const Cell& root
    )
        : law_(law),
          tree_(tree),
          temperature_(temperature),
          process_seed_(process_seed),
          root_(root),
          entry_(static_cast<std::size_t>(tree.count), -1),
          exit_(static_cast<std::size_t>(tree.count), -1),
          leaf_tree_node_(static_cast<std::size_t>(law.nodes), -1) {
        certificate_specs_.reserve(3);
        int clock = 0;
        std::function<void(int)> visit = [&](int node) {
            if (node < 0 || node >= tree_.count) {
                throw std::logic_error("RouteS serial certificate tree index out of range");
            }
            entry_[static_cast<std::size_t>(node)] = clock++;
            if (singleton(tree_, node)) {
                const int vertex = tree_.singleton[node];
                if (vertex < 0 || vertex >= law_.nodes) {
                    throw std::logic_error("RouteS serial certificate singleton out of range");
                }
                leaf_tree_node_[static_cast<std::size_t>(vertex)] = node;
            } else {
                visit(tree_.child0[node]);
                visit(tree_.child1[node]);
            }
            exit_[static_cast<std::size_t>(node)] = clock;
        };
        visit(0);
        for (int vertex = 0; vertex < law_.nodes; ++vertex) {
            if (leaf_tree_node_[static_cast<std::size_t>(vertex)] < 0) {
                throw std::logic_error("RouteS serial certificate missed a singleton");
            }
        }
    }

    bool better(const SelectedLeaf& first, const SelectedLeaf& second) const {
        if (first.utility != second.utility) {
            return first.utility > second.utility;
        }
        if (first.left == second.left && first.right == second.right) {
            return false;
        }
        ++last_binary64_serial_tie_comparison_count;
        const std::vector<SerialPathStep> first_path = reconstruct(first.left, first.right);
        const std::vector<SerialPathStep> second_path = reconstruct(second.left, second.right);
        const int comparison = compare_cell_serial(
            first_path,
            static_cast<int>(first_path.size()) - 1,
            second_path,
            static_cast<int>(second_path.size()) - 1
        );
        if (comparison == 0) {
            throw std::logic_error("RouteS serial certificate did not separate dyads");
        }
        return comparison < 0;
    }


#include "certificate_prefix_members.inc"
#include "lazy_priority_members.inc"
#include "path_demand_members.inc"

    void fill_compact_certificate_cached(
        int left,
        int right,
        CompactSerialPathCertificate& output
    ) const {
        output.priority.clear();
        output.raw_child_index.clear();
        output.cell_left.clear(); output.cell_right.clear(); output.priority_ready.clear();
        Cell parent = root_;
        int cache_index=0;bool reported_fallback=false;
        output.priority.push_back(parent.priority);
        output.raw_child_index.push_back(-1);
        output.cell_left.push_back(parent.left); output.cell_right.push_back(parent.right); output.priority_ready.push_back(1);
        while (!(singleton(tree_, parent.left) && singleton(tree_, parent.right))) {
            int cached_raw=-1;
            if(static_cast<int>(output.priority.size())-1<CERT_CACHE_DEPTH &&
               cached_certificate_child(cache_index,left,right,parent,cached_raw)){
                output.priority.push_back(parent.priority);output.raw_child_index.push_back(cached_raw);
                output.cell_left.push_back(parent.left);output.cell_right.push_back(parent.right);output.priority_ready.push_back(1);
                continue;
            }
            cache_index=-1;
            if(!reported_fallback){++certificate_prefix_fallbacks_;reported_fallback=true;}
            split_specs(tree_, parent, certificate_specs_);
            const std::int64_t parent_count = cell_count(
                tree_, parent.left, parent.right, parent.left == parent.right
            );
            const std::uint32_t split_hash = extend_path(
                parent.path_hash, 0x7FFFFFFFU
            );
            const double winner_threshold = uniform53(
                process_seed_, split_hash, 0
            ) * static_cast<double>(parent_count);
            std::int64_t cumulative = 0;
            int winner = static_cast<int>(certificate_specs_.size()) - 1;
            for (int index = 0;
                 index < static_cast<int>(certificate_specs_.size());
                 ++index) {
                cumulative += certificate_specs_[index].count;
                if (winner_threshold < static_cast<double>(cumulative)) {
                    winner = index;
                    break;
                }
            }
            int chosen = -1;
            std::uint64_t draw_index = 1;
            Cell chosen_cell{};
            for (int index = 0;
                 index < static_cast<int>(certificate_specs_.size());
                 ++index) {
                const ChildSpec& spec = certificate_specs_[index];
                const bool demanded = child_contains_dyad(spec, left, right);
                const std::uint64_t original_draw = draw_index;
                if (index != winner) ++draw_index;
                ++replay_children_scanned_;
                if (!demanded) {
                    if (index != winner) ++replay_truncated_omitted_;
                    continue;
                }
                if (index != winner) ++replay_truncated_computed_;
                const double maximum = index == winner
                    ? parent.maximum
                    : truncated_gumbel(
                          uniform53(process_seed_, split_hash, original_draw),
                          std::log(static_cast<double>(spec.count)), parent.maximum
                      );
                if (demanded) {
                    if (chosen >= 0) {
                        throw std::logic_error(
                            "RouteS compact serial certificate found overlapping dyad children"
                        );
                    }
                    chosen = index;
                    chosen_cell = Cell{
                        spec.left,
                        spec.right,
                        extend_path(
                            parent.path_hash,
                            static_cast<std::uint32_t>(spec.raw_index)
                        ),
                        maximum,
                        0.0, // Unread private path-walk priority; never a frontier record.
                        0,
                    };
                }
            }
            if (chosen < 0) {
                throw std::logic_error(
                    "RouteS compact serial certificate lost dyad path"
                );
            }
            parent = chosen_cell;
            output.priority.push_back(parent.maximum);
            output.cell_left.push_back(parent.left);output.cell_right.push_back(parent.right);output.priority_ready.push_back(0);
            ++lazy_suffix_nodes_;
            output.raw_child_index.push_back(
                certificate_specs_[chosen].raw_index
            );
        }
        lazy_maximum_depth_=std::max(lazy_maximum_depth_,output.priority.size());
        if (std::min(tree_.singleton[parent.left], tree_.singleton[parent.right]) !=
                std::min(left, right) ||
            std::max(tree_.singleton[parent.left], tree_.singleton[parent.right]) !=
                std::max(left, right)) {
            throw std::logic_error(
                "RouteS compact serial certificate ended at the wrong dyad"
            );
        }
    }

    void fill_compact_certificate_original(
        int left,
        int right,
        CompactSerialPathCertificate& output
    ) const {
        output.priority.clear();
        output.raw_child_index.clear();
        Cell parent = root_;
        output.priority.push_back(parent.priority);
        output.raw_child_index.push_back(-1);
        while (!(singleton(tree_, parent.left) && singleton(tree_, parent.right))) {
            split_specs(tree_, parent, certificate_specs_);
            const std::int64_t parent_count = cell_count(
                tree_, parent.left, parent.right, parent.left == parent.right
            );
            const std::uint32_t split_hash = extend_path(
                parent.path_hash, 0x7FFFFFFFU
            );
            const double winner_threshold = uniform53(
                process_seed_, split_hash, 0
            ) * static_cast<double>(parent_count);
            std::int64_t cumulative = 0;
            int winner = static_cast<int>(certificate_specs_.size()) - 1;
            for (int index = 0;
                 index < static_cast<int>(certificate_specs_.size());
                 ++index) {
                cumulative += certificate_specs_[index].count;
                if (winner_threshold < static_cast<double>(cumulative)) {
                    winner = index;
                    break;
                }
            }
            int chosen = -1;
            std::uint64_t draw_index = 1;
            Cell chosen_cell{};
            for (int index = 0;
                 index < static_cast<int>(certificate_specs_.size());
                 ++index) {
                const ChildSpec& spec = certificate_specs_[index];
                const double maximum = index == winner
                    ? parent.maximum
                    : truncated_gumbel(
                          uniform53(process_seed_, split_hash, draw_index++),
                          std::log(static_cast<double>(spec.count)), parent.maximum
                      );
                if (child_contains_dyad(spec, left, right)) {
                    if (chosen >= 0) {
                        throw std::logic_error(
                            "RouteS compact serial certificate found overlapping dyad children"
                        );
                    }
                    chosen = index;
                    const double upper = cell_upper(
                        law_, tree_, spec.left, spec.right
                    );
                    chosen_cell = Cell{
                        spec.left,
                        spec.right,
                        extend_path(
                            parent.path_hash,
                            static_cast<std::uint32_t>(spec.raw_index)
                        ),
                        maximum,
                        upper + temperature_ * maximum,
                        0,
                    };
                }
            }
            if (chosen < 0) {
                throw std::logic_error(
                    "RouteS compact serial certificate lost dyad path"
                );
            }
            parent = chosen_cell;
            output.priority.push_back(parent.priority);
            output.raw_child_index.push_back(
                certificate_specs_[chosen].raw_index
            );
        }
        if (std::min(tree_.singleton[parent.left], tree_.singleton[parent.right]) !=
                std::min(left, right) ||
            std::max(tree_.singleton[parent.left], tree_.singleton[parent.right]) !=
                std::max(left, right)) {
            throw std::logic_error(
                "RouteS compact serial certificate ended at the wrong dyad"
            );
        }
    }

    bool better_compact(
        const SelectedLeaf& first,
        CompactSerialPathCertificate& first_certificate,
        bool& first_ready,
        const SelectedLeaf& second,
        CompactSerialPathCertificate& second_certificate,
        bool& second_ready
    ) const {
        if (first.utility != second.utility) {
            return first.utility > second.utility;
        }
        if (first.left == second.left && first.right == second.right) {
            return false;
        }
        ++last_binary64_serial_tie_comparison_count;
        if (!first_ready) {
            fill_compact_certificate(first.left, first.right, first_certificate);
            first_ready = true;
            ++last_tie_cache_certificate_build_count;
        } else {
            ++last_tie_cache_certificate_reuse_count;
        }
        if (!second_ready) {
            fill_compact_certificate(second.left, second.right, second_certificate);
            second_ready = true;
            ++last_tie_cache_certificate_build_count;
        } else {
            ++last_tie_cache_certificate_reuse_count;
        }
        int first_index = static_cast<int>(first_certificate.priority.size()) - 1;
        int second_index = static_cast<int>(second_certificate.priority.size()) - 1;
        const auto same_prefix = [](
            const CompactSerialPathCertificate& first_value,
            int first_value_index,
            const CompactSerialPathCertificate& second_value,
            int second_value_index
        ) {
            if (first_value_index != second_value_index) {
                return false;
            }
            for (int index = 1; index <= first_value_index; ++index) {
                if (first_value.raw_child_index[static_cast<std::size_t>(index)] !=
                    second_value.raw_child_index[static_cast<std::size_t>(index)]) {
                    return false;
                }
            }
            return true;
        };
        while (true) {
            if (same_prefix(
                    first_certificate, first_index,
                    second_certificate, second_index)) {
                throw std::logic_error(
                    "RouteS compact serial certificate did not separate dyads"
                );
            }
            if (first_index == 0) {
                return true;
            }
            if (second_index == 0) {
                return false;
            }
            const int first_parent = first_index - 1;
            const int second_parent = second_index - 1;
            if (same_prefix(
                    first_certificate, first_parent,
                    second_certificate, second_parent)) {
                return first_certificate.raw_child_index[
                           static_cast<std::size_t>(first_index)
                       ] < second_certificate.raw_child_index[
                           static_cast<std::size_t>(second_index)
                       ];
            }
            const double first_priority = materialize_compact_priority(first_certificate, first_parent);
            const double second_priority = materialize_compact_priority(second_certificate, second_parent);
            if (first_priority != second_priority) {
                return first_priority > second_priority;
            }
            first_index = first_parent;
            second_index = second_parent;
        }
    }

private:
    bool contains(int tree_node, int vertex) const {
        const int leaf = leaf_tree_node_[static_cast<std::size_t>(vertex)];
        return entry_[static_cast<std::size_t>(tree_node)] <=
                   entry_[static_cast<std::size_t>(leaf)] &&
               entry_[static_cast<std::size_t>(leaf)] <
                   exit_[static_cast<std::size_t>(tree_node)];
    }

    bool child_contains_dyad(const ChildSpec& spec, int left, int right) const {
        if (spec.same) {
            return contains(spec.left, left) && contains(spec.left, right);
        }
        return (contains(spec.left, left) && contains(spec.right, right)) ||
               (contains(spec.left, right) && contains(spec.right, left));
    }

    std::vector<SerialPathStep> reconstruct(int left, int right) const {
        std::vector<SerialPathStep> output;
        output.reserve(64);
        Cell parent = root_;
        output.push_back(SerialPathStep{parent, -1});
        std::vector<ChildSpec> specs;
        specs.reserve(3);
        while (!(singleton(tree_, parent.left) && singleton(tree_, parent.right))) {
            split_specs(tree_, parent, specs);
            const std::int64_t parent_count = cell_count(
                tree_, parent.left, parent.right, parent.left == parent.right
            );
            const std::uint32_t split_hash = extend_path(parent.path_hash, 0x7FFFFFFFU);
            const double winner_threshold = uniform53(process_seed_, split_hash, 0) *
                static_cast<double>(parent_count);
            std::int64_t cumulative = 0;
            int winner = static_cast<int>(specs.size()) - 1;
            for (int index = 0; index < static_cast<int>(specs.size()); ++index) {
                cumulative += specs[index].count;
                if (winner_threshold < static_cast<double>(cumulative)) {
                    winner = index;
                    break;
                }
            }
            int chosen = -1;
            std::uint64_t draw_index = 1;
            Cell chosen_cell{};
            for (int index = 0; index < static_cast<int>(specs.size()); ++index) {
                const ChildSpec& spec = specs[index];
                const double maximum = index == winner
                    ? parent.maximum
                    : truncated_gumbel(
                          uniform53(process_seed_, split_hash, draw_index++),
                          std::log(static_cast<double>(spec.count)), parent.maximum
                      );
                if (child_contains_dyad(spec, left, right)) {
                    if (chosen >= 0) {
                        throw std::logic_error(
                            "RouteS serial certificate found overlapping dyad children"
                        );
                    }
                    chosen = index;
                    const double upper = cell_upper(law_, tree_, spec.left, spec.right);
                    chosen_cell = Cell{
                        spec.left,
                        spec.right,
                        extend_path(parent.path_hash, static_cast<std::uint32_t>(spec.raw_index)),
                        maximum,
                        upper + temperature_ * maximum,
                        0,
                    };
                }
            }
            if (chosen < 0) {
                throw std::logic_error("RouteS serial certificate lost dyad path");
            }
            parent = chosen_cell;
            output.push_back(SerialPathStep{parent, specs[chosen].raw_index});
        }
        if (std::min(tree_.singleton[parent.left], tree_.singleton[parent.right]) !=
                std::min(left, right) ||
            std::max(tree_.singleton[parent.left], tree_.singleton[parent.right]) !=
                std::max(left, right)) {
            throw std::logic_error("RouteS serial certificate ended at the wrong dyad");
        }
        return output;
    }

    static bool same_path_prefix(
        const std::vector<SerialPathStep>& first,
        int first_index,
        const std::vector<SerialPathStep>& second,
        int second_index
    ) {
        if (first_index != second_index) {
            return false;
        }
        for (int index = 1; index <= first_index; ++index) {
            if (first[static_cast<std::size_t>(index)].raw_child_index !=
                second[static_cast<std::size_t>(index)].raw_child_index) {
                return false;
            }
        }
        return true;
    }

    static int compare_cell_serial(
        const std::vector<SerialPathStep>& first,
        int first_index,
        const std::vector<SerialPathStep>& second,
        int second_index
    ) {
        if (same_path_prefix(first, first_index, second, second_index)) {
            return 0;
        }
        if (first_index == 0) {
            return -1;
        }
        if (second_index == 0) {
            return 1;
        }
        const int first_parent = first_index - 1;
        const int second_parent = second_index - 1;
        if (same_path_prefix(first, first_parent, second, second_parent)) {
            const int first_raw =
                first[static_cast<std::size_t>(first_index)].raw_child_index;
            const int second_raw =
                second[static_cast<std::size_t>(second_index)].raw_child_index;
            return first_raw < second_raw ? -1 : 1;
        }
        const double first_priority =
            first[static_cast<std::size_t>(first_parent)].cell.priority;
        const double second_priority =
            second[static_cast<std::size_t>(second_parent)].cell.priority;
        if (first_priority != second_priority) {
            return first_priority > second_priority ? -1 : 1;
        }
        return compare_cell_serial(
            first, first_parent, second, second_parent
        );
    }

    const LawView& law_;
    const TreeView& tree_;
    double temperature_;
    std::uint64_t process_seed_;
    Cell root_;
    std::vector<int> entry_;
    std::vector<int> exit_;
    std::vector<int> leaf_tree_node_;
    mutable std::vector<ChildSpec> certificate_specs_;
};

}  // namespace

ROUTE_S_EXPORT long long route_s_last_source_candidate_prefix_limit() {
    return static_cast<long long>(last_candidate_prefix_limit);
}

ROUTE_S_EXPORT long long route_s_last_source_maximum_materialized_segment_size() {
    return static_cast<long long>(last_maximum_materialized_segment_size);
}

ROUTE_S_EXPORT long long route_s_last_source_maximum_dfs_stack_size() {
    return static_cast<long long>(last_maximum_dfs_stack_size);
}

ROUTE_S_EXPORT long long route_s_last_source_topk_round_count() {
    return static_cast<long long>(last_topk_round_count);
}

ROUTE_S_EXPORT long long route_s_last_source_used_frontier_free_topk() {
    return static_cast<long long>(last_used_frontier_free_topk);
}

ROUTE_S_EXPORT long long route_s_last_source_used_monolithic_fallback() {
    return static_cast<long long>(last_used_monolithic_fallback);
}

ROUTE_S_EXPORT long long route_s_last_source_binary64_serial_tie_comparison_count() {
    return static_cast<long long>(last_binary64_serial_tie_comparison_count);
}

ROUTE_S_EXPORT int route_s_edge_eta_batch(
    int n_nodes,
    int rank,
    int parent_count,
    int component_count,
    const double* sociability,
    const double* star,
    const double* interaction,
    const double* weights,
    double dust,
    const int* parent_fiber,
    const int* component_fiber,
    const double* marked_shift,
    const double* component_shift,
    const double* parent_flux,
    std::int64_t pair_count,
    const std::int64_t* left_nodes,
    const std::int64_t* right_nodes,
    double* output_eta
) {
    try {
        if (n_nodes < 2 || rank < 1 || parent_count < 1 ||
            component_count < 4 || pair_count < 0) {
            return 1;
        }
        double component_upper = component_shift[0];
        for (int index = 1; index < component_count; ++index) {
            component_upper = std::max(component_upper, component_shift[index]);
        }
        const LawView law{
            n_nodes, rank, parent_count, component_count, sociability, star,
            interaction, weights, dust, parent_fiber, component_fiber,
            marked_shift, component_shift, parent_flux, component_upper
        };
        for (std::int64_t index = 0; index < pair_count; ++index) {
            const std::int64_t left = left_nodes[index];
            const std::int64_t right = right_nodes[index];
            if (left < 0 || right < 0 || left >= n_nodes || right >= n_nodes ||
                left == right) {
                return 2;
            }
            output_eta[index] = edge_eta(
                law, static_cast<int>(left), static_cast<int>(right)
            );
        }
        return 0;
    } catch (...) {
        return 99;
    }
}

ROUTE_S_EXPORT int route_s_flag_face_delta_batch(
    int n_nodes,
    const std::int64_t* offsets,
    const std::int64_t* neighbors,
    std::int64_t unit_count,
    const std::int64_t* donors,
    const std::int64_t* receivers,
    const std::int64_t* shared_neighbors,
    std::int64_t* output_delta
) {
    try {
        if (n_nodes < 1 || unit_count < 0) {
            return 1;
        }
        auto intersection_size = [&](int first, int second) -> std::int64_t {
            std::int64_t left = offsets[first];
            std::int64_t left_end = offsets[first + 1];
            std::int64_t right = offsets[second];
            std::int64_t right_end = offsets[second + 1];
            std::int64_t count = 0;
            while (left < left_end && right < right_end) {
                const std::int64_t a = neighbors[left];
                const std::int64_t b = neighbors[right];
                if (a < b) {
                    ++left;
                } else if (b < a) {
                    ++right;
                } else {
                    ++count;
                    ++left;
                    ++right;
                }
            }
            return count;
        };
        for (std::int64_t index = 0; index < unit_count; ++index) {
            const int donor = static_cast<int>(donors[index]);
            const int receiver = static_cast<int>(receivers[index]);
            const int neighbor = static_cast<int>(shared_neighbors[index]);
            if (donor < 0 || receiver < 0 || neighbor < 0 || donor >= n_nodes ||
                receiver >= n_nodes || neighbor >= n_nodes) {
                return 2;
            }
            const std::int64_t removed = intersection_size(donor, neighbor);
            const std::int64_t added = intersection_size(receiver, neighbor);
            const std::int64_t begin = offsets[receiver];
            const std::int64_t end = offsets[receiver + 1];
            const bool disappearing = std::binary_search(
                neighbors + begin, neighbors + end, static_cast<std::int64_t>(donor)
            );
            output_delta[index] = added - static_cast<std::int64_t>(disappearing) - removed;
        }
        return 0;
    } catch (...) {
        return 99;
    }
}
