
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include "diagnostic_timing.inc"
static long long trace_ties();
static long long trace_certs();
static long long trace_reuses();
static auto trace_started=std::chrono::steady_clock::now();
static double trace_last=-10.0, trace_first=-1.0;
static std::uint64_t trace_hash=1469598103934665603ULL;
static long long trace_bins[10]={};
static long long trace_streak=0,trace_max_streak=0;
static double trace_seconds(){return std::chrono::duration<double>(std::chrono::steady_clock::now()-trace_started).count();}

static long long observe_eta_calls=0,observe_eta_saturated=0;
static long long observe_sample_count[5]={},observe_sample_ns[5]={};
struct ObserveSample { ObserveSample(int,bool){} };
static void observe_extra(){}
static void observe_backend(const char* event,long long visited,long long processed,long long admitted,long long rejected,long long size,long long round,bool force=false){
 static double last=-10;
 if(!force && (visited&262143)!=0 && (processed&262143)!=0)return;
 double t=trace_seconds();if(!force && t-last<5)return;last=t;
 const char* p=std::getenv("SOURCE_TRACE_LOG");if(!p)return;auto f=std::fopen(p,"a");if(!f)return;
 std::fprintf(f,"{\"event\":\"%s\",\"seconds\":%.9g,\"visited\":%lld,\"processed\":%lld,\"accepted\":%lld,\"rejected\":%lld,\"queue_or_leaf_size\":%lld,\"round\":%lld,\"ties\":%lld}\n",event,t,visited,processed,admitted,rejected,size,round,trace_ties());std::fclose(f);
}
static void trace_event(const char* event,const char* detail){
 DiagnosticIO io;
 const char* p=std::getenv("SOURCE_TRACE_LOG"); if(!p)return;
 auto f=std::fopen(p,"a");if(!f)return;
 std::fprintf(f,"{\"event\":\"%s\",\"seconds\":%.9g,\"detail\":\"",event,trace_seconds());
 for(const char* c=detail;*c;++c){if(*c=='"'||*c=='\\')std::fputc('\\',f); if(*c>=32)std::fputc(*c,f);}
 std::fprintf(f,"\"}\n");std::fclose(f);
}
static void trace_probability(double,bool,long long,long long,double){}
static void trace_snapshot(long long visited,long long processed,long long accepted,long long rejected,
 long long frontier,long long ready,double head,double utility,double probability_sum,bool force=false){
 if(!force && (visited&262143)!=0 && (processed==0 || (processed&262143)!=0))return;
 double t=trace_seconds();if(processed>0 && trace_first<0)trace_first=t;
 if(!force && t-trace_last<5.0)return;trace_last=t;
 DiagnosticIO io;
 const char*p=std::getenv("SOURCE_TRACE_LOG");
 if(p){if(auto f=std::fopen(p,"a")){
 std::fprintf(f,"{\"event\":\"A_progress\",\"seconds\":%.9g,\"visited\":%lld,\"processed\":%lld,\"accepted\":%lld,\"rejected\":%lld,\"frontier\":%lld,\"ready\":%lld,\"head\":%.17g,\"utility\":%.17g,\"probability_sum\":%.17g,\"ties\":%lld,\"certificate_builds\":%lld,\"certificate_reuses\":%lld,\"diagnostic_io_seconds_so_far\":%.9g}\n",
 t,visited,processed,accepted,rejected,frontier,ready,head,utility,probability_sum,trace_ties(),trace_certs(),trace_reuses(),diagnostic_io_seconds);std::fclose(f);}}
 const char*limit=std::getenv("SOURCE_TRACE_LIMIT_VISITS");
 if(limit && visited>=std::atoll(limit))throw std::runtime_error("BOUNDED_REAL_CHECKPOINT_VALIDATION_STOP");
}
// Law-exact OperatorBound source executor with a stricter joint Cell supremum.
//
// The frozen RouteS priority is retained for child order and historical serial
// tie certificates.  The stricter priority is used only for sound pruning.

#include <queue>
#include "exact_run_queue.hpp"
#include "immediate_successor_queue.hpp"
static long long research_admitted=0;

#include "source_law.hpp"

namespace {

thread_local std::int64_t last_operator_bound_joint_bound_call_count = 0;
thread_local std::int64_t last_operator_bound_tightened_cell_count = 0;
thread_local std::int64_t last_operator_bound_component_restricted_count = 0;
thread_local std::int64_t last_operator_bound_joint_only_prune_count = 0;
thread_local std::int64_t last_operator_bound_operator_only_prune_count = 0;
thread_local std::int64_t last_operator_bound_operator_parent_prune_count = 0;
thread_local double last_operator_bound_cumulative_upper_reduction = 0.0;
thread_local double last_operator_bound_maximum_upper_reduction = 0.0;

struct JointTreeSummaryOperatorBound {
    std::vector<double> sociability_min;
    std::vector<double> star_min;
    std::vector<double> mass_max;
    std::vector<double> weighted_norm_max;
    std::vector<double> interaction_norm_max;
    std::vector<std::uint64_t> component_mask;
    bool nonnegative_atoms;
};

JointTreeSummaryOperatorBound build_joint_tree_summary_operator_bound(
    const LawView& law, const TreeView& tree
) {
    JointTreeSummaryOperatorBound summary{};
    summary.weighted_norm_max.resize(static_cast<std::size_t>(tree.count));
#ifndef ROUTE_S_OPERATOR_ONLY
    summary.sociability_min.resize(static_cast<std::size_t>(tree.count));
    summary.star_min.resize(static_cast<std::size_t>(tree.count));
    summary.mass_max.resize(static_cast<std::size_t>(tree.count));
    summary.interaction_norm_max.resize(static_cast<std::size_t>(tree.count));
    summary.component_mask.resize(static_cast<std::size_t>(tree.count));
    summary.nonnegative_atoms = true;
#endif
    for (int node = tree.count - 1; node >= 0; --node) {
        if (singleton(tree, node)) {
            const int vertex = tree.singleton[node];
            const double sociability = law.sociability[vertex];
            const double star = law.star[vertex];
            double weighted_norm_squared = 0.0;
#ifndef ROUTE_S_OPERATOR_ONLY
            summary.sociability_min[static_cast<std::size_t>(node)] = sociability;
            summary.star_min[static_cast<std::size_t>(node)] = star;
            summary.mass_max[static_cast<std::size_t>(node)] = sociability + star;
            double interaction_norm_squared = 0.0;
#endif
            const double* coordinate = law.interaction +
                static_cast<std::int64_t>(vertex) * law.rank;
            for (int index = 0; index < law.rank; ++index) {
                const double square = coordinate[index] * coordinate[index];
                weighted_norm_squared += std::abs(law.weights[index]) * square;
#ifndef ROUTE_S_OPERATOR_ONLY
                interaction_norm_squared += square;
#endif
            }
            summary.weighted_norm_max[static_cast<std::size_t>(node)] =
                std::sqrt(weighted_norm_squared);
#ifndef ROUTE_S_OPERATOR_ONLY
            summary.interaction_norm_max[static_cast<std::size_t>(node)] =
                std::sqrt(interaction_norm_squared);
            summary.component_mask[static_cast<std::size_t>(node)] =
                std::uint64_t{1} << law.component_fiber[vertex];
            summary.nonnegative_atoms = summary.nonnegative_atoms &&
                sociability >= 0.0 && star >= 0.0;
#endif
            continue;
        }
        const int first = tree.child0[node];
        const int second = tree.child1[node];
#ifndef ROUTE_S_OPERATOR_ONLY
        summary.sociability_min[static_cast<std::size_t>(node)] = std::min(
            summary.sociability_min[static_cast<std::size_t>(first)],
            summary.sociability_min[static_cast<std::size_t>(second)]
        );
        summary.star_min[static_cast<std::size_t>(node)] = std::min(
            summary.star_min[static_cast<std::size_t>(first)],
            summary.star_min[static_cast<std::size_t>(second)]
        );
        summary.mass_max[static_cast<std::size_t>(node)] = std::max(
            summary.mass_max[static_cast<std::size_t>(first)],
            summary.mass_max[static_cast<std::size_t>(second)]
        );
#endif
        summary.weighted_norm_max[static_cast<std::size_t>(node)] = std::max(
            summary.weighted_norm_max[static_cast<std::size_t>(first)],
            summary.weighted_norm_max[static_cast<std::size_t>(second)]
        );
#ifndef ROUTE_S_OPERATOR_ONLY
        summary.interaction_norm_max[static_cast<std::size_t>(node)] = std::max(
            summary.interaction_norm_max[static_cast<std::size_t>(first)],
            summary.interaction_norm_max[static_cast<std::size_t>(second)]
        );
        summary.component_mask[static_cast<std::size_t>(node)] =
            summary.component_mask[static_cast<std::size_t>(first)] |
            summary.component_mask[static_cast<std::size_t>(second)];
#endif
    }
    return summary;
}

struct CellBoundsOperatorBound {
    double frozen;
#ifndef ROUTE_S_OPERATOR_ONLY
    double operator_only;
    double operator_parent;
#endif
    double joint;
};

CellBoundsOperatorBound cell_bounds_operator_bound_original(
    const LawView& law,
    const TreeView& tree,
    const JointTreeSummaryOperatorBound& summary,
    int left,
    int right
) {
    if (singleton(tree, left) && singleton(tree, right)) {
        const double exact = edge_eta(
            law, tree.singleton[left], tree.singleton[right]
        );
#ifdef ROUTE_S_OPERATOR_ONLY
        return CellBoundsOperatorBound{exact, exact};
#else
        return CellBoundsOperatorBound{exact, exact, exact, exact};
#endif
    }

    const double* left_low = tree.low + static_cast<std::int64_t>(left) * law.rank;
    const double* left_high = tree.high + static_cast<std::int64_t>(left) * law.rank;
    const double* right_low = tree.low + static_cast<std::int64_t>(right) * law.rank;
    const double* right_high = tree.high + static_cast<std::int64_t>(right) * law.rank;
    double bilinear_box_upper = 0.0;
#ifndef ROUTE_S_OPERATOR_ONLY
    double similarity_lower = 0.0;
    double similarity_upper = 0.0;
#endif
    for (int index = 0; index < law.rank; ++index) {
        const double first = left_low[index] * right_low[index];
        const double second = left_low[index] * right_high[index];
        const double third = left_high[index] * right_low[index];
        const double fourth = left_high[index] * right_high[index];
        const double lower = std::min(std::min(first, second), std::min(third, fourth));
        const double upper = std::max(std::max(first, second), std::max(third, fourth));
        bilinear_box_upper +=
            law.weights[index] * (law.weights[index] >= 0.0 ? upper : lower);
#ifndef ROUTE_S_OPERATOR_ONLY
        similarity_lower += lower;
        similarity_upper += upper;
#endif
    }
    const double frozen_pair = softplus(bilinear_box_upper) + 1.0e-6;
    const double frozen_intensity =
        tree.sociability_max[left] * tree.sociability_max[right] * frozen_pair +
        tree.sociability_max[left] * tree.star_max[right] +
        tree.sociability_max[right] * tree.star_max[left] + law.dust;
    const double frozen_parent =
        cached_marked_upper(law, tree, left) +
        cached_marked_upper(law, tree, right) +
        cached_parent_upper(law, tree, left, right);
    const double frozen = log_expm1_intensity(frozen_intensity) + frozen_parent +
        law.component_upper + 1.0e-12;
    const double bilinear_norm_upper =
        summary.weighted_norm_max[static_cast<std::size_t>(left)] *
        summary.weighted_norm_max[static_cast<std::size_t>(right)];
#ifdef ROUTE_S_OPERATOR_ONLY
    if (bilinear_norm_upper >= bilinear_box_upper) {
        return CellBoundsOperatorBound{frozen, frozen};
    }
#endif
    const double bilinear = std::min(bilinear_box_upper, bilinear_norm_upper);
#ifndef ROUTE_S_OPERATOR_ONLY
    const double similarity_norm =
        summary.interaction_norm_max[static_cast<std::size_t>(left)] *
        summary.interaction_norm_max[static_cast<std::size_t>(right)];
    similarity_lower = std::max(similarity_lower, -similarity_norm);
    similarity_upper = std::min(similarity_upper, similarity_norm);
#endif
    const double pair = softplus(bilinear) + 1.0e-6;
    const double s_left = tree.sociability_max[left];
    const double s_right = tree.sociability_max[right];
    const double t_left = tree.star_max[left];
    const double t_right = tree.star_max[right];
    const double separate_intensity =
        s_left * s_right * pair + s_left * t_right + s_right * t_left + law.dust;
    const double operator_only = log_expm1_intensity(separate_intensity) +
        frozen_parent + law.component_upper + 1.0e-12;
#ifdef ROUTE_S_OPERATOR_ONLY
    const double tightened = std::min(frozen, operator_only);
    const double reduction = frozen - tightened;
    if (reduction > 0.0) {
        ++last_operator_bound_tightened_cell_count;
        last_operator_bound_cumulative_upper_reduction += reduction;
        last_operator_bound_maximum_upper_reduction = std::max(
            last_operator_bound_maximum_upper_reduction, reduction
        );
    }
    return CellBoundsOperatorBound{frozen, tightened};
#else
    double intensity_upper = separate_intensity;
    if (summary.nonnegative_atoms) {
        const double mass_intensity =
            summary.mass_max[static_cast<std::size_t>(left)] *
                summary.mass_max[static_cast<std::size_t>(right)] +
            std::max(pair - 1.0, 0.0) * s_left * s_right + law.dust;
        intensity_upper = std::min(intensity_upper, mass_intensity);
    }

    const std::uint64_t left_parent_mask = tree.parent_mask[left];
    const std::uint64_t right_parent_mask = tree.parent_mask[right];
    double joint_parent = -std::numeric_limits<double>::infinity();
    for (int parent_left = 0; parent_left < law.parent_count; ++parent_left) {
        if (!((left_parent_mask >> parent_left) & 1ULL)) {
            continue;
        }
        for (int parent_right = 0; parent_right < law.parent_count; ++parent_right) {
            if (!((right_parent_mask >> parent_right) & 1ULL)) {
                continue;
            }
            joint_parent = std::max(
                joint_parent,
                law.marked_shift[parent_left] + law.marked_shift[parent_right] +
                    law.parent_flux[parent_left * law.parent_count + parent_right]
            );
        }
    }

    double joint_component = -std::numeric_limits<double>::infinity();
    const std::uint64_t left_component_mask =
        summary.component_mask[static_cast<std::size_t>(left)];
    const std::uint64_t right_component_mask =
        summary.component_mask[static_cast<std::size_t>(right)];
    const bool left_single_component = left_component_mask != 0 &&
        (left_component_mask & (left_component_mask - 1)) == 0;
    const bool right_single_component = right_component_mask != 0 &&
        (right_component_mask & (right_component_mask - 1)) == 0;
    const bool guaranteed_component_one = left_single_component &&
        right_single_component && left_component_mask == right_component_mask &&
        similarity_lower >= 0.0;
    if (guaranteed_component_one) {
        joint_component = law.component_shift[1];
    }
    if ((left_component_mask & right_component_mask) != 0 && similarity_upper >= 0.0) {
        joint_component = std::max(joint_component, law.component_shift[1]);
    }
    if (!guaranteed_component_one) {
        if (t_left + t_right >= 2.0) {
            joint_component = std::max(joint_component, law.component_shift[0]);
        }
        if (summary.star_min[static_cast<std::size_t>(left)] +
                summary.star_min[static_cast<std::size_t>(right)] < 2.0) {
            if (summary.sociability_min[static_cast<std::size_t>(left)] +
                    summary.sociability_min[static_cast<std::size_t>(right)] <= 2.0) {
                joint_component = std::max(joint_component, law.component_shift[3]);
            }
            if (s_left + s_right > 2.0) {
                joint_component = std::max(joint_component, law.component_shift[2]);
            }
        }
    }
    if (!std::isfinite(joint_component)) {
        return CellBoundsOperatorBound{frozen, operator_only, operator_only, frozen};
    }
    if (joint_component < law.component_upper) {
        ++last_operator_bound_component_restricted_count;
    }
    const double proposed =
        log_expm1_intensity(intensity_upper) + joint_parent + joint_component + 1.0e-12;
    const double operator_parent = log_expm1_intensity(separate_intensity) +
        joint_parent + law.component_upper + 1.0e-12;
    if (!std::isfinite(proposed)) {
        return CellBoundsOperatorBound{frozen, operator_only, operator_parent, frozen};
    }
    const double tightened = std::min(frozen, proposed);
    const double reduction = frozen - tightened;
    if (reduction > 0.0) {
        ++last_operator_bound_tightened_cell_count;
        last_operator_bound_cumulative_upper_reduction += reduction;
        last_operator_bound_maximum_upper_reduction = std::max(
            last_operator_bound_maximum_upper_reduction, reduction
        );
    }
    return CellBoundsOperatorBound{
        frozen,
        std::min(frozen, operator_only),
        std::min(frozen, operator_parent),
        tightened,
    };
#endif
}

#include "cell_saturation.inc"

struct JointCellOperatorBound {
    Cell historical;
#ifndef ROUTE_S_OPERATOR_ONLY
    double operator_only_priority;
    double operator_parent_priority;
#endif
    double prune_priority;
};

struct StreamingFrontierLower {
    bool operator()(
        const JointCellOperatorBound& first,
        const JointCellOperatorBound& second
    ) const {
        if (first.prune_priority != second.prune_priority) {
            return first.prune_priority < second.prune_priority;
        }
        if (first.historical.priority != second.historical.priority) {
            return first.historical.priority < second.historical.priority;
        }
        if (first.historical.path_hash != second.historical.path_hash) {
            return first.historical.path_hash > second.historical.path_hash;
        }
        if (first.historical.left != second.historical.left) {
            return first.historical.left > second.historical.left;
        }
        return first.historical.right > second.historical.right;
    }
};

struct CachedStreamingReadyLeaf {
    SelectedLeaf leaf;
    CompactSerialPathCertificate certificate;
    bool certificate_ready = false;
};

struct StreamingReadyLower {
    const HistoricalLeafOrder* order;

    bool operator()(
        CachedStreamingReadyLeaf* first,
        CachedStreamingReadyLeaf* second
    ) const {
        return order->better_compact(
            second->leaf,
            second->certificate,
            second->certificate_ready,
            first->leaf,
            first->certificate,
            first->certificate_ready
        );
    }
};

void joint_child_cells_operator_bound(
    const LawView& law,
    const TreeView& tree,
    const JointTreeSummaryOperatorBound& summary,
    const JointCellOperatorBound& parent,
    double temperature,
    std::uint64_t process_seed,
    std::vector<ChildSpec>& specs,
    std::vector<JointCellOperatorBound>& children
) {
    split_specs(tree, parent.historical, specs);
    const std::int64_t parent_count = cell_count(
        tree,
        parent.historical.left,
        parent.historical.right,
        parent.historical.left == parent.historical.right
    );
    std::int64_t count_sum = 0;
    for (const ChildSpec& child : specs) {
        count_sum += child.count;
    }
    if (count_sum != parent_count) {
        throw std::logic_error("OperatorBound joint Cell partition lost dyad mass");
    }
    const std::uint32_t split_hash = extend_path(
        parent.historical.path_hash, 0x7FFFFFFFU
    );
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
            ? parent.historical.maximum
            : truncated_gumbel(
                  uniform53(process_seed, split_hash, draw_index++),
                  std::log(static_cast<double>(spec.count)),
                  parent.historical.maximum
              );
        const CellBoundsOperatorBound bounds = cell_bounds_operator_bound(
            law, tree, summary, spec.left, spec.right
        );
        children.push_back(JointCellOperatorBound{
            Cell{
                spec.left,
                spec.right,
                extend_path(
                    parent.historical.path_hash,
                    static_cast<std::uint32_t>(spec.raw_index)
                ),
                maximum,
                bounds.frozen + temperature * maximum,
                0,
            },
#ifndef ROUTE_S_OPERATOR_ONLY
            bounds.operator_only + temperature * maximum,
            bounds.operator_parent + temperature * maximum,
#endif
            bounds.joint + temperature * maximum,
        });
    }
    std::sort(
        children.begin(),
        children.end(),
        [](const JointCellOperatorBound& first, const JointCellOperatorBound& second) {
            return cell_higher(first.historical, second.historical);
        }
    );
}

bool generate_joint_streaming_frontier_operator_bound(
    const LawView& law,
    const TreeView& tree,
    std::int64_t target_edges,
    double degree_sigma,
    double temperature,
    std::uint32_t root_path_hash,
    std::uint64_t order_seed,
    std::uint64_t mark_seed,
    const double* target,
    std::int64_t* output_edges,
    double* output_utilities,
    std::int64_t* integer_diagnostics,
    double* real_diagnostics
) {
    const std::int64_t maximum_edges =
        static_cast<std::int64_t>(law.nodes) * (law.nodes - 1) / 2;
    if (target_edges == 0) {
        std::fill(integer_diagnostics, integer_diagnostics + 5, std::int64_t{0});
        std::fill(real_diagnostics, real_diagnostics + 9, 0.0);
        last_candidate_prefix_limit = 0;
        last_maximum_materialized_segment_size = 0;
        last_maximum_dfs_stack_size = 1;
        last_topk_round_count = 1;
        return true;
    }

    diagnostic_io_seconds=0.0;diagnostic_io_scopes=0;
    trace_started=std::chrono::steady_clock::now();trace_last=-10;trace_first=-1;
    trace_hash=1469598103934665603ULL;std::memset(trace_bins,0,sizeof(trace_bins));trace_streak=trace_max_streak=0;
    trace_event("A_start","exact streaming frontier; original source distribution and random streams");
    initialize_phase_query_cache(law,tree);
    initialize_bound_certificate(law,tree);
    const JointTreeSummaryOperatorBound summary =
        build_joint_tree_summary_operator_bound(law, tree);
    const std::uint64_t process_seed = order_seed ^ (GUMBEL_DOMAIN * GOLDEN64);
    const double inverse_variance = 1.0 / (degree_sigma * degree_sigma);
    const double root_maximum = gumbel(
        uniform53(process_seed, root_path_hash, 0),
        std::log(static_cast<double>(maximum_edges))
    );
    const CellBoundsOperatorBound root_bounds =
        cell_bounds_operator_bound(law, tree, summary, 0, 0);
    const Cell historical_root{
        0,
        0,
        root_path_hash,
        root_maximum,
        root_bounds.frozen + temperature * root_maximum,
        0,
    };
    const JointCellOperatorBound root{
        historical_root,
#ifndef ROUTE_S_OPERATOR_ONLY
        root_bounds.operator_only + temperature * root_maximum,
        root_bounds.operator_parent + temperature * root_maximum,
#endif
        root_bounds.joint + temperature * root_maximum,
    };
    HistoricalLeafOrder historical_order(
        law, tree, temperature, process_seed, historical_root
    );
    using BaseFrontier=ExactRunQueue<JointCellOperatorBound,StreamingFrontierLower>;
    ImmediateSuccessorQueue<JointCellOperatorBound,StreamingFrontierLower,BaseFrontier> frontier;
    frontier.push(root);
    std::vector<CachedStreamingReadyLeaf*> ready;
    ready.reserve(16);
    std::vector<std::unique_ptr<CachedStreamingReadyLeaf>> ready_storage;
    ready_storage.reserve(16);
    std::vector<CachedStreamingReadyLeaf*> ready_free;
    ready_free.reserve(16);
    const StreamingReadyLower ready_lower{&historical_order};
    const auto acquire_ready = [&]() {
        CachedStreamingReadyLeaf* output = nullptr;
        if (!ready_free.empty()) {
            output = ready_free.back();
            ready_free.pop_back();
        } else {
            ready_storage.push_back(
                std::make_unique<CachedStreamingReadyLeaf>()
            );
            output = ready_storage.back().get();
            last_tie_cache_maximum_ready_pool_size = std::max<std::int64_t>(
                last_tie_cache_maximum_ready_pool_size,
                static_cast<std::int64_t>(ready_storage.size())
            );
        }
        output->certificate_ready = false;
        return output;
    };
    std::vector<ChildSpec> specs;
    specs.reserve(3);
    std::vector<JointCellOperatorBound> children;
    children.reserve(3);

    std::vector<double> log_increment(law.nodes);
    double increment_sum = 0.0;
    for (int node = 0; node < law.nodes; ++node) {
        log_increment[node] = (target[node] - 0.5) * inverse_variance;
        increment_sum += log_increment[node];
    }
    const double initial_increment_mean =
        increment_sum / static_cast<double>(law.nodes);
    std::vector<int> degree(law.nodes, 0);
    std::int64_t admitted = 0;
    std::int64_t processed = 0;
    std::int64_t rejected = 0;
    std::int64_t forced = 0;
    std::int64_t visited = 0;
    std::int64_t maximum_frontier = 1;
    double probability_sum = 0.0;
    double probability_minimum = 1.0;
    double probability_maximum = 0.0;
    double threshold = -std::numeric_limits<double>::infinity();

    research_admitted=0;
    const char* cap_text=std::getenv("S_PREFIX_CANDIDATES");
    const long long prefix_cap=cap_text?std::atoll(cap_text):0;
    const char* rank_path=std::getenv("S_ACCEPT_RANKS");
    std::unique_ptr<std::FILE,decltype(&std::fclose)> rank_file(rank_path?std::fopen(rank_path,"wb"):nullptr,&std::fclose);
    if(rank_path&&!rank_file)throw std::runtime_error("cannot open research rank trace");
    while (admitted < target_edges) {
        while (
            ready.empty() ||
            (!frontier.empty() &&
             frontier.top().prune_priority >= ready.front()->leaf.utility)
        ) {
            if (frontier.empty()) {
                throw std::logic_error("streaming OperatorBound frontier exhausted");
            }
            const JointCellOperatorBound parent = frontier.top();
            {ObserveSample sample(0,(visited&4095)==0);frontier.pop();}
            ++visited;
            trace_snapshot(visited,processed,admitted,rejected,frontier.size(),ready.size(),parent.prune_priority,ready.empty()?0:ready.front()->leaf.utility,probability_sum);
            if (
                singleton(tree, parent.historical.left) &&
                singleton(tree, parent.historical.right)
            ) {
                const int left = tree.singleton[parent.historical.left];
                const int right = tree.singleton[parent.historical.right];
                const double utility = edge_eta(law, left, right) +
                    temperature * parent.historical.maximum;
                CachedStreamingReadyLeaf* ready_leaf = acquire_ready();
                ready_leaf->leaf = SelectedLeaf{left, right, utility};
                ready.push_back(ready_leaf);
                {ObserveSample sample(3,(observe_eta_calls&4095)==0);std::push_heap(ready.begin(), ready.end(), ready_lower);}
            } else {
                {ObserveSample sample(1,(visited&4095)==1);joint_child_cells_operator_bound(
                    law,
                    tree,
                    summary,
                    parent,
                    temperature,
                    process_seed,
                    specs,
                    children
                );}
                {ObserveSample sample(2,(visited&4095)==1);for (const JointCellOperatorBound& child : children) {
                    frontier.push(child);
                }}
                maximum_frontier = std::max<std::int64_t>(
                    maximum_frontier, static_cast<std::int64_t>(frontier.size())
                );
            }
        }

        {ObserveSample sample(3,(processed&4095)==0);std::pop_heap(ready.begin(), ready.end(), ready_lower);}
        CachedStreamingReadyLeaf* ready_leaf = ready.back();
        const SelectedLeaf leaf = ready_leaf->leaf;
        ready.pop_back();
        ready_free.push_back(ready_leaf);
        ++processed;
        const double mean_increment = increment_sum / static_cast<double>(law.nodes);
        const double centered = log_increment[leaf.left] + log_increment[leaf.right] -
            2.0 * mean_increment;
        const double probability = std::exp(std::min(0.0, centered));
        probability_sum += probability;
        probability_minimum = std::min(probability_minimum, probability);
        probability_maximum = std::max(probability_maximum, probability);
        const double mark = uniform53(
            mark_seed, MARK_DOMAIN, static_cast<std::uint64_t>(processed - 1)
        );
        const std::int64_t unseen_after = maximum_edges - processed;
        const bool must_accept = admitted + unseen_after < target_edges;
        trace_probability(probability,must_accept || mark<=probability,leaf.left,leaf.right,leaf.utility);
        if (must_accept || mark <= probability) {
            if (must_accept) {
                ++forced;
            }
            if(rank_file)write_research_rank(rank_file.get(),processed);
            const std::size_t edge_index = static_cast<std::size_t>(admitted++);
            output_edges[2 * edge_index] = std::min(leaf.left, leaf.right);
            output_edges[2 * edge_index + 1] = std::max(leaf.left, leaf.right);
            output_utilities[edge_index] = leaf.utility;
            ++degree[leaf.left];
            ++degree[leaf.right];
            log_increment[leaf.left] -= inverse_variance;
            log_increment[leaf.right] -= inverse_variance;
            increment_sum -= 2.0 * inverse_variance;
            threshold = leaf.utility;
        } else {
            ++rejected;
        }
        trace_snapshot(visited,processed,admitted,rejected,frontier.size(),ready.size(),frontier.empty()?0:frontier.top().prune_priority,leaf.utility,probability_sum,processed==1);
        if(prefix_cap && processed>=prefix_cap)break;
    }
    trace_snapshot(visited,processed,admitted,rejected,frontier.size(),ready.size(),frontier.empty()?0:frontier.top().prune_priority,threshold,probability_sum,true);
    historical_order.write_certificate_cache_stats();
    historical_order.write_lazy_priority_stats(ready_storage);
    historical_order.write_replay_demand_stats();
    write_phase_query_cache_stats();
    write_bound_certificate_stats();
    if(rank_file){DiagnosticIO io;rank_file.reset();}
    research_admitted=admitted;
    trace_event(admitted==target_edges?"A_complete":"A_prefix_complete","research exact candidate boundary");
    const char* state_path=std::getenv("S_STATE_DUMP");
    if(state_path){
        DiagnosticIO io;
        std::unique_ptr<std::FILE,decltype(&std::fclose)> f(std::fopen(state_path,"wb"),&std::fclose);
        if(!f || std::fwrite(log_increment.data(),sizeof(double),law.nodes,f.get())!=static_cast<std::size_t>(law.nodes) ||
           std::fwrite(degree.data(),sizeof(int),law.nodes,f.get())!=static_cast<std::size_t>(law.nodes) ||
           std::fwrite(&increment_sum,sizeof(double),1,f.get())!=1)
            throw std::runtime_error("research final state write failed");
    }

    double target_l1 = 0.0;
    for (int node = 0; node < law.nodes; ++node) {
        target_l1 += std::abs(target[node] - static_cast<double>(degree[node]));
    }
    const double next_upper = frontier.empty()
        ? -std::numeric_limits<double>::infinity()
        : frontier.top().prune_priority;
    const double next_ready = ready.empty()
        ? -std::numeric_limits<double>::infinity()
        : ready.front()->leaf.utility;
    integer_diagnostics[0] = visited;
    integer_diagnostics[1] = maximum_frontier;
    integer_diagnostics[2] = processed;
    integer_diagnostics[3] = rejected;
    integer_diagnostics[4] = forced;
    real_diagnostics[0] = probability_sum /
        static_cast<double>(std::max<std::int64_t>(processed, 1));
    real_diagnostics[1] = probability_minimum;
    real_diagnostics[2] = probability_maximum;
    real_diagnostics[3] = initial_increment_mean;
    real_diagnostics[4] = increment_sum / static_cast<double>(law.nodes);
    real_diagnostics[5] = target_l1;
    real_diagnostics[6] = static_cast<double>(processed) /
        static_cast<double>(maximum_edges);
    real_diagnostics[7] = threshold;
    real_diagnostics[8] = std::max(next_upper, next_ready);
    last_candidate_prefix_limit = processed;
    last_maximum_materialized_segment_size = maximum_frontier;
    last_maximum_dfs_stack_size = maximum_frontier;
    last_topk_round_count = 1;
    return true;
}

}  // namespace

ROUTE_S_EXPORT long long operator_bound_last_joint_bound_call_count() {
    return static_cast<long long>(last_operator_bound_joint_bound_call_count);
}

ROUTE_S_EXPORT long long operator_bound_last_tightened_cell_count() {
    return static_cast<long long>(last_operator_bound_tightened_cell_count);
}

ROUTE_S_EXPORT long long operator_bound_last_component_restricted_count() {
    return static_cast<long long>(last_operator_bound_component_restricted_count);
}

ROUTE_S_EXPORT long long operator_bound_last_joint_only_prune_count() {
    return static_cast<long long>(last_operator_bound_joint_only_prune_count);
}

ROUTE_S_EXPORT long long operator_bound_last_operator_only_prune_count() {
    return static_cast<long long>(last_operator_bound_operator_only_prune_count);
}

ROUTE_S_EXPORT long long operator_bound_last_operator_parent_prune_count() {
    return static_cast<long long>(last_operator_bound_operator_parent_prune_count);
}

ROUTE_S_EXPORT double operator_bound_last_cumulative_upper_reduction() {
    return last_operator_bound_cumulative_upper_reduction;
}

ROUTE_S_EXPORT double operator_bound_last_maximum_upper_reduction() {
    return last_operator_bound_maximum_upper_reduction;
}

ROUTE_S_EXPORT long long route_s_tie_cache_certificate_build_count() {
    return static_cast<long long>(last_tie_cache_certificate_build_count);
}

ROUTE_S_EXPORT long long route_s_tie_cache_certificate_reuse_count() {
    return static_cast<long long>(last_tie_cache_certificate_reuse_count);
}

ROUTE_S_EXPORT long long route_s_tie_cache_maximum_ready_pool_size() {
    return static_cast<long long>(last_tie_cache_maximum_ready_pool_size);
}

ROUTE_S_EXPORT int route_s_generate_edges(
    int n_nodes,
    int rank,
    int parent_count,
    int component_count,
    std::int64_t target_edges,
    double degree_sigma,
    double temperature,
    const double* target,
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
    int tree_count,
    const int* child0,
    const int* child1,
    const int* tree_size,
    const int* singleton_node,
    const double* coordinate_low,
    const double* coordinate_high,
    const double* tree_sociability_max,
    const double* tree_star_max,
    const std::uint64_t* tree_parent_mask,
    std::uint32_t root_path_hash,
    std::uint64_t order_seed,
    std::uint64_t mark_seed,
    std::int64_t* output_edges,
    double* output_utilities,
    std::int64_t* integer_diagnostics,
    double* real_diagnostics
) {
    try {
        last_candidate_prefix_limit = 0;
        last_maximum_materialized_segment_size = 0;
        last_maximum_dfs_stack_size = 0;
        last_topk_round_count = 0;
        last_used_frontier_free_topk = 0;
        last_used_monolithic_fallback = 0;
        last_binary64_serial_tie_comparison_count = 0;
        last_tie_cache_certificate_build_count = 0;
        last_tie_cache_certificate_reuse_count = 0;
        last_tie_cache_maximum_ready_pool_size = 0;
        last_operator_bound_joint_bound_call_count = 0;
        last_operator_bound_tightened_cell_count = 0;
        last_operator_bound_component_restricted_count = 0;
        last_operator_bound_joint_only_prune_count = 0;
        last_operator_bound_operator_only_prune_count = 0;
        last_operator_bound_operator_parent_prune_count = 0;
        last_operator_bound_cumulative_upper_reduction = 0.0;
        last_operator_bound_maximum_upper_reduction = 0.0;
        const std::int64_t maximum_edges =
            static_cast<std::int64_t>(n_nodes) * (n_nodes - 1) / 2;
        if (n_nodes < 2 || rank < 1 || parent_count < 1 || parent_count > 64 ||
            component_count < 4 || component_count > 64 || tree_count < 1 ||
            target_edges < 0 || target_edges > maximum_edges || degree_sigma <= 0.0 ||
            temperature <= 0.0) {
            return 1;
        }
        const TreeView tree{
            tree_count,
            rank,
            child0,
            child1,
            tree_size,
            singleton_node,
            coordinate_low,
            coordinate_high,
            tree_sociability_max,
            tree_star_max,
            tree_parent_mask,
        };
        double component_upper = component_shift[0];
        for (int index = 1; index < component_count; ++index) {
            component_upper = std::max(component_upper, component_shift[index]);
        }
        const LawView law{
            n_nodes,
            rank,
            parent_count,
            component_count,
            sociability,
            star,
            interaction,
            weights,
            dust,
            parent_fiber,
            component_fiber,
            marked_shift,
            component_shift,
            parent_flux,
            component_upper,
        };
        if (generate_joint_streaming_frontier_operator_bound(
                law,
                tree,
                target_edges,
                degree_sigma,
                temperature,
                root_path_hash,
                order_seed,
                mark_seed,
                target,
                output_edges,
                output_utilities,
                integer_diagnostics,
                real_diagnostics
            )) {
            last_used_frontier_free_topk = 1;
            return 0;
        }
    } catch (const std::bad_alloc& error) {
        trace_event("A_bad_alloc",error.what());
        return 90;

    } catch (const std::exception& error) {
        trace_event("A_exception",error.what());
        return 91;

    }
    trace_event("A_unexpected_false", "No automatic restart; explicit failure");
    return 92;

}

static long long trace_ties(){return last_binary64_serial_tie_comparison_count;}
static long long trace_certs(){return last_tie_cache_certificate_build_count;}
static long long trace_reuses(){return last_tie_cache_certificate_reuse_count;}

ROUTE_S_EXPORT long long route_s_research_admitted(){return research_admitted;}

ROUTE_S_EXPORT double route_s_diagnostic_io_seconds(){return diagnostic_io_seconds;}
ROUTE_S_EXPORT unsigned long long route_s_diagnostic_io_scopes(){return diagnostic_io_scopes;}
