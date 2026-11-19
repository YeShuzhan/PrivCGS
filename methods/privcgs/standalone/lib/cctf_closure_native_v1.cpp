#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

namespace {

struct Edge {
    int u;
    int v;
};

static inline std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

static inline std::uint64_t priority(int u, int v, std::uint64_t seed) {
    const auto left = static_cast<std::uint64_t>(static_cast<std::uint32_t>(u));
    const auto right = static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
    return splitmix64(seed ^ (left << 32U) ^ right);
}

}  // namespace

// Stable mutual-top-D projection followed by exact global wedge and triangle
// counting. Input is a validated symmetric CSR. Output counters are:
// projected edges, projected triangles, projected wedges, maximum degree,
// discarded edges.
extern "C" int cctf_closure_counts_v1(
    int n,
    std::int64_t directed_edge_count,
    const std::int64_t* offsets,
    const std::int64_t* neighbors,
    int degree_cap,
    std::uint64_t public_seed,
    std::int64_t* counters) {
    if (n < 2 || directed_edge_count < 0 || (directed_edge_count & 1LL) != 0 ||
        offsets == nullptr || neighbors == nullptr || degree_cap < 1 ||
        counters == nullptr) {
        return 1;
    }
    try {
        const std::int64_t expected_edges = directed_edge_count / 2;
        std::vector<Edge> edges;
        edges.reserve(static_cast<std::size_t>(expected_edges));
        using RankedEdge = std::pair<std::uint64_t, std::int64_t>;
        std::vector<std::priority_queue<RankedEdge>> selected(static_cast<std::size_t>(n));

        auto consider = [degree_cap](std::priority_queue<RankedEdge>& heap,
                                     const RankedEdge& candidate) {
            if (static_cast<int>(heap.size()) < degree_cap) {
                heap.push(candidate);
            } else if (candidate < heap.top()) {
                heap.pop();
                heap.push(candidate);
            }
        };

        for (int u = 0; u < n; ++u) {
            if (offsets[u] > offsets[u + 1]) {
                return 2;
            }
            for (std::int64_t p = offsets[u]; p < offsets[u + 1]; ++p) {
                const std::int64_t raw_v = neighbors[p];
                if (raw_v <= u) {
                    continue;
                }
                if (raw_v >= n) {
                    return 3;
                }
                const int v = static_cast<int>(raw_v);
                const std::int64_t edge_id = static_cast<std::int64_t>(edges.size());
                edges.push_back(Edge{u, v});
                const RankedEdge ranked{priority(u, v, public_seed), edge_id};
                consider(selected[static_cast<std::size_t>(u)], ranked);
                consider(selected[static_cast<std::size_t>(v)], ranked);
            }
        }
        if (static_cast<std::int64_t>(edges.size()) != expected_edges) {
            return 4;
        }

        std::vector<std::uint8_t> selected_endpoint(edges.size(), 0U);
        for (int u = 0; u < n; ++u) {
            auto& heap = selected[static_cast<std::size_t>(u)];
            while (!heap.empty()) {
                ++selected_endpoint[static_cast<std::size_t>(heap.top().second)];
                heap.pop();
            }
        }

        std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(n));
        std::int64_t projected_edges = 0;
        for (std::size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
            if (selected_endpoint[edge_id] != 2U) {
                continue;
            }
            const Edge edge = edges[edge_id];
            adjacency[static_cast<std::size_t>(edge.u)].push_back(edge.v);
            adjacency[static_cast<std::size_t>(edge.v)].push_back(edge.u);
            ++projected_edges;
        }

        int maximum_degree = 0;
        std::int64_t wedges = 0;
        for (int u = 0; u < n; ++u) {
            auto& row = adjacency[static_cast<std::size_t>(u)];
            std::sort(row.begin(), row.end());
            const auto d = static_cast<std::int64_t>(row.size());
            maximum_degree = std::max(maximum_degree, static_cast<int>(d));
            wedges += d * (d - 1) / 2;
        }

        std::vector<std::vector<int>> forward(static_cast<std::size_t>(n));
        for (std::size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
            if (selected_endpoint[edge_id] != 2U) {
                continue;
            }
            const Edge edge = edges[edge_id];
            const auto left_key = std::make_pair(adjacency[edge.u].size(), edge.u);
            const auto right_key = std::make_pair(adjacency[edge.v].size(), edge.v);
            if (left_key < right_key) {
                forward[edge.u].push_back(edge.v);
            } else {
                forward[edge.v].push_back(edge.u);
            }
        }

        std::vector<int> mark(static_cast<std::size_t>(n), -1);
        std::int64_t triangles = 0;
        for (int u = 0; u < n; ++u) {
            for (const int w : forward[static_cast<std::size_t>(u)]) {
                mark[static_cast<std::size_t>(w)] = u;
            }
            for (const int v : forward[static_cast<std::size_t>(u)]) {
                for (const int w : forward[static_cast<std::size_t>(v)]) {
                    if (mark[static_cast<std::size_t>(w)] == u) {
                        ++triangles;
                    }
                }
            }
        }

        counters[0] = projected_edges;
        counters[1] = triangles;
        counters[2] = wedges;
        counters[3] = maximum_degree;
        counters[4] = static_cast<std::int64_t>(edges.size() - projected_edges);
        return 0;
    } catch (...) {
        return 5;
    }
}
