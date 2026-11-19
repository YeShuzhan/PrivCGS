#include <algorithm>
#include <cstdint>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C"
#endif

using i64 = std::int64_t;

ROUTE_S_EXPORT int route_s_validate_undirected_csr(
    i64 n_nodes,
    i64 neighbor_count,
    const i64* offsets,
    const i64* neighbors
) {
    if (n_nodes <= 1 || neighbor_count < 0 || (neighbor_count & 1LL) != 0) {
        return 1;
    }
    if (offsets[0] != 0 || offsets[n_nodes] != neighbor_count) {
        return 2;
    }
    for (i64 node = 0; node < n_nodes; ++node) {
        if (offsets[node] < 0 || offsets[node] > offsets[node + 1] ||
            offsets[node + 1] > neighbor_count) {
            return 3;
        }
    }
    for (i64 node = 0; node < n_nodes; ++node) {
        i64 previous = -1;
        for (i64 position = offsets[node]; position < offsets[node + 1]; ++position) {
            const i64 neighbor = neighbors[position];
            if (neighbor < 0 || neighbor >= n_nodes) {
                return 4;
            }
            if (neighbor == node) {
                return 5;
            }
            if (position > offsets[node] && neighbor <= previous) {
                return 6;
            }
            previous = neighbor;
        }
    }
    for (i64 node = 0; node < n_nodes; ++node) {
        for (i64 position = offsets[node]; position < offsets[node + 1]; ++position) {
            const i64 neighbor = neighbors[position];
            const i64* begin = neighbors + offsets[neighbor];
            const i64* end = neighbors + offsets[neighbor + 1];
            if (!std::binary_search(begin, end, node)) {
                return 7;
            }
        }
    }
    return 0;
}
