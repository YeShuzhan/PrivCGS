#include <cstdint>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

std::uint64_t edge_counter(std::uint64_t seed, std::int64_t left,
                           std::int64_t right) {
    std::uint64_t value = seed;
    value ^= (static_cast<std::uint64_t>(left) + 1ULL) * 0xD6E8FEB86659FD93ULL;
    value ^= (static_cast<std::uint64_t>(right) + 1ULL) * 0xA5A3564E27F8862FULL;
    return splitmix64(value);
}

std::uint64_t exact_uniform_index(std::uint64_t counter,
                                  std::uint64_t support) {
    if (support <= 1) return 0;
    const std::uint64_t threshold = static_cast<std::uint64_t>(-support) % support;
    std::uint64_t value = splitmix64(counter);
    while (value < threshold) {
        counter = splitmix64(counter ^ 0x8CB92BA72F3D8DD7ULL);
        value = splitmix64(counter);
    }
    const unsigned __int128 product =
        static_cast<unsigned __int128>(value) * support;
    return static_cast<std::uint64_t>(product >> 64);
}

}  // namespace

ROUTE_S_EXPORT int route_s_counter_palm_tokens(
    std::int64_t n_nodes,
    std::int64_t edge_count,
    const std::int64_t* source_edges,
    const std::int64_t* csr_offsets,
    const std::int64_t* csr_neighbors,
    std::uint64_t public_seed,
    std::uint64_t* output_keys,
    std::int64_t* output_count,
    std::int64_t* skipped_count) {
    if (n_nodes <= 0 || edge_count < 0 || source_edges == nullptr ||
        csr_offsets == nullptr || csr_neighbors == nullptr || output_keys == nullptr ||
        output_count == nullptr || skipped_count == nullptr) return 1;
    std::int64_t produced = 0;
    std::int64_t skipped = 0;
    for (std::int64_t index = 0; index < edge_count; ++index) {
        const std::int64_t left = source_edges[2 * index];
        const std::int64_t right = source_edges[2 * index + 1];
        if (left < 0 || right < 0 || left >= n_nodes || right >= n_nodes ||
            left >= right) return 2;
        const std::uint64_t counter = edge_counter(public_seed, left, right);
        const bool reverse =
            (splitmix64(counter ^ 0x243F6A8885A308D3ULL) & 1ULL) != 0;
        const std::int64_t pivot = reverse ? right : left;
        const std::int64_t other = reverse ? left : right;
        const std::int64_t start = csr_offsets[pivot];
        const std::int64_t degree = csr_offsets[pivot + 1] - start;
        if (degree <= 1) {
            ++skipped;
            continue;
        }
        const std::uint64_t offset = exact_uniform_index(
            counter ^ 0x13198A2E03707344ULL,
            static_cast<std::uint64_t>(degree - 1));
        std::int64_t selected =
            csr_neighbors[start + static_cast<std::int64_t>(offset)];
        if (selected == other) selected = csr_neighbors[start + degree - 1];
        if (selected == other || selected == pivot || selected < 0 ||
            selected >= n_nodes) {
            ++skipped;
            continue;
        }
        const std::int64_t first = selected < other ? selected : other;
        const std::int64_t second = selected < other ? other : selected;
        output_keys[produced] = static_cast<std::uint64_t>(first) *
                                    static_cast<std::uint64_t>(n_nodes) +
                                static_cast<std::uint64_t>(second);
        ++produced;
    }
    output_count[0] = produced;
    skipped_count[0] = skipped;
    return 0;
}
