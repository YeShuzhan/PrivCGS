#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

std::uint64_t edge_key(std::int64_t n, std::int64_t left, std::int64_t right) {
    const std::int64_t low = std::min(left, right);
    const std::int64_t high = std::max(left, right);
    return static_cast<std::uint64_t>(low) * static_cast<std::uint64_t>(n) +
           static_cast<std::uint64_t>(high);
}

bool contains_key(const std::uint64_t* keys, std::int64_t count,
                  std::uint64_t value) {
    return std::binary_search(keys, keys + count, value);
}

std::uint64_t ordered_double_key(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits >> 63) ? ~bits : (bits ^ (std::uint64_t{1} << 63));
}

template <class Key>
void stable_radix_field(std::vector<std::int64_t>& order,
                        std::vector<std::int64_t>& scratch,
                        int byte_count, Key key) {
    for (int byte = 0; byte < byte_count; ++byte) {
        std::array<std::size_t, 256> count{};
        const int shift = 8 * byte;
        for (const std::int64_t index : order) {
            ++count[static_cast<std::uint8_t>(key(index) >> shift)];
        }
        std::array<std::size_t, 256> offset{};
        for (std::size_t value = 1; value < offset.size(); ++value) {
            offset[value] = offset[value - 1] + count[value - 1];
        }
        for (const std::int64_t index : order) {
            const std::uint8_t digit =
                static_cast<std::uint8_t>(key(index) >> shift);
            scratch[offset[digit]++] = index;
        }
        order.swap(scratch);
    }
}

}  // namespace

ROUTE_S_EXPORT int route_s_count_relocation_candidates(
    std::int64_t n, const std::int64_t* path, const std::int64_t* offsets,
    const std::int64_t* neighbors, std::int64_t edge_count,
    const std::uint64_t* edge_keys, std::int64_t* count_out) {
    if (n <= 0 || path == nullptr || offsets == nullptr || neighbors == nullptr ||
        edge_count < 0 || edge_keys == nullptr || count_out == nullptr) return 1;
    std::int64_t count = 0;
    for (std::int64_t position = 0; position + 1 < n; ++position) {
        const std::int64_t left = path[position];
        const std::int64_t right = path[position + 1];
        for (std::int64_t cursor = offsets[left]; cursor < offsets[left + 1]; ++cursor) {
            const std::int64_t neighbor = neighbors[cursor];
            if (neighbor == right) continue;
            if (!contains_key(edge_keys, edge_count, edge_key(n, right, neighbor))) ++count;
        }
        for (std::int64_t cursor = offsets[right]; cursor < offsets[right + 1]; ++cursor) {
            const std::int64_t neighbor = neighbors[cursor];
            if (neighbor == left) continue;
            if (!contains_key(edge_keys, edge_count, edge_key(n, left, neighbor))) ++count;
        }
    }
    count_out[0] = count;
    return 0;
}

ROUTE_S_EXPORT int route_s_fill_relocation_candidates(
    std::int64_t n, const std::int64_t* path, const std::int64_t* offsets,
    const std::int64_t* neighbors, std::int64_t edge_count,
    const std::uint64_t* edge_keys, std::int64_t capacity,
    std::int64_t* path_index, std::int8_t* direction,
    std::int64_t* old_left, std::int64_t* old_right,
    std::int64_t* new_left, std::int64_t* new_right) {
    if (capacity < 0 || path_index == nullptr || direction == nullptr ||
        old_left == nullptr || old_right == nullptr || new_left == nullptr ||
        new_right == nullptr) return 1;
    std::int64_t output = 0;
    auto emit = [&](std::int64_t position, std::int8_t sign,
                    std::int64_t old_u, std::int64_t old_v,
                    std::int64_t new_u, std::int64_t new_v) -> bool {
        if (output >= capacity) return false;
        path_index[output] = position;
        direction[output] = sign;
        old_left[output] = std::min(old_u, old_v);
        old_right[output] = std::max(old_u, old_v);
        new_left[output] = std::min(new_u, new_v);
        new_right[output] = std::max(new_u, new_v);
        ++output;
        return true;
    };
    for (std::int64_t position = 0; position + 1 < n; ++position) {
        const std::int64_t left = path[position];
        const std::int64_t right = path[position + 1];
        for (std::int64_t cursor = offsets[left]; cursor < offsets[left + 1]; ++cursor) {
            const std::int64_t neighbor = neighbors[cursor];
            if (neighbor == right) continue;
            if (!contains_key(edge_keys, edge_count, edge_key(n, right, neighbor)) &&
                !emit(position, 1, left, neighbor, right, neighbor)) return 2;
        }
        for (std::int64_t cursor = offsets[right]; cursor < offsets[right + 1]; ++cursor) {
            const std::int64_t neighbor = neighbors[cursor];
            if (neighbor == left) continue;
            if (!contains_key(edge_keys, edge_count, edge_key(n, left, neighbor)) &&
                !emit(position, -1, right, neighbor, left, neighbor)) return 3;
        }
    }
    return output == capacity ? 0 : 4;
}

ROUTE_S_EXPORT int route_s_sort_and_admit_relocation_ids(
    std::int64_t count, const double* gain, std::int64_t unique_dyad_count,
    const std::int64_t* old_id, const std::int64_t* new_id,
    std::int64_t* selected_index, std::int64_t* selected_count_out) {
    if (count < 0 || gain == nullptr || unique_dyad_count < 0 || old_id == nullptr ||
        new_id == nullptr || selected_index == nullptr || selected_count_out == nullptr)
        return 1;
    std::vector<std::int64_t> order(static_cast<std::size_t>(count));
    std::vector<std::int64_t> scratch(static_cast<std::size_t>(count));
    std::iota(order.begin(), order.end(), std::int64_t{0});
    stable_radix_field(order, scratch, 8, [&](std::int64_t index) {
        return ~ordered_double_key(gain[index]);
    });
    std::vector<std::uint8_t> reserved(
        static_cast<std::size_t>(unique_dyad_count), std::uint8_t{0});
    std::int64_t selected = 0;
    for (const std::int64_t index : order) {
        const std::int64_t old_value = old_id[index];
        const std::int64_t new_value = new_id[index];
        if (old_value < 0 || old_value >= unique_dyad_count || new_value < 0 ||
            new_value >= unique_dyad_count) return 2;
        if (reserved[static_cast<std::size_t>(old_value)] ||
            reserved[static_cast<std::size_t>(new_value)]) continue;
        reserved[static_cast<std::size_t>(old_value)] = 1;
        reserved[static_cast<std::size_t>(new_value)] = 1;
        selected_index[selected++] = index;
    }
    selected_count_out[0] = selected;
    return 0;
}

ROUTE_S_EXPORT int route_s_count_descent_candidates(
    std::int64_t n, const std::int64_t* path, const std::int8_t* direction,
    const std::int64_t* requested, const std::int64_t* offsets,
    const std::int64_t* neighbors, std::int64_t edge_count,
    const std::uint64_t* edge_keys, std::int64_t* count_by_bundle,
    std::int64_t* count_out) {
    if (n <= 0 || path == nullptr || direction == nullptr || requested == nullptr ||
        offsets == nullptr || neighbors == nullptr || edge_count < 0 ||
        edge_keys == nullptr || count_by_bundle == nullptr || count_out == nullptr)
        return 1;
    std::int64_t total = 0;
    for (std::int64_t position = 0; position + 1 < n; ++position) {
        count_by_bundle[position] = 0;
        if (requested[position] <= 0 || direction[position] == 0) continue;
        const std::int64_t left = path[position];
        const std::int64_t right = path[position + 1];
        const std::int64_t donor = direction[position] > 0 ? left : right;
        const std::int64_t receiver = direction[position] > 0 ? right : left;
        for (std::int64_t cursor = offsets[donor]; cursor < offsets[donor + 1]; ++cursor) {
            const std::int64_t neighbor = neighbors[cursor];
            if (neighbor == receiver) continue;
            if (!contains_key(edge_keys, edge_count, edge_key(n, receiver, neighbor))) {
                ++count_by_bundle[position];
                ++total;
            }
        }
    }
    count_out[0] = total;
    return 0;
}

ROUTE_S_EXPORT int route_s_fill_descent_candidates(
    std::int64_t n, const std::int64_t* path, const std::int8_t* direction,
    const std::int64_t* requested, const std::int64_t* offsets,
    const std::int64_t* neighbors, std::int64_t edge_count,
    const std::uint64_t* edge_keys, std::int64_t capacity,
    std::int64_t* path_index, std::int64_t* old_left,
    std::int64_t* old_right, std::int64_t* new_left,
    std::int64_t* new_right) {
    if (n <= 0 || capacity < 0 || path_index == nullptr || old_left == nullptr ||
        old_right == nullptr || new_left == nullptr || new_right == nullptr) return 1;
    std::int64_t output = 0;
    for (std::int64_t position = 0; position + 1 < n; ++position) {
        if (requested[position] <= 0 || direction[position] == 0) continue;
        const std::int64_t left = path[position];
        const std::int64_t right = path[position + 1];
        const std::int64_t donor = direction[position] > 0 ? left : right;
        const std::int64_t receiver = direction[position] > 0 ? right : left;
        for (std::int64_t cursor = offsets[donor]; cursor < offsets[donor + 1]; ++cursor) {
            const std::int64_t neighbor = neighbors[cursor];
            if (neighbor == receiver) continue;
            if (contains_key(edge_keys, edge_count, edge_key(n, receiver, neighbor))) continue;
            if (output >= capacity) return 2;
            path_index[output] = position;
            old_left[output] = std::min(donor, neighbor);
            old_right[output] = std::max(donor, neighbor);
            new_left[output] = std::min(receiver, neighbor);
            new_right[output] = std::max(receiver, neighbor);
            ++output;
        }
    }
    return output == capacity ? 0 : 3;
}

ROUTE_S_EXPORT int route_s_admit_descent_candidates(
    std::int64_t candidate_count, std::int64_t bundle_count,
    std::int64_t unique_dyad_count, const std::int64_t* order,
    const std::int64_t* path_index, const std::int64_t* requested,
    const std::int64_t* old_id, const std::int64_t* new_id,
    std::int64_t* selected_index, std::int64_t* admitted_by_bundle,
    std::int64_t* selected_count_out) {
    if (candidate_count < 0 || bundle_count < 0 || unique_dyad_count < 0 ||
        order == nullptr || path_index == nullptr || requested == nullptr ||
        old_id == nullptr || new_id == nullptr || selected_index == nullptr ||
        admitted_by_bundle == nullptr || selected_count_out == nullptr) return 1;
    std::vector<unsigned char> reserved(static_cast<std::size_t>(unique_dyad_count), 0);
    std::fill(admitted_by_bundle, admitted_by_bundle + bundle_count, 0);
    std::int64_t selected = 0;
    for (std::int64_t rank = 0; rank < candidate_count; ++rank) {
        const std::int64_t index = order[rank];
        if (index < 0 || index >= candidate_count) return 2;
        const std::int64_t bundle = path_index[index];
        if (bundle < 0 || bundle >= bundle_count) return 3;
        if (admitted_by_bundle[bundle] >= requested[bundle]) continue;
        const std::int64_t old_value = old_id[index];
        const std::int64_t new_value = new_id[index];
        if (old_value < 0 || old_value >= unique_dyad_count || new_value < 0 ||
            new_value >= unique_dyad_count) return 4;
        if (reserved[old_value] || reserved[new_value]) continue;
        reserved[old_value] = 1;
        reserved[new_value] = 1;
        ++admitted_by_bundle[bundle];
        selected_index[selected++] = index;
    }
    selected_count_out[0] = selected;
    return 0;
}
