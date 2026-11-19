#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C"
#endif

namespace {

using i64 = std::int64_t;
using u64 = std::uint64_t;

constexpr u64 empty_key = std::numeric_limits<u64>::max();

u64 mix(u64 value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

u64 edge_key(const i64* edges, i64 index, i64 n_nodes) {
    const i64 first = edges[2 * index];
    const i64 second = edges[2 * index + 1];
    const u64 left = static_cast<u64>(std::min(first, second));
    const u64 right = static_cast<u64>(std::max(first, second));
    return left * static_cast<u64>(n_nodes) + right;
}

class flat_set {
public:
    explicit flat_set(u64 expected) {
        u64 capacity = 8;
        while (capacity - capacity / 4 < std::max<u64>(expected, 1)) {
            if (capacity > (std::numeric_limits<u64>::max() >> 1U)) {
                throw std::bad_alloc();
            }
            capacity <<= 1U;
        }
        slots_.assign(static_cast<std::size_t>(capacity), empty_key);
        mask_ = capacity - 1U;
    }

    bool contains(u64 key) const {
        u64 slot = mix(key) & mask_;
        for (;;) {
            const u64 current = slots_[static_cast<std::size_t>(slot)];
            if (current == empty_key) {
                return false;
            }
            if (current == key) {
                return true;
            }
            slot = (slot + 1U) & mask_;
        }
    }

    void insert(u64 key) {
        u64 slot = mix(key) & mask_;
        for (;;) {
            u64& current = slots_[static_cast<std::size_t>(slot)];
            if (current == key) {
                return;
            }
            if (current == empty_key) {
                current = key;
                ++size_;
                return;
            }
            slot = (slot + 1U) & mask_;
        }
    }

    i64 size() const { return static_cast<i64>(size_); }
    i64 capacity() const { return static_cast<i64>(slots_.size()); }

private:
    std::vector<u64> slots_;
    u64 mask_ = 0;
    u64 size_ = 0;
};

void insert_edges(flat_set& reserved, const i64* edges, i64 count, i64 n_nodes) {
    for (i64 index = 0; index < count; ++index) {
        reserved.insert(edge_key(edges, index, n_nodes));
    }
}

}  // namespace

ROUTE_S_EXPORT int route_s_completion_admit(
    i64 n_nodes,
    i64 bundle_count,
    const i64* core_negative_old,
    i64 core_negative_count,
    const i64* core_negative_new,
    const i64* core_positive_old,
    i64 core_positive_count,
    const i64* core_positive_new,
    const i64* descent_negative_offsets,
    const i64* descent_negative_old,
    const i64* descent_negative_new,
    const i64* descent_positive_offsets,
    const i64* descent_positive_old,
    const i64* descent_positive_new,
    const std::int8_t* positive_direction,
    const i64* required_count,
    std::int8_t* negative_mask,
    std::int8_t* positive_mask,
    i64* selected_count,
    i64* reserved_count,
    i64* table_capacity
) {
    if (n_nodes <= 1 || bundle_count <= 0 ||
        descent_negative_offsets[0] != 0 || descent_positive_offsets[0] != 0) {
        return 1;
    }
    try {
        u64 expected = 2ULL * static_cast<u64>(
            core_negative_count + core_positive_count
        );
        for (i64 index = 0; index < bundle_count; ++index) {
            if (required_count[index] < 0) {
                return 2;
            }
            const i64* offsets = positive_direction[index] != 0
                ? descent_positive_offsets : descent_negative_offsets;
            const i64 available = offsets[index + 1] - offsets[index];
            if (available < 0) {
                return 2;
            }
            expected += 2ULL * static_cast<u64>(
                std::min(required_count[index], available)
            );
        }
        flat_set reserved(expected);
        insert_edges(reserved, core_negative_old, core_negative_count, n_nodes);
        insert_edges(reserved, core_negative_new, core_negative_count, n_nodes);
        insert_edges(reserved, core_positive_old, core_positive_count, n_nodes);
        insert_edges(reserved, core_positive_new, core_positive_count, n_nodes);

        i64 selected = 0;
        for (i64 bundle = 0; bundle < bundle_count; ++bundle) {
            const bool positive = positive_direction[bundle] != 0;
            const i64* offsets = positive
                ? descent_positive_offsets : descent_negative_offsets;
            const i64* old_edges = positive
                ? descent_positive_old : descent_negative_old;
            const i64* new_edges = positive
                ? descent_positive_new : descent_negative_new;
            std::int8_t* mask = positive ? positive_mask : negative_mask;
            i64 admitted = 0;
            for (i64 unit = offsets[bundle];
                 unit < offsets[bundle + 1] && admitted < required_count[bundle];
                 ++unit) {
                const u64 old_key = edge_key(old_edges, unit, n_nodes);
                const u64 new_key = edge_key(new_edges, unit, n_nodes);
                if (reserved.contains(old_key) || reserved.contains(new_key)) {
                    continue;
                }
                mask[unit] = 1;
                reserved.insert(old_key);
                reserved.insert(new_key);
                ++admitted;
                ++selected;
            }
        }
        selected_count[0] = selected;
        reserved_count[0] = reserved.size();
        table_capacity[0] = reserved.capacity();
        return 0;
    } catch (const std::bad_alloc&) {
        return 3;
    } catch (...) {
        return 4;
    }
}
