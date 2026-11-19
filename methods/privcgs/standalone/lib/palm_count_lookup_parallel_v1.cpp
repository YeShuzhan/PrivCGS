#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

void lookup_range(
    const std::uint64_t* unique,
    const std::int64_t* count,
    std::int64_t unique_size,
    const std::uint64_t* query,
    std::int64_t* output,
    std::int64_t start,
    std::int64_t stop) {
    for (std::int64_t position = start; position < stop; ++position) {
        const std::uint64_t key = query[position];
        std::int64_t left = 0;
        std::int64_t right = unique_size;
        while (left < right) {
            const std::int64_t middle = left + (right - left) / 2;
            if (unique[middle] < key) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        output[position] =
            (left < unique_size && unique[left] == key) ? count[left] : 0;
    }
}

}  // namespace

extern "C" __declspec(dllexport) std::int32_t route_s_parallel_count_lookup_v1(
    const std::uint64_t* unique,
    const std::int64_t* count,
    std::int64_t unique_size,
    const std::uint64_t* query,
    std::int64_t query_size,
    std::int64_t* output,
    std::int32_t requested_threads) {
    if (unique_size < 0 || query_size < 0 || requested_threads <= 0) {
        return 1;
    }
    try {
        if (query_size == 0) {
            return 0;
        }
        const std::int32_t thread_count = static_cast<std::int32_t>(std::min<std::int64_t>(
            std::max<std::int32_t>(requested_threads, 1), query_size));
        if (thread_count == 1 || query_size < 262144) {
            lookup_range(unique, count, unique_size, query, output, 0, query_size);
            return 0;
        }
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(thread_count));
        for (std::int32_t worker = 0; worker < thread_count; ++worker) {
            const std::int64_t start = query_size * worker / thread_count;
            const std::int64_t stop = query_size * (worker + 1) / thread_count;
            workers.emplace_back(
                lookup_range,
                unique,
                count,
                unique_size,
                query,
                output,
                start,
                stop);
        }
        for (auto& worker : workers) {
            worker.join();
        }
        return 0;
    } catch (const std::bad_alloc&) {
        return 2;
    } catch (...) {
        return 3;
    }
}
