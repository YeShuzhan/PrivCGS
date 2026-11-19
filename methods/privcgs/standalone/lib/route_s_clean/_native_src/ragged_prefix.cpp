#include <algorithm>
#include <cstdint>
#include <emmintrin.h>
#include <vector>

#if defined(_WIN32)
#define ROUTE_S_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTE_S_EXPORT extern "C" __attribute__((visibility("default")))
#endif

ROUTE_S_EXPORT std::int32_t route_s_ragged_prefix(
    std::int64_t bundle_count,
    const std::int64_t* state_offsets,
    const std::int64_t* negative_offsets,
    const std::int64_t* positive_offsets,
    const double* negative,
    const double* positive,
    double* output) {
    if (bundle_count < 0 || state_offsets == nullptr ||
        negative_offsets == nullptr || positive_offsets == nullptr ||
        output == nullptr || state_offsets[0] != 0 ||
        negative_offsets[0] != 0 || positive_offsets[0] != 0) {
        return 1;
    }
    std::int64_t maximum_negative = 0;
    for (std::int64_t bundle = 0; bundle < bundle_count; ++bundle) {
        const std::int64_t negative_count =
            negative_offsets[bundle + 1] - negative_offsets[bundle];
        const std::int64_t positive_count =
            positive_offsets[bundle + 1] - positive_offsets[bundle];
        const std::int64_t state_count =
            state_offsets[bundle + 1] - state_offsets[bundle];
        if (negative_count < 0 || positive_count < 0 ||
            state_count != negative_count + positive_count + 1) {
            return 2;
        }
        maximum_negative = std::max(maximum_negative, negative_count);
    }
    if ((negative_offsets[bundle_count] > 0 && negative == nullptr) ||
        (positive_offsets[bundle_count] > 0 && positive == nullptr)) {
        return 3;
    }
    std::vector<double> cumulative(static_cast<std::size_t>(maximum_negative));
    for (std::int64_t bundle = 0; bundle < bundle_count; ++bundle) {
        const std::int64_t state_start = state_offsets[bundle];
        const std::int64_t negative_start = negative_offsets[bundle];
        const std::int64_t negative_stop = negative_offsets[bundle + 1];
        const std::int64_t positive_start = positive_offsets[bundle];
        const std::int64_t positive_stop = positive_offsets[bundle + 1];
        const std::int64_t negative_count = negative_stop - negative_start;
        // The packed graph law freezes binary64 scalar left-to-right cumsum
        // order.  Explicit SSE2 addition prevents x87 excess precision and
        // optimizer reassociation when later score tilts mix magnitudes.
        __m128d sum = _mm_set_sd(0.0);
        for (std::int64_t index = 0; index < negative_count; ++index) {
            sum = _mm_add_sd(sum, _mm_load_sd(negative + negative_start + index));
            _mm_store_sd(&cumulative[static_cast<std::size_t>(index)], sum);
        }
        for (std::int64_t index = 0; index < negative_count; ++index) {
            output[state_start + index] = cumulative[
                static_cast<std::size_t>(negative_count - index - 1)];
        }
        output[state_start + negative_count] = 0.0;
        sum = _mm_set_sd(0.0);
        for (std::int64_t index = positive_start; index < positive_stop; ++index) {
            sum = _mm_add_sd(sum, _mm_load_sd(positive + index));
            _mm_store_sd(
                output + state_start + negative_count + 1 + index - positive_start,
                sum);
        }
    }
    return 0;
}
