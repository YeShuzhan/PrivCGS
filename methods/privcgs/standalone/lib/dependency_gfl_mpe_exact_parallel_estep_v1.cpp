#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace {

struct GroupRecord {
  std::int64_t begin = 0;
  std::int64_t end = 0;
  std::int64_t centre = 0;
  int lower = 0;
  int upper = -1;
  int width = 0;
  int d_min = 0;
  std::int64_t value_offset = 0;
  int error = 0;
};

}  // namespace

#ifdef GCFR_GFL_MPE_RKSS1

extern "C" int gcfr_gfl_mpe_rkss1_precompute_rows_v1(
    std::int64_t row_count,
    const double* observation_sorted,
    const std::int64_t* center_sorted,
    double sigma,
    double* row_log_x,
    double* row_x,
    double* row_half_square) {
  if (row_count < 0 || observation_sorted == nullptr || center_sorted == nullptr ||
      !(sigma > 0.0) || row_log_x == nullptr || row_x == nullptr ||
      row_half_square == nullptr) {
    return 1;
  }
  const double inverse_sigma_squared = 1.0 / (sigma * sigma);
  for (std::int64_t row = 0; row < row_count; ++row) {
    const double residual =
        observation_sorted[row] - static_cast<double>(center_sorted[row]);
    const double log_x = residual * inverse_sigma_squared;
    const double standardized_residual = residual / sigma;
    row_log_x[row] = log_x;
    row_x[row] = std::exp(log_x);
    row_half_square[row] =
        0.5 * standardized_residual * standardized_residual;
  }
  return 0;
}

#define GCFR_GFL_MPE_ESTEP_NAME gcfr_gfl_mpe_exact_parallel_estep_rkss1_v1

#else

#define GCFR_GFL_MPE_ESTEP_NAME gcfr_gfl_mpe_exact_parallel_estep_v1

#endif

// Exact parallel schedule for the grouped full-log E step.
//
// Each centre fibre owns disjoint observation rows and computes its moment
// vector in the original row/position order.  Fibre contributions are stored
// independently.  After all workers join, the moving-log-scale component
// accumulation is replayed serially in the original fibre/position order.
// Thus scheduling cannot affect any binary64 operation that reaches output.
extern "C" int GCFR_GFL_MPE_ESTEP_NAME(
    std::int64_t row_count,
    const double* observation_sorted,
    const std::int64_t* center_sorted,
    const std::int64_t* original_index_sorted,
    std::int64_t group_count,
    const std::int64_t* group_starts,
    int half_bandwidth,
    int component_count,
    const double* log_probability,
    double sigma,
    int requested_thread_count,
#ifdef GCFR_GFL_MPE_RKSS1
    const double* row_log_x_sorted,
    const double* row_x_sorted,
    const double* row_half_square_sorted,
#endif
    double* log_expected,
    double* row_log_density_original) {
  if (row_count < 0 || observation_sorted == nullptr ||
      center_sorted == nullptr || original_index_sorted == nullptr ||
      group_count < 0 || group_starts == nullptr || half_bandwidth < 0 ||
      component_count <= 0 || log_probability == nullptr || !(sigma > 0.0) ||
      requested_thread_count <= 0 || log_expected == nullptr ||
      row_log_density_original == nullptr) {
    return 1;
  }
  if ((row_count == 0 && group_count != 0) ||
      (row_count > 0 && group_count <= 0) || group_starts[0] != 0 ||
      group_starts[group_count] != row_count) {
    return 1;
  }

  const double negative_infinity =
      -std::numeric_limits<double>::infinity();
  const double inverse_sigma_squared = 1.0 / (sigma * sigma);
  std::vector<GroupRecord> groups(static_cast<std::size_t>(group_count));
  std::int64_t total_group_width = 0;
  for (std::int64_t group_index = 0; group_index < group_count; ++group_index) {
    GroupRecord& group = groups[static_cast<std::size_t>(group_index)];
    group.begin = group_starts[group_index];
    group.end = group_starts[group_index + 1];
    group.value_offset = total_group_width;
    if (group.begin < 0 || group.begin >= group.end || group.end > row_count) {
      group.error = 1;
      continue;
    }
    group.centre = center_sorted[group.begin];
    if (center_sorted[group.end - 1] != group.centre ||
        (group_index > 0 && center_sorted[group.begin - 1] == group.centre) ||
        (group_index + 1 < group_count &&
         center_sorted[group.end] == group.centre)) {
      group.error = 1;
      continue;
    }
    const std::int64_t raw_lower = group.centre - half_bandwidth;
    const std::int64_t raw_upper = group.centre + half_bandwidth;
    group.lower = static_cast<int>(std::max<std::int64_t>(0, raw_lower));
    group.upper = static_cast<int>(
        std::min<std::int64_t>(component_count - 1, raw_upper));
    if (group.lower > group.upper) {
      group.error = 2;
      continue;
    }
    group.width = group.upper - group.lower + 1;
    group.d_min = group.lower - static_cast<int>(group.centre);
    total_group_width += group.width;
  }
  for (const GroupRecord& group : groups) {
    if (group.error != 0) return group.error;
  }

  std::vector<double> group_value(
      static_cast<std::size_t>(total_group_width), negative_infinity);
  std::vector<int> group_error(static_cast<std::size_t>(group_count), 0);
  std::atomic<std::int64_t> next_group{0};
  const int thread_count = std::max(
      1, std::min<int>(requested_thread_count, static_cast<int>(group_count)));

  auto worker = [&]() {
    std::vector<double> log_coefficient;
    std::vector<double> coefficient;
    std::vector<double> moment;
#ifdef GCFR_GFL_MPE_PPRS1
    std::vector<double> power_path;
#endif
    while (true) {
      const std::int64_t group_index =
          next_group.fetch_add(1, std::memory_order_relaxed);
      if (group_index >= group_count) break;
      const GroupRecord& group =
          groups[static_cast<std::size_t>(group_index)];
      const int width = group.width;
      log_coefficient.resize(static_cast<std::size_t>(width));
      coefficient.resize(static_cast<std::size_t>(width));
      moment.assign(static_cast<std::size_t>(width), 0.0);
#ifdef GCFR_GFL_MPE_PPRS1
      power_path.resize(static_cast<std::size_t>(width));
#endif

      double coefficient_scale = negative_infinity;
      for (int position = 0; position < width; ++position) {
        const int component = group.lower + position;
        const int d = component - static_cast<int>(group.centre);
        const double value =
            log_probability[component] -
            0.5 * static_cast<double>(d) * static_cast<double>(d) *
                inverse_sigma_squared;
        log_coefficient[static_cast<std::size_t>(position)] = value;
        coefficient_scale = std::max(coefficient_scale, value);
      }
      if (!std::isfinite(coefficient_scale)) {
        group_error[static_cast<std::size_t>(group_index)] = 3;
        continue;
      }
      for (int position = 0; position < width; ++position) {
        const double value =
            log_coefficient[static_cast<std::size_t>(position)];
        coefficient[static_cast<std::size_t>(position)] =
            std::isfinite(value) ? std::exp(value - coefficient_scale) : 0.0;
      }

      bool valid = true;
      for (std::int64_t row = group.begin; row < group.end; ++row) {
#ifdef GCFR_GFL_MPE_RKSS1
        const double log_x = row_log_x_sorted[row];
        const double x = row_x_sorted[row];
#else
        const double residual =
            observation_sorted[row] - static_cast<double>(group.centre);
        const double log_x = residual * inverse_sigma_squared;
        const double x = std::exp(log_x);
#endif
        double power = 1.0;
        double polynomial = 0.0;
        for (int position = 0; position < width; ++position) {
#ifdef GCFR_GFL_MPE_PPRS1
          power_path[static_cast<std::size_t>(position)] = power;
#endif
          polynomial +=
              coefficient[static_cast<std::size_t>(position)] * power;
          power *= x;
        }
        if (!(polynomial > 0.0) || !std::isfinite(polynomial)) {
          group_error[static_cast<std::size_t>(group_index)] = 4;
          valid = false;
          break;
        }
        const double log_polynomial = std::log(polynomial);
#ifdef GCFR_GFL_MPE_RKSS1
        const double half_square = row_half_square_sorted[row];
#else
        const double standardized_residual = residual / sigma;
        const double half_square =
            0.5 * standardized_residual * standardized_residual;
#endif
        const double density =
            coefficient_scale + static_cast<double>(group.d_min) * log_x +
            log_polynomial - half_square;
        const std::int64_t original_index = original_index_sorted[row];
        if (original_index < 0 || original_index >= row_count ||
            !std::isfinite(density)) {
          group_error[static_cast<std::size_t>(group_index)] = 5;
          valid = false;
          break;
        }
        row_log_density_original[original_index] = density;

        const double inverse_polynomial = 1.0 / polynomial;
#ifdef GCFR_GFL_MPE_PPRS1
        for (int position = 0; position < width; ++position) {
          moment[static_cast<std::size_t>(position)] +=
              power_path[static_cast<std::size_t>(position)] * inverse_polynomial;
        }
#else
        power = 1.0;
        for (int position = 0; position < width; ++position) {
          moment[static_cast<std::size_t>(position)] +=
              power * inverse_polynomial;
          power *= x;
        }
#endif
      }
      if (!valid) continue;

      for (int position = 0; position < width; ++position) {
        const double fibre_moment =
            moment[static_cast<std::size_t>(position)];
        if (!(fibre_moment > 0.0) || !std::isfinite(fibre_moment)) {
          group_error[static_cast<std::size_t>(group_index)] = 6;
          valid = false;
          break;
        }
        const double value =
            log_coefficient[static_cast<std::size_t>(position)] -
            coefficient_scale + std::log(fibre_moment);
        group_value[static_cast<std::size_t>(
            group.value_offset + position)] = value;
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(thread_count));
  for (int thread = 0; thread < thread_count; ++thread) {
    workers.emplace_back(worker);
  }
  for (std::thread& thread : workers) thread.join();

  std::vector<double> component_max(
      static_cast<std::size_t>(component_count), negative_infinity);
  std::vector<double> component_sum(
      static_cast<std::size_t>(component_count), 0.0);
  for (std::int64_t group_index = 0; group_index < group_count; ++group_index) {
    const int error = group_error[static_cast<std::size_t>(group_index)];
    if (error != 0) return error;
    const GroupRecord& group =
        groups[static_cast<std::size_t>(group_index)];
    for (int position = 0; position < group.width; ++position) {
      const double value = group_value[static_cast<std::size_t>(
          group.value_offset + position)];
      if (!std::isfinite(value)) continue;
      const int component = group.lower + position;
      double& old_max = component_max[static_cast<std::size_t>(component)];
      double& scaled_sum = component_sum[static_cast<std::size_t>(component)];
      if (!std::isfinite(old_max)) {
        old_max = value;
        scaled_sum = 1.0;
      } else if (value > old_max) {
        scaled_sum = scaled_sum * std::exp(old_max - value) + 1.0;
        old_max = value;
      } else {
        scaled_sum += std::exp(value - old_max);
      }
    }
  }

  for (int component = 0; component < component_count; ++component) {
    const double maximum =
        component_max[static_cast<std::size_t>(component)];
    const double sum = component_sum[static_cast<std::size_t>(component)];
    log_expected[component] =
        std::isfinite(maximum) && sum > 0.0
            ? maximum + std::log(sum)
            : negative_infinity;
  }
  return 0;
}
