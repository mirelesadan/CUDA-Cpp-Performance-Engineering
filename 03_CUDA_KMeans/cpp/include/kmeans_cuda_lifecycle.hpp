#pragma once

// Internal characterization support, not a supported user-facing owner/API.
// Available only with PROJECT2_ENABLE_LIFECYCLE_BENCHMARK.
#include "kmeans_serial.hpp"
#include <array>
#include <string>
#include <vector>

namespace kmeans::benchmark {

enum Stage : std::size_t {
    validation, allocation, h2d, initialization, initial_assignment, count,
    centroid_update, flag_reset, reassignment, flag_d2h, output_allocation,
    labels_d2h, centroids_d2h, free, stage_count
};
inline constexpr const char* stage_names[] = {
    "validation", "allocation", "h2d", "initialization", "initial_assignment",
    "count", "centroid_update", "flag_reset", "reassignment", "flag_d2h",
    "output_allocation", "labels_d2h", "centroids_d2h", "free"
};
inline constexpr const char* buffer_names[] = {
    "input", "centroids", "labels_a", "labels_b", "counts", "change_flag",
    "centroid_partials", "count_partials"
};

struct LifecycleSample {
    std::string mode;
    std::string instrumentation;
    std::size_t fits = 1;
    double wall_ms = 0.0;
    double fit_wall_ms = 0.0;
    double event_collection_ms = 0.0;
    std::array<double, stage_count> host_ms{};
    std::array<double, stage_count> event_ms{};
};

struct LifecycleReport {
    Result result;
    std::size_t checked_fits = 0;
    std::array<std::size_t, 8> device_bytes{};
    double diagnostic_event_setup_ms = 0.0;
    double diagnostic_event_destroy_ms = 0.0;
    std::vector<LifecycleSample> samples;
};

// runs=sequences=0: correctness only. Otherwise runs>=7, sequences>=5.
// A_native times the unchanged retained API. A_shadow is a lifecycle-equivalent
// diagnostic; B/C reuse allocation; D includes the entire amortized lifecycle.
// All resident fits restart from frozen seeds. Event diagnostics are separate
// from authoritative uninstrumented wall timings.
LifecycleReport characterize_cuda_lifecycle(
    const std::vector<float>& input, std::size_t n, std::size_t d,
    std::size_t k, std::size_t max_updates = 100,
    std::size_t runs = 7, std::size_t sequences = 5);

}  // namespace kmeans::benchmark
