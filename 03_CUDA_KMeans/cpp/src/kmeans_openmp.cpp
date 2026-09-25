#include "kmeans_serial.hpp"

#include <omp.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef KMEANS_PHASE_TIMING
#include <chrono>
#endif

namespace kmeans {
namespace detail {
std::vector<float> update_serial(const std::vector<float>& input,
                                 const std::vector<std::int32_t>& labels,
                                 const std::vector<float>& previous,
                                 std::size_t n, std::size_t d, std::size_t k);
}  // namespace detail
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Project 2 requires IEEE-754 binary32 float");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "Project 2 requires IEEE-754 binary64 double");

constexpr std::size_t max_n = 1u << 20;
constexpr std::size_t max_d = 32;
constexpr std::size_t max_k = 32;
constexpr float max_magnitude = 1024.0f;

#ifdef KMEANS_PHASE_TIMING
using ProfileClock = std::chrono::steady_clock;

double elapsed_ms(ProfileClock::time_point start, ProfileClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
#endif

void validate(const std::vector<float>& input, std::size_t n,
              std::size_t d, std::size_t k, std::size_t max_updates,
              int thread_count) {
    if (n < 2 || n > max_n || d < 1 || d > max_d ||
        k < 2 || k > max_k || k > n) {
        throw std::invalid_argument("require 2 <= K <= min(N,32), K <= N <= 2^20, 1 <= D <= 32");
    }
    if (input.size() != n * d) {
        throw std::invalid_argument("input size must equal N * D");
    }
    if (max_updates < 1 || max_updates > 100) {
        throw std::invalid_argument("update cap must be between 1 and 100");
    }
    if (thread_count < 1 || thread_count > omp_get_num_procs()) {
        throw std::invalid_argument("thread count must be between 1 and available OpenMP processors");
    }
    for (const float value : input) {
        if (!std::isfinite(value) || std::fabs(value) > max_magnitude) {
            throw std::invalid_argument("input must be finite with absolute coordinates <= 1024");
        }
    }
}

// An explicit team is part of this experiment's API. Restore the caller's
// dynamic-team setting on every return, including exceptions.
class DynamicTeamGuard {
public:
    DynamicTeamGuard() : previous_(omp_get_dynamic()) { omp_set_dynamic(0); }
    ~DynamicTeamGuard() { omp_set_dynamic(previous_); }
    DynamicTeamGuard(const DynamicTeamGuard&) = delete;
    DynamicTeamGuard& operator=(const DynamicTeamGuard&) = delete;

private:
    int previous_;
};

std::vector<std::int32_t> assign_openmp(const std::vector<float>& input,
                                        const std::vector<float>& centroids,
                                        std::size_t n, std::size_t d,
                                        std::size_t k, int thread_count) {
    std::vector<std::int32_t> labels(n);
    const float* const input_values = input.data();
    const float* const centroid_values = centroids.data();
    int actual_threads = 0;

#pragma omp parallel num_threads(thread_count)
    {
        if (omp_get_thread_num() == 0) {
            actual_threads = omp_get_num_threads();
        }
#pragma omp for schedule(static)
        for (int sample = 0; sample < static_cast<int>(n); ++sample) {
            const std::size_t sample_index = static_cast<std::size_t>(sample);
            const float* const sample_row = input_values + sample_index * d;
            float best_distance = std::numeric_limits<float>::infinity();
            std::int32_t best_cluster = 0;
            for (std::size_t cluster = 0; cluster < k; ++cluster) {
                const float* const centroid_row = centroid_values + cluster * d;
                float distance = 0.0f;
                for (std::size_t feature = 0; feature < d; ++feature) {
                    // Identical ordered FP32 operations to assign_addressed().
                    const float difference = sample_row[feature] - centroid_row[feature];
                    const float square = difference * difference;
                    distance = distance + square;
                }
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = static_cast<std::int32_t>(cluster);
                }
            }
            labels[sample_index] = best_cluster;
        }
    }
    if (actual_threads != thread_count) {
        throw std::runtime_error("OpenMP runtime did not provide the requested thread count");
    }
    return labels;
}

}  // namespace

#ifdef KMEANS_PHASE_TIMING
// Defined by the separately linked phase-timing serial control library.
void record_phase_timings(const PhaseTimings& phases);
#endif

Result kmeans_openmp_with_update_cap(const std::vector<float>& input,
                                     std::size_t n, std::size_t d,
                                     std::size_t k, std::size_t max_updates,
                                     int thread_count) {
#ifdef KMEANS_PHASE_TIMING
    PhaseTimings phases;
    auto phase_start = ProfileClock::now();
#endif
    validate(input, n, d, k, max_updates, thread_count);
    DynamicTeamGuard dynamic_team_guard;
#ifdef KMEANS_PHASE_TIMING
    phases.validation_ms += elapsed_ms(phase_start, ProfileClock::now());
    phase_start = ProfileClock::now();
#endif
    std::vector<float> centroids(k * d);
    const auto seed_rows = initial_row_indices(n, k);
    for (std::size_t cluster = 0; cluster < k; ++cluster) {
        for (std::size_t feature = 0; feature < d; ++feature) {
            centroids[cluster * d + feature] = input[seed_rows[cluster] * d + feature];
        }
    }
#ifdef KMEANS_PHASE_TIMING
    phases.initialization_ms += elapsed_ms(phase_start, ProfileClock::now());
    phase_start = ProfileClock::now();
#endif
    auto labels = assign_openmp(input, centroids, n, d, k, thread_count);
#ifdef KMEANS_PHASE_TIMING
    phases.initial_assignment_ms += elapsed_ms(phase_start, ProfileClock::now());
    phases.assignment_passes = 1;
#endif
    for (std::size_t pass = 1; pass <= max_updates; ++pass) {
#ifdef KMEANS_PHASE_TIMING
        phase_start = ProfileClock::now();
#endif
        auto next_centroids = detail::update_serial(input, labels, centroids, n, d, k);
#ifdef KMEANS_PHASE_TIMING
        phases.centroid_update_ms += elapsed_ms(phase_start, ProfileClock::now());
        ++phases.centroid_updates;
        phase_start = ProfileClock::now();
#endif
        auto next_labels = assign_openmp(input, next_centroids, n, d, k, thread_count);
#ifdef KMEANS_PHASE_TIMING
        phases.reassignment_ms += elapsed_ms(phase_start, ProfileClock::now());
        ++phases.assignment_passes;
        phase_start = ProfileClock::now();
#endif
        const bool converged = next_labels == labels;
#ifdef KMEANS_PHASE_TIMING
        phases.convergence_check_ms += elapsed_ms(phase_start, ProfileClock::now());
#endif
        centroids = std::move(next_centroids);
        labels = std::move(next_labels);
        if (converged) {
#ifdef KMEANS_PHASE_TIMING
            record_phase_timings(phases);
#endif
            return {std::move(labels), std::move(centroids), pass, true};
        }
    }
#ifdef KMEANS_PHASE_TIMING
    record_phase_timings(phases);
#endif
    return {std::move(labels), std::move(centroids), max_updates, false};
}

Result kmeans_openmp(const std::vector<float>& input, std::size_t n,
                     std::size_t d, std::size_t k, int thread_count) {
    return kmeans_openmp_with_update_cap(input, n, d, k, 100, thread_count);
}

}  // namespace kmeans
