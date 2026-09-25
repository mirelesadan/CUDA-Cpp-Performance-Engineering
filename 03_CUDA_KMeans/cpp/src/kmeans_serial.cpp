#include "kmeans_serial.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef KMEANS_PHASE_TIMING
#include <chrono>
#endif

namespace kmeans {
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
thread_local PhaseTimings recorded_phases;

double elapsed_ms(ProfileClock::time_point start, ProfileClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
#endif

void validate(const std::vector<float>& input, std::size_t n,
              std::size_t d, std::size_t k, std::size_t max_updates) {
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
    for (const float value : input) {
        if (!std::isfinite(value) || std::fabs(value) > max_magnitude) {
            throw std::invalid_argument("input must be finite with absolute coordinates <= 1024");
        }
    }
}

std::vector<std::int32_t> assign(const std::vector<float>& input,
                                 const std::vector<float>& centroids,
                                 std::size_t n, std::size_t d, std::size_t k) {
    std::vector<std::int32_t> labels(n);
    for (std::size_t sample = 0; sample < n; ++sample) {
        float best_distance = std::numeric_limits<float>::infinity();
        std::int32_t best_cluster = 0;
        for (std::size_t cluster = 0; cluster < k; ++cluster) {
            float distance = 0.0f;
            for (std::size_t feature = 0; feature < d; ++feature) {
                // Separate float operations in ascending feature order match
                // the NumPy reference. /fp:precise prevents contraction.
                const float difference =
                    input[sample * d + feature] - centroids[cluster * d + feature];
                const float square = difference * difference;
                distance = distance + square;
            }
            // Strict less-than preserves the first (lowest-index) exact tie.
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = static_cast<std::int32_t>(cluster);
            }
        }
        labels[sample] = best_cluster;
    }
    return labels;
}

std::vector<std::int32_t> assign_addressed(const std::vector<float>& input,
                                           const std::vector<float>& centroids,
                                           std::size_t n, std::size_t d,
                                           std::size_t k) {
    std::vector<std::int32_t> labels(n);
    for (std::size_t sample = 0; sample < n; ++sample) {
        const float* const sample_row = input.data() + sample * d;
        float best_distance = std::numeric_limits<float>::infinity();
        std::int32_t best_cluster = 0;
        for (std::size_t cluster = 0; cluster < k; ++cluster) {
            const float* const centroid_row = centroids.data() + cluster * d;
            float distance = 0.0f;
            for (std::size_t feature = 0; feature < d; ++feature) {
                // Only row addressing changes: identical ordered FP32 operations.
                const float difference = sample_row[feature] - centroid_row[feature];
                const float square = difference * difference;
                distance = distance + square;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = static_cast<std::int32_t>(cluster);
            }
        }
        labels[sample] = best_cluster;
    }
    return labels;
}

std::vector<float> update(const std::vector<float>& input,
                          const std::vector<std::int32_t>& labels,
                          const std::vector<float>& previous,
                          std::size_t n, std::size_t d, std::size_t k) {
    std::vector<std::size_t> counts(k, 0);
    std::vector<double> sums(k * d, 0.0);
    for (std::size_t sample = 0; sample < n; ++sample) {
        const auto cluster = static_cast<std::size_t>(labels[sample]);
        ++counts[cluster];
        for (std::size_t feature = 0; feature < d; ++feature) {
            const auto position = cluster * d + feature;
            sums[position] = sums[position] +
                             static_cast<double>(input[sample * d + feature]);
        }
    }
    auto centroids = previous;  // Empty clusters retain exact float bits.
    for (std::size_t cluster = 0; cluster < k; ++cluster) {
        if (counts[cluster] == 0) {
            continue;
        }
        for (std::size_t feature = 0; feature < d; ++feature) {
            const auto position = cluster * d + feature;
            const double mean = sums[position] / static_cast<double>(counts[cluster]);
            centroids[position] = static_cast<float>(mean);
        }
    }
    return centroids;
}

}  // namespace

#ifdef KMEANS_PHASE_TIMING
PhaseTimings last_phase_timings() {
    return recorded_phases;
}
#endif

std::vector<std::size_t> initial_row_indices(std::size_t n, std::size_t k) {
    if (n < 2 || n > max_n || k < 2 || k > max_k || k > n) {
        throw std::invalid_argument("initial rows require 2 <= K <= min(N,32), N <= 2^20");
    }
    std::vector<std::size_t> rows(k);
    for (std::size_t cluster = 0; cluster < k; ++cluster) {
        rows[cluster] = ((2 * cluster + 1) * n) / (2 * k);
    }
    return rows;
}

Result kmeans_serial_with_update_cap(const std::vector<float>& input,
                                     std::size_t n, std::size_t d,
                                     std::size_t k, std::size_t max_updates) {
#ifdef KMEANS_PHASE_TIMING
    PhaseTimings phases;
    auto phase_start = ProfileClock::now();
#endif
    validate(input, n, d, k, max_updates);
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
    auto labels = assign(input, centroids, n, d, k);
#ifdef KMEANS_PHASE_TIMING
    phases.initial_assignment_ms += elapsed_ms(phase_start, ProfileClock::now());
    phases.assignment_passes = 1;
#endif
    for (std::size_t pass = 1; pass <= max_updates; ++pass) {
#ifdef KMEANS_PHASE_TIMING
        phase_start = ProfileClock::now();
#endif
        auto next_centroids = update(input, labels, centroids, n, d, k);
#ifdef KMEANS_PHASE_TIMING
        phases.centroid_update_ms += elapsed_ms(phase_start, ProfileClock::now());
        ++phases.centroid_updates;
        phase_start = ProfileClock::now();
#endif
        auto next_labels = assign(input, next_centroids, n, d, k);
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
            recorded_phases = phases;
#endif
            return {std::move(labels), std::move(centroids), pass, true};
        }
    }
#ifdef KMEANS_PHASE_TIMING
    recorded_phases = phases;
#endif
    return {std::move(labels), std::move(centroids), max_updates, false};
}

Result kmeans_serial(const std::vector<float>& input, std::size_t n,
                     std::size_t d, std::size_t k) {
    return kmeans_serial_with_update_cap(input, n, d, k, 100);
}

Result kmeans_serial_addressed_with_update_cap(const std::vector<float>& input,
                                               std::size_t n, std::size_t d,
                                               std::size_t k,
                                               std::size_t max_updates) {
    validate(input, n, d, k, max_updates);
    std::vector<float> centroids(k * d);
    const auto seed_rows = initial_row_indices(n, k);
    for (std::size_t cluster = 0; cluster < k; ++cluster) {
        for (std::size_t feature = 0; feature < d; ++feature) {
            centroids[cluster * d + feature] = input[seed_rows[cluster] * d + feature];
        }
    }
    auto labels = assign_addressed(input, centroids, n, d, k);
    for (std::size_t pass = 1; pass <= max_updates; ++pass) {
        auto next_centroids = update(input, labels, centroids, n, d, k);
        auto next_labels = assign_addressed(input, next_centroids, n, d, k);
        const bool converged = next_labels == labels;
        centroids = std::move(next_centroids);
        labels = std::move(next_labels);
        if (converged) {
            return {std::move(labels), std::move(centroids), pass, true};
        }
    }
    return {std::move(labels), std::move(centroids), max_updates, false};
}

Result kmeans_serial_addressed(const std::vector<float>& input, std::size_t n,
                               std::size_t d, std::size_t k) {
    return kmeans_serial_addressed_with_update_cap(input, n, d, k, 100);
}

}  // namespace kmeans
