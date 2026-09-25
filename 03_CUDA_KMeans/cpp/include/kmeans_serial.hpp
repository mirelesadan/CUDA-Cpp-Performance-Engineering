#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kmeans {

struct Result {
    std::vector<std::int32_t> labels;
    std::vector<float> centroids;
    std::size_t update_count;
    bool converged;
};

#ifdef KMEANS_PHASE_TIMING
// Available only in the separate profiling build; normal Release has no timers.
struct PhaseTimings {
    double validation_ms = 0.0;
    double initialization_ms = 0.0;
    double initial_assignment_ms = 0.0;
    double centroid_update_ms = 0.0;
    double reassignment_ms = 0.0;
    double convergence_check_ms = 0.0;
    std::size_t assignment_passes = 0;
    std::size_t centroid_updates = 0;
};

PhaseTimings last_phase_timings();
#endif

// Deterministic input-row initialization from the frozen Project 2 contract.
std::vector<std::size_t> initial_row_indices(std::size_t n, std::size_t k);

// Normal public baseline: dense row-major input with logical shape (N,D),
// one initial assignment, then at most 100 update/reassignment passes.
Result kmeans_serial(const std::vector<float>& input, std::size_t n,
                     std::size_t d, std::size_t k);

// Diagnostic hook for the public one-pass nonconvergence fixture only.
// It does not change the 100-pass contract of kmeans_serial().
Result kmeans_serial_with_update_cap(const std::vector<float>& input,
                                     std::size_t n, std::size_t d,
                                     std::size_t k, std::size_t max_updates);

// Isolated address-calculation candidate. The original serial implementation
// above remains the control for exact-result and same-session comparisons.
Result kmeans_serial_addressed(const std::vector<float>& input, std::size_t n,
                               std::size_t d, std::size_t k);

// Test-only reduced-cap equivalent for the frozen nonconvergence fixture.
Result kmeans_serial_addressed_with_update_cap(const std::vector<float>& input,
                                               std::size_t n, std::size_t d,
                                               std::size_t k,
                                               std::size_t max_updates);

// Available when PROJECT2_ENABLE_OPENMP is enabled. Each assignment pass
// divides independent sample rows among an explicit number of OpenMP threads;
// centroid accumulation and convergence control remain serial and ordered.
Result kmeans_openmp(const std::vector<float>& input, std::size_t n,
                     std::size_t d, std::size_t k, int thread_count);

// Test-only reduced-cap equivalent for the frozen nonconvergence fixture.
Result kmeans_openmp_with_update_cap(const std::vector<float>& input,
                                     std::size_t n, std::size_t d,
                                     std::size_t k, std::size_t max_updates,
                                     int thread_count);

}  // namespace kmeans
