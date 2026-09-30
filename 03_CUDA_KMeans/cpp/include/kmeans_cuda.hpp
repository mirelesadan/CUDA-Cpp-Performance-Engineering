#pragma once

#include "kmeans_serial.hpp"

#include <cstddef>
#include <vector>

namespace kmeans {

// GPU-stage fields are cumulative CUDA-event milliseconds; setup_ms and
// native_wall_ms are host steady-clock measurements. Device algorithm
// sums initialization, initial assignment, count/update, reassignment, and
// device flag reset; the tiny convergence-flag D2H is excluded. One-shot GPU
// adds H2D, convergence-flag D2H, and final D2H, excluding allocation.
// Native wall uses host steady_clock across the complete call.
struct CudaStageTimings {
    double setup_ms = 0.0;
    double h2d_ms = 0.0;
    double initialization_ms = 0.0;
    double initial_assignment_ms = 0.0;
    double count_ms = 0.0;
    double centroid_update_ms = 0.0;
    double reassignment_ms = 0.0;
    double convergence_ms = 0.0;
    double final_d2h_ms = 0.0;
    double device_algorithm_ms = 0.0;
    double one_shot_gpu_ms = 0.0;
    double native_wall_ms = 0.0;
};

struct CudaRun {
    Result result;
    CudaStageTimings timings;
};

// Full frozen 100-update Lloyd contract. CUDA support is opt-in at build time.
Result kmeans_cuda(const std::vector<float>& input, std::size_t n,
                   std::size_t d, std::size_t k);

// Reduced-cap counterpart for the public nonconvergence fixture.
Result kmeans_cuda_with_update_cap(const std::vector<float>& input,
                                   std::size_t n, std::size_t d,
                                   std::size_t k, std::size_t max_updates);

// Diagnostic one-shot path. Normal calls above create no CUDA events.
CudaRun kmeans_cuda_diagnostic(const std::vector<float>& input, std::size_t n,
                               std::size_t d, std::size_t k,
                               std::size_t max_updates = 100);

}  // namespace kmeans
