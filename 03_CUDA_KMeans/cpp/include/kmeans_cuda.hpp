#pragma once

#include "kmeans_serial.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace kmeans {

struct FitMetadata {
    std::size_t update_count = 0;
    bool converged = false;
};

// Supported fixed-shape, synchronous CUDA owner; uses the retained parallel
// count + tiled centroid pipeline. No host input is retained or read by fit().
// upload/fit/download require the construction device to be current. Not thread-safe.
// Do not reset its CUDA context while it is alive. Copy/move are disabled.
class CudaKMeansBuffer {
public:
    CudaKMeansBuffer(std::size_t n, std::size_t d, std::size_t k);
    ~CudaKMeansBuffer() noexcept;
    CudaKMeansBuffer(const CudaKMeansBuffer&) = delete;
    CudaKMeansBuffer& operator=(const CudaKMeansBuffer&) = delete;
    CudaKMeansBuffer(CudaKMeansBuffer&&) = delete;
    CudaKMeansBuffer& operator=(CudaKMeansBuffer&&) = delete;

    // Full finite/bounded input validation, then H2D. Invalid arguments leave
    // the previous state intact. Successful upload invalidates prior output.
    void upload(const std::vector<float>& input);
    // Independent frozen 100-update fit, reseeded from the uploaded snapshot.
    // Returns only metadata; final labels/centroids remain on the device.
    FitMetadata fit();
    // Validation-only counterpart for the frozen nonconvergence fixture.
    FitMetadata fit_with_update_cap(std::size_t max_updates);
    // Fresh independent host vectors; does not consume the resident output.
    Result download();
    std::size_t device_bytes() const noexcept;

    // CUDA-operation failures invalidate the owner; reconstruct it to recover.
    // Invalid lifecycle calls throw logic_error without changing its state.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

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

// Retained deterministic two-stage FP64 centroid reduction. Assignment
// and integer counting are the unchanged correctness-first kernels.
Result kmeans_cuda_tiled(const std::vector<float>& input, std::size_t n,
                         std::size_t d, std::size_t k);
Result kmeans_cuda_tiled_with_update_cap(const std::vector<float>& input,
                                         std::size_t n, std::size_t d,
                                         std::size_t k, std::size_t max_updates);
CudaRun kmeans_cuda_tiled_diagnostic(const std::vector<float>& input,
                                     std::size_t n, std::size_t d, std::size_t k,
                                     std::size_t max_updates = 100);

// Experiment-only direct update timing. Frozen first-assignment labels/counts,
// persistent device buffers/events, paired normal kernel launches; timings
// exclude allocation, copies, initialization, assignment, and counting.
struct CudaUpdatePairTimings {
    std::vector<double> control_ms;
    std::vector<double> tiled_ms;
    std::vector<double> partial_ms;
    std::vector<double> finalize_ms;
};
CudaUpdatePairTimings benchmark_cuda_update_pair(
    const std::vector<float>& input, std::size_t n, std::size_t d,
    std::size_t k, std::size_t rounds);

// Retained tiled centroid update with a separate two-stage integer count.
Result kmeans_cuda_parallel_count(const std::vector<float>& input, std::size_t n,
                                  std::size_t d, std::size_t k);
Result kmeans_cuda_parallel_count_with_update_cap(
    const std::vector<float>& input, std::size_t n, std::size_t d,
    std::size_t k, std::size_t max_updates);
CudaRun kmeans_cuda_parallel_count_diagnostic(
    const std::vector<float>& input, std::size_t n, std::size_t d,
    std::size_t k, std::size_t max_updates = 100);

// Validation only: compare both count kernels on the actual labels used by
// every Lloyd update. This path is never used for timing.
struct CudaCountValidation {
    Result result;
    std::size_t updates_checked = 0;
    std::size_t count_values_checked = 0;
};
CudaCountValidation kmeans_cuda_parallel_count_checked(
    const std::vector<float>& input, std::size_t n, std::size_t d,
    std::size_t k, std::size_t max_updates = 100);

struct CudaCountPairTimings {
    std::vector<std::int32_t> control_counts;
    std::vector<std::int32_t> parallel_counts;
    std::vector<double> control_ms;
    std::vector<double> parallel_ms;
    // Separate component runs: their extra midpoint event is excluded from
    // the primary two-event, two-kernel parallel_ms measurement.
    std::vector<double> partial_ms;
    std::vector<double> finalize_ms;
};
// Persistent-buffer direct count experiment on fixed labels. rounds=0 only
// validates exact counts/repeatability; timed experiments require 7..50.
CudaCountPairTimings benchmark_cuda_count_pair(
    const std::vector<std::int32_t>& labels, std::size_t k, std::size_t rounds);

}  // namespace kmeans
