#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fixed_median.hpp"

namespace phase_a {

struct CudaDeviceInfo {
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessor_count = 0;
    std::size_t global_memory_bytes = 0;
};

struct CudaTimingMilliseconds {
    double host_to_device = 0.0;
    double kernel = 0.0;
    double device_to_host = 0.0;
    double total_gpu_path = 0.0;
};

struct CudaFilterResult {
    std::vector<double> output;
    CudaTimingMilliseconds timing;
};

struct CudaKernelTimingSequence {
    std::vector<double> output;
    std::vector<double> warmup_kernel_milliseconds;
    std::vector<double> steady_kernel_milliseconds;
    std::vector<double> post_idle_kernel_milliseconds;
};

struct CudaPageablePathBenchmarkResult {
    std::vector<double> output;
    std::vector<CudaTimingMilliseconds> timed_runs;
};

struct CudaResidencyBenchmarkResult {
    std::size_t iteration_count = 0;
    std::vector<CudaTimingMilliseconds> timed_runs;
};

struct CudaTransferCharacterizationResult {
    std::vector<double> output;
    std::vector<CudaTimingMilliseconds> pageable_path_runs;
    std::vector<CudaResidencyBenchmarkResult> residency_results;
    std::vector<double> diagnostic_pageable_h2d_milliseconds;
    std::vector<double> diagnostic_pageable_d2h_milliseconds;
    std::vector<double> diagnostic_pinned_h2d_milliseconds;
    std::vector<double> diagnostic_pinned_d2h_milliseconds;
    std::string pinned_memory_error;
};

class CudaMedianBuffer {
public:
    CudaMedianBuffer(const double* host_input, const Dimensions4D& dimensions);
    ~CudaMedianBuffer();

    CudaMedianBuffer(const CudaMedianBuffer&) = delete;
    CudaMedianBuffer& operator=(const CudaMedianBuffer&) = delete;
    CudaMedianBuffer(CudaMedianBuffer&&) = delete;
    CudaMedianBuffer& operator=(CudaMedianBuffer&&) = delete;

    void filter();
    void download(double* host_output) const;
    Dimensions4D dimensions() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

CudaDeviceInfo cuda_device_info();

CudaFilterResult fixed_median_3x3_cuda_baseline(
    const std::vector<double>& input,
    const Dimensions4D& dimensions);

CudaKernelTimingSequence benchmark_fixed_median_3x3_cuda_baseline_kernel(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    std::size_t warmup_launch_count,
    std::size_t timed_launch_count,
    unsigned int idle_milliseconds,
    std::size_t post_idle_launch_count);

CudaPageablePathBenchmarkResult benchmark_fixed_median_3x3_cuda_pageable_path(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    std::size_t warmup_run_count,
    std::size_t timed_run_count);

CudaTransferCharacterizationResult characterize_fixed_median_3x3_cuda_transfers(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    std::size_t pageable_warmup_count,
    std::size_t pageable_timed_count,
    const std::vector<std::size_t>& residency_iteration_counts,
    std::size_t residency_warmup_count,
    std::size_t residency_timed_count,
    std::size_t transfer_diagnostic_warmup_count,
    std::size_t transfer_diagnostic_timed_count);

} // namespace phase_a
