#pragma once

#include <cstddef>
#include <vector>

#include "adaptive_median.hpp"

namespace phase_b {

struct AdaptiveCudaTimingMilliseconds {
    double host_to_device = 0.0;
    double plane_minimum_kernel = 0.0;
    double adaptive_filter_kernel = 0.0;
    double device_to_host = 0.0;
    double total_gpu_path = 0.0;
    double native_wall = 0.0;
};

struct AdaptiveCudaResult {
    std::vector<double> output;
    AdaptiveCudaTimingMilliseconds timing;
};

struct AdaptiveCudaKernelConfiguration {
    unsigned int threads_per_block = 0;
    unsigned int plane_minimum_blocks = 0;
    unsigned int adaptive_filter_blocks = 0;
};

struct AdaptiveCudaKernelBenchmarkResult {
    std::vector<double> output;
    std::vector<double> plane_minimum_kernel_milliseconds;
    std::vector<double> adaptive_filter_kernel_milliseconds;
};

struct AdaptiveCudaSplitKernelBenchmarkResult {
    std::vector<double> monolithic_output;
    std::vector<double> split_output;
    std::vector<double> monolithic_adaptive_milliseconds;
    std::vector<double> common_3x3_milliseconds;
    std::vector<double> fallback_milliseconds;
    std::vector<double> split_combined_milliseconds;
};

// Correctness-first one-shot CUDA baseline. Device allocation is intentionally
// internal; total_gpu_path spans pageable H2D, both kernels, and pageable D2H,
// while native_wall also includes validation, allocation, events, and cleanup.
AdaptiveCudaResult adaptive_median_s3_smax7_cuda_baseline(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

// One-shot form of the retained split candidate. Its adaptive_filter_kernel
// timing spans the common and fallback launches together.
AdaptiveCudaResult adaptive_median_s3_smax7_cuda_split(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

// Executes the same numerical path and collects branch outcomes separately
// from benchmark timing.
AdaptiveMedianDiagnosticResult adaptive_median_s3_smax7_cuda_baseline_diagnostics(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

// Retained split path: the common kernel uses only a nine-value window,
// while a dense byte flag selects the rare outputs revisited by the unchanged
// 5x5/7x7 fallback logic.
AdaptiveMedianDiagnosticResult adaptive_median_s3_smax7_cuda_split_diagnostics(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

// Benchmark-only stable kernel sequence. Buffers and events are reused within
// this call; this does not expose persistent device ownership to applications.
AdaptiveCudaKernelBenchmarkResult benchmark_adaptive_median_s3_smax7_cuda_kernels(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions,
    std::size_t warmup_launch_count,
    std::size_t timed_launch_count);

AdaptiveCudaSplitKernelBenchmarkResult
benchmark_adaptive_median_s3_smax7_cuda_split_kernels(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions,
    std::size_t warmup_launch_count,
    std::size_t timed_launch_count);

AdaptiveCudaKernelConfiguration adaptive_median_cuda_kernel_configuration(
    const phase_a::Dimensions4D& dimensions);

} // namespace phase_b
