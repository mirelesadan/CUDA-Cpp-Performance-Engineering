#pragma once

#include <cstddef>
#include <vector>

#include "adaptive_median_cuda.hpp"

// Benchmark-only API. It does not add a new application/Python ownership model.
namespace phase_b::detail {

struct AdaptiveTransferOneShotSample {
    double host_preparation_ms = 0.0;
    double device_allocation_ms = 0.0;
    double event_setup_ms = 0.0;
    double host_to_pinned_ms = 0.0;
    double host_to_device_ms = 0.0;
    double plane_minimum_ms = 0.0;
    double common_3x3_ms = 0.0;
    double fallback_ms = 0.0;
    double device_to_host_ms = 0.0;
    double pinned_to_host_ms = 0.0;
    double gpu_path_ms = 0.0;
    double device_free_ms = 0.0;
    double native_wall_ms = 0.0;
};

struct AdaptiveTransferPairSample {
    double pageable_h2d_ms = 0.0;
    double pinned_h2d_ms = 0.0;
    double pageable_d2h_ms = 0.0;
    double pinned_d2h_ms = 0.0;
};

struct AdaptiveResidentSample {
    double initial_h2d_ms = 0.0;
    double repeated_device_ms = 0.0;
    double final_d2h_ms = 0.0;
    double total_gpu_path_ms = 0.0;
};

struct AdaptiveResidentSeries {
    std::size_t repetition_count = 0;
    std::vector<AdaptiveResidentSample> samples;
};

struct AdaptiveTransferExperimentResult {
    std::vector<AdaptiveTransferOneShotSample> pageable_one_shot;
    std::vector<AdaptiveTransferOneShotSample> pinned_staging_one_shot;
    std::vector<AdaptiveTransferPairSample> transfer_pairs;
    std::vector<AdaptiveResidentSeries> resident;
    double pinned_buffer_allocation_ms = 0.0;
    double pinned_buffer_free_ms = 0.0;
    std::size_t input_device_bytes = 0;
    std::size_t output_device_bytes = 0;
    std::size_t plane_minima_device_bytes = 0;
    std::size_t fallback_flags_device_bytes = 0;
};

// All paths compare their final output bit for bit with expected before
// returning. A resident sequence repeatedly filters the same device input;
// outputs do not feed into later operations.
void validate_adaptive_cuda_transfer_modes(
    const std::vector<double>& input,
    const std::vector<double>& expected,
    const phase_a::Dimensions4D& dimensions);

AdaptiveTransferExperimentResult benchmark_adaptive_cuda_transfer_residency(
    const std::vector<double>& input,
    const std::vector<double>& expected,
    const phase_a::Dimensions4D& dimensions,
    std::size_t one_shot_runs,
    std::size_t transfer_pair_runs,
    std::size_t resident_sequence_runs,
    const std::vector<std::size_t>& resident_repetition_counts);

} // namespace phase_b::detail
