#pragma once

#include <vector>

#include "fixed_median.hpp"

namespace phase_b {

struct AdaptiveMedianStatistics {
    std::size_t total_outputs = 0;
    std::size_t finished_at_3x3 = 0;
    std::size_t expanded_to_5x5 = 0;
    std::size_t finished_at_5x5 = 0;
    std::size_t expanded_to_7x7 = 0;
    std::size_t finished_at_7x7 = 0;
    std::size_t maximum_window_fallback = 0;
    std::size_t stage_b_retained_center = 0;
    std::size_t stage_b_replaced_with_median = 0;
    std::size_t median_computations = 0;
};

struct AdaptiveMedianDiagnosticResult {
    std::vector<double> output;
    AdaptiveMedianStatistics statistics;
};

// Correctness-first implementation of the established Phase B contract:
// adaptive scan-space windows of 3x3, 5x5, and 7x7 with per-plane
// global-minimum padding. This intentionally favors traceability over speed.
std::vector<double> adaptive_median_s3_smax7(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

// Runs the identical numerical path while collecting branch counts. Keep this
// separate from benchmark timing because the counters add diagnostic overhead.
AdaptiveMedianDiagnosticResult adaptive_median_s3_smax7_diagnostics(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

} // namespace phase_b
