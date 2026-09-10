#pragma once

#include <vector>

#include "fixed_median.hpp"

namespace phase_b {

// Correctness-first implementation of the established Phase B contract:
// adaptive scan-space windows of 3x3, 5x5, and 7x7 with per-plane
// global-minimum padding. This intentionally favors traceability over speed.
std::vector<double> adaptive_median_s3_smax7(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

} // namespace phase_b
