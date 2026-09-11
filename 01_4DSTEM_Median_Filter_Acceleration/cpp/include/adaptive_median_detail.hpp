#pragma once

#include "adaptive_median.hpp"

namespace phase_b::detail {

// Internal, independently tested primitive used only by the specialized
// three-by-three adaptive path. The nine input values are reordered in place.
double median_of_nine_in_place(double* values);

// Internal A/B experiment entry points. These preserve the public adaptive
// API while allowing the benchmark and validation targets to compare direct
// padded-row 3x3 gathering with the retained specialized implementation.
std::vector<double> adaptive_median_s3_smax7_direct_3x3_gather(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

AdaptiveMedianDiagnosticResult adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions);

} // namespace phase_b::detail
