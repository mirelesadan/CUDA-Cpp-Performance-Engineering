#pragma once

namespace phase_b::detail {

// Internal, independently tested primitive used only by the specialized
// three-by-three adaptive path. The nine input values are reordered in place.
double median_of_nine_in_place(double* values);

} // namespace phase_b::detail
