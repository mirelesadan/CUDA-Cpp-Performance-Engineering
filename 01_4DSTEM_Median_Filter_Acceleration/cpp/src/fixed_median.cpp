#include "fixed_median.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <omp.h>
#include <stdexcept>

namespace phase_a {
namespace {

std::size_t checked_element_count(const Dimensions4D& dimensions)
{
    const std::array<std::size_t, 4> sizes = {
        dimensions.scan_y,
        dimensions.scan_x,
        dimensions.detector_y,
        dimensions.detector_x,
    };

    std::size_t count = 1;
    for (const std::size_t size : sizes) {
        if (size == 0) {
            throw std::invalid_argument("Fixed median requires four nonempty dimensions.");
        }
        if (count > std::numeric_limits<std::size_t>::max() / size) {
            throw std::overflow_error("The 4D shape product exceeds std::size_t.");
        }
        count *= size;
    }

    return count;
}

void compare_swap(double& left, double& right)
{
    if (right < left) {
        std::swap(left, right);
    }
}

double median_of_nine(std::array<double, 9>& values)
{
    // A fixed 19-comparator selection network places the fifth ordered value
    // at index 4. It does not fully sort the other eight values.
    compare_swap(values[1], values[2]);
    compare_swap(values[4], values[5]);
    compare_swap(values[7], values[8]);
    compare_swap(values[0], values[1]);
    compare_swap(values[3], values[4]);
    compare_swap(values[6], values[7]);
    compare_swap(values[1], values[2]);
    compare_swap(values[4], values[5]);
    compare_swap(values[7], values[8]);
    compare_swap(values[0], values[3]);
    compare_swap(values[5], values[8]);
    compare_swap(values[4], values[7]);
    compare_swap(values[3], values[6]);
    compare_swap(values[1], values[4]);
    compare_swap(values[2], values[5]);
    compare_swap(values[4], values[7]);
    compare_swap(values[4], values[2]);
    compare_swap(values[6], values[4]);
    compare_swap(values[4], values[2]);

    return values[4];
}

struct NthElementMedian {
    double operator()(std::array<double, 9>& values) const
    {
        std::nth_element(values.begin(), values.begin() + 4, values.end());
        return values[4];
    }
};

struct MedianOfNine {
    double operator()(std::array<double, 9>& values) const
    {
        return median_of_nine(values);
    }
};

} // namespace

std::size_t flat_index(
    const Dimensions4D& dimensions,
    std::size_t scan_y_index,
    std::size_t scan_x_index,
    std::size_t detector_y_index,
    std::size_t detector_x_index)
{
    if (scan_y_index >= dimensions.scan_y || scan_x_index >= dimensions.scan_x ||
        detector_y_index >= dimensions.detector_y || detector_x_index >= dimensions.detector_x) {
        throw std::out_of_range("4D coordinate is outside the array shape.");
    }

    // NumPy C order stores the last axis contiguously, so detector_x changes fastest.
    return (((scan_y_index * dimensions.scan_x + scan_x_index) * dimensions.detector_y + detector_y_index) *
            dimensions.detector_x + detector_x_index);
}

std::size_t reflect_index(std::ptrdiff_t index, std::size_t size)
{
    if (size == 0) {
        throw std::invalid_argument("Cannot reflect an index into an empty dimension.");
    }
    if (size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("Dimension is too large for signed reflected indexing.");
    }

    const std::ptrdiff_t extent = static_cast<std::ptrdiff_t>(size);

    // Half-sample symmetry repeats the edge value:
    // -1 becomes 0, and size becomes size - 1. Repeating these steps also
    // handles indices farther outside the array without special cases.
    while (index < 0 || index >= extent) {
        if (index < 0) {
            index = -(index + 1);
        }
        else {
            index = extent - 1 - (index - extent);
        }
    }

    return static_cast<std::size_t>(index);
}

namespace {

template <typename MedianSelector>
std::vector<double> fixed_median_3x3_impl(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    MedianSelector select_median)
{
    const std::size_t element_count = checked_element_count(dimensions);
    if (input.size() != element_count) {
        throw std::invalid_argument("Input element count does not match the supplied 4D dimensions.");
    }
    if (dimensions.scan_y > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        dimensions.scan_x > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("Scan dimensions are too large for reflected indexing.");
    }

    // A distinct output vector keeps the input immutable, like creating a new NumPy array.
    std::vector<double> output(element_count);

    for (std::size_t scan_y = 0; scan_y < dimensions.scan_y; ++scan_y) {
        for (std::size_t scan_x = 0; scan_x < dimensions.scan_x; ++scan_x) {
            for (std::size_t detector_y = 0; detector_y < dimensions.detector_y; ++detector_y) {
                for (std::size_t detector_x = 0; detector_x < dimensions.detector_x; ++detector_x) {
                    std::array<double, 9> neighborhood{};
                    std::size_t neighborhood_index = 0;

                    for (std::ptrdiff_t offset_y = -1; offset_y <= 1; ++offset_y) {
                        const std::size_t reflected_y = reflect_index(
                            static_cast<std::ptrdiff_t>(scan_y) + offset_y,
                            dimensions.scan_y);

                        for (std::ptrdiff_t offset_x = -1; offset_x <= 1; ++offset_x) {
                            const std::size_t reflected_x = reflect_index(
                                static_cast<std::ptrdiff_t>(scan_x) + offset_x,
                                dimensions.scan_x);

                            const std::size_t input_index = flat_index(
                                dimensions,
                                reflected_y,
                                reflected_x,
                                detector_y,
                                detector_x);
                            neighborhood[neighborhood_index] = input[input_index];
                            ++neighborhood_index;
                        }
                    }

                    const std::size_t output_index = flat_index(
                        dimensions,
                        scan_y,
                        scan_x,
                        detector_y,
                        detector_x);
                    output[output_index] = select_median(neighborhood);
                }
            }
        }
    }

    return output;
}

} // namespace

std::vector<double> fixed_median_3x3(
    const std::vector<double>& input,
    const Dimensions4D& dimensions)
{
    return fixed_median_3x3_impl(input, dimensions, NthElementMedian{});
}

std::vector<double> fixed_median_3x3_median9(
    const std::vector<double>& input,
    const Dimensions4D& dimensions)
{
    return fixed_median_3x3_impl(input, dimensions, MedianOfNine{});
}

std::vector<double> fixed_median_3x3_median9_direct_addressing(
    const std::vector<double>& input,
    const Dimensions4D& dimensions)
{
    const std::size_t element_count = checked_element_count(dimensions);
    if (input.size() != element_count) {
        throw std::invalid_argument("Input element count does not match the supplied 4D dimensions.");
    }
    if (dimensions.scan_y > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        dimensions.scan_x > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("Scan dimensions are too large for reflected indexing.");
    }

    std::vector<double> output(element_count);

    const std::size_t detector_plane_stride = dimensions.detector_y * dimensions.detector_x;
    const std::size_t scan_y_stride = dimensions.scan_x * detector_plane_stride;

    for (std::size_t scan_y = 0; scan_y < dimensions.scan_y; ++scan_y) {
        const std::size_t output_scan_y_base = scan_y * scan_y_stride;

        for (std::size_t scan_x = 0; scan_x < dimensions.scan_x; ++scan_x) {
            const std::size_t output_scan_base =
                output_scan_y_base + scan_x * detector_plane_stride;

            for (std::size_t detector_y = 0; detector_y < dimensions.detector_y; ++detector_y) {
                const std::size_t detector_row_base = detector_y * dimensions.detector_x;

                for (std::size_t detector_x = 0; detector_x < dimensions.detector_x; ++detector_x) {
                    const std::size_t detector_offset = detector_row_base + detector_x;
                    std::array<double, 9> neighborhood{};
                    std::size_t neighborhood_index = 0;

                    for (std::ptrdiff_t offset_y = -1; offset_y <= 1; ++offset_y) {
                        const std::size_t reflected_y = reflect_index(
                            static_cast<std::ptrdiff_t>(scan_y) + offset_y,
                            dimensions.scan_y);
                        const std::size_t reflected_scan_y_base = reflected_y * scan_y_stride;

                        for (std::ptrdiff_t offset_x = -1; offset_x <= 1; ++offset_x) {
                            const std::size_t reflected_x = reflect_index(
                                static_cast<std::ptrdiff_t>(scan_x) + offset_x,
                                dimensions.scan_x);
                            const std::size_t input_scan_base =
                                reflected_scan_y_base + reflected_x * detector_plane_stride;

                            neighborhood[neighborhood_index] = input[input_scan_base + detector_offset];
                            ++neighborhood_index;
                        }
                    }

                    output[output_scan_base + detector_offset] = MedianOfNine{}(neighborhood);
                }
            }
        }
    }

    return output;
}

int openmp_max_threads()
{
    return omp_get_max_threads();
}

int openmp_processor_count()
{
    return omp_get_num_procs();
}

std::vector<double> fixed_median_3x3_median9_direct_addressing_openmp(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    int thread_count)
{
    const std::size_t element_count = checked_element_count(dimensions);
    if (input.size() != element_count) {
        throw std::invalid_argument("Input element count does not match the supplied 4D dimensions.");
    }
    if (dimensions.scan_y > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        dimensions.scan_x > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("Scan dimensions are too large for reflected indexing.");
    }
    if (thread_count < 1) {
        throw std::invalid_argument("OpenMP thread count must be positive.");
    }

    std::vector<double> output(element_count);

    const std::size_t detector_plane_stride = dimensions.detector_y * dimensions.detector_x;
    const std::size_t scan_y_stride = dimensions.scan_x * detector_plane_stride;

    // Each scan-y slab is contiguous and independent. Static scheduling keeps
    // the coarse slabs stable across threads and avoids shared output cache lines.
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::ptrdiff_t scan_y_signed = 0;
         scan_y_signed < static_cast<std::ptrdiff_t>(dimensions.scan_y);
         ++scan_y_signed) {
        const std::size_t scan_y = static_cast<std::size_t>(scan_y_signed);
        const std::size_t output_scan_y_base = scan_y * scan_y_stride;

        for (std::size_t scan_x = 0; scan_x < dimensions.scan_x; ++scan_x) {
            const std::size_t output_scan_base =
                output_scan_y_base + scan_x * detector_plane_stride;

            for (std::size_t detector_y = 0; detector_y < dimensions.detector_y; ++detector_y) {
                const std::size_t detector_row_base = detector_y * dimensions.detector_x;

                for (std::size_t detector_x = 0; detector_x < dimensions.detector_x; ++detector_x) {
                    const std::size_t detector_offset = detector_row_base + detector_x;
                    std::array<double, 9> neighborhood{};
                    std::size_t neighborhood_index = 0;

                    for (std::ptrdiff_t offset_y = -1; offset_y <= 1; ++offset_y) {
                        const std::size_t reflected_y = reflect_index(
                            static_cast<std::ptrdiff_t>(scan_y) + offset_y,
                            dimensions.scan_y);
                        const std::size_t reflected_scan_y_base = reflected_y * scan_y_stride;

                        for (std::ptrdiff_t offset_x = -1; offset_x <= 1; ++offset_x) {
                            const std::size_t reflected_x = reflect_index(
                                static_cast<std::ptrdiff_t>(scan_x) + offset_x,
                                dimensions.scan_x);
                            const std::size_t input_scan_base =
                                reflected_scan_y_base + reflected_x * detector_plane_stride;

                            neighborhood[neighborhood_index] = input[input_scan_base + detector_offset];
                            ++neighborhood_index;
                        }
                    }

                    output[output_scan_base + detector_offset] = MedianOfNine{}(neighborhood);
                }
            }
        }
    }

    return output;
}

} // namespace phase_a
