#include "adaptive_median.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace phase_b {
namespace {

constexpr std::size_t padding = 3;
constexpr std::array<std::size_t, 3> window_sizes = {3, 5, 7};

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* message)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

std::size_t checked_element_count(const phase_a::Dimensions4D& dimensions)
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
            throw std::invalid_argument("Adaptive median requires four nonempty dimensions.");
        }
        count = checked_multiply(count, size, "The 4D shape product exceeds std::size_t.");
    }
    return count;
}

std::size_t padded_index(
    std::size_t row,
    std::size_t column,
    std::size_t padded_scan_x)
{
    return row * padded_scan_x + column;
}

double median_of_window(std::vector<double>& values)
{
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

} // namespace

std::vector<double> adaptive_median_s3_smax7(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions)
{
    const std::size_t element_count = checked_element_count(dimensions);
    if (input.size() != element_count) {
        throw std::invalid_argument("Input element count does not match the supplied 4D dimensions.");
    }
    if (!std::all_of(input.begin(), input.end(), [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Adaptive median requires finite float64 values.");
    }
    if (dimensions.scan_y > std::numeric_limits<std::size_t>::max() - 2 * padding ||
        dimensions.scan_x > std::numeric_limits<std::size_t>::max() - 2 * padding) {
        throw std::overflow_error("Padded scan dimensions exceed std::size_t.");
    }

    const std::size_t padded_scan_y = dimensions.scan_y + 2 * padding;
    const std::size_t padded_scan_x = dimensions.scan_x + 2 * padding;
    const std::size_t padded_element_count = checked_multiply(
        padded_scan_y,
        padded_scan_x,
        "The padded scan-plane size exceeds std::size_t.");

    // The output is always a distinct allocation, leaving input unchanged.
    std::vector<double> output(element_count);

    for (std::size_t detector_y = 0; detector_y < dimensions.detector_y; ++detector_y) {
        for (std::size_t detector_x = 0; detector_x < dimensions.detector_x; ++detector_x) {
            double plane_minimum = input[phase_a::flat_index(
                dimensions, 0, 0, detector_y, detector_x)];
            for (std::size_t scan_y = 0; scan_y < dimensions.scan_y; ++scan_y) {
                for (std::size_t scan_x = 0; scan_x < dimensions.scan_x; ++scan_x) {
                    plane_minimum = std::min(
                        plane_minimum,
                        input[phase_a::flat_index(
                            dimensions, scan_y, scan_x, detector_y, detector_x)]);
                }
            }

            // Reproduce np.pad(..., mode="constant", constant_values=plane_minimum)
            // for one complete scan-space plane.
            std::vector<double> padded_plane(padded_element_count, plane_minimum);
            for (std::size_t scan_y = 0; scan_y < dimensions.scan_y; ++scan_y) {
                for (std::size_t scan_x = 0; scan_x < dimensions.scan_x; ++scan_x) {
                    padded_plane[padded_index(
                        scan_y + padding,
                        scan_x + padding,
                        padded_scan_x)] = input[phase_a::flat_index(
                            dimensions, scan_y, scan_x, detector_y, detector_x)];
                }
            }

            for (std::size_t scan_y = 0; scan_y < dimensions.scan_y; ++scan_y) {
                for (std::size_t scan_x = 0; scan_x < dimensions.scan_x; ++scan_x) {
                    const std::size_t center_y = scan_y + padding;
                    const std::size_t center_x = scan_x + padding;
                    const double center = padded_plane[padded_index(
                        center_y, center_x, padded_scan_x)];
                    double result = center;

                    for (const std::size_t window_size : window_sizes) {
                        const std::size_t radius = window_size / 2;
                        std::vector<double> window;
                        window.reserve(window_size * window_size);
                        double local_minimum = std::numeric_limits<double>::infinity();
                        double local_maximum = -std::numeric_limits<double>::infinity();

                        for (std::size_t window_y = center_y - radius;
                             window_y <= center_y + radius;
                             ++window_y) {
                            for (std::size_t window_x = center_x - radius;
                                 window_x <= center_x + radius;
                                 ++window_x) {
                                const double value = padded_plane[padded_index(
                                    window_y, window_x, padded_scan_x)];
                                window.push_back(value);
                                local_minimum = std::min(local_minimum, value);
                                local_maximum = std::max(local_maximum, value);
                            }
                        }

                        const double local_median = median_of_window(window);
                        if (local_minimum < local_median && local_median < local_maximum) {
                            result = local_minimum < center && center < local_maximum
                                ? center
                                : local_median;
                            break;
                        }

                        // The Python reference returns the center associated with
                        // the final tested window when growth beyond sMax is requested.
                        if (window_size == window_sizes.back()) {
                            result = center;
                        }
                    }

                    output[phase_a::flat_index(
                        dimensions, scan_y, scan_x, detector_y, detector_x)] = result;
                }
            }
        }
    }

    return output;
}

} // namespace phase_b
