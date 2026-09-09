#pragma once

#include <cstddef>
#include <vector>

namespace phase_a {

struct Dimensions4D {
    std::size_t scan_y;
    std::size_t scan_x;
    std::size_t detector_y;
    std::size_t detector_x;
};

std::size_t flat_index(
    const Dimensions4D& dimensions,
    std::size_t scan_y_index,
    std::size_t scan_x_index,
    std::size_t detector_y_index,
    std::size_t detector_x_index);

std::size_t reflect_index(std::ptrdiff_t index, std::size_t size);

std::vector<double> fixed_median_3x3(
    const std::vector<double>& input,
    const Dimensions4D& dimensions);

std::vector<double> fixed_median_3x3_median9(
    const std::vector<double>& input,
    const Dimensions4D& dimensions);

std::vector<double> fixed_median_3x3_median9_direct_addressing(
    const std::vector<double>& input,
    const Dimensions4D& dimensions);

int openmp_max_threads();

int openmp_processor_count();

std::vector<double> fixed_median_3x3_median9_direct_addressing_openmp(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    int thread_count);

} // namespace phase_a
