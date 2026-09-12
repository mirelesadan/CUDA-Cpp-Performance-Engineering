#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "adaptive_median.hpp"
#include "adaptive_median_cuda.hpp"
#include "adaptive_median_detail.hpp"
#include "fixed_median.hpp"
#include "npy.hpp"

#ifndef PHASE_B_REFERENCE_INPUT_PATH
#error "PHASE_B_REFERENCE_INPUT_PATH must be provided by CMake."
#endif
#ifndef PHASE_B_REFERENCE_OUTPUT_PATH
#error "PHASE_B_REFERENCE_OUTPUT_PATH must be provided by CMake."
#endif

namespace {

struct LoadedNpy {
    npy::npy_data<double> array;
    std::size_t element_count;
};

LoadedNpy load_npy(const std::filesystem::path& path, const char* label)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(std::string("Could not open ") + label + ": " + path.string());
    }
    const npy::header_t header = npy::parse_header(npy::read_header(stream));
    if (header.dtype.str() != "<f8" || header.shape.size() != 4 || header.fortran_order) {
        throw std::runtime_error(std::string(label) + " must be C-order four-dimensional float64.");
    }
    std::size_t count = 1;
    for (const auto dimension : header.shape) {
        if (dimension == 0 || count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error(std::string(label) + " has invalid dimensions.");
        }
        count *= dimension;
    }
    stream.clear();
    stream.seekg(0);
    auto array = npy::read_npy<double>(stream);
    if (!stream || array.data.size() != count || array.shape != header.shape || array.fortran_order) {
        throw std::runtime_error(std::string(label) + " payload does not match its header.");
    }
    return {std::move(array), count};
}

phase_a::Dimensions4D dimensions_from(const npy::shape_t& shape)
{
    return {shape[0], shape[1], shape[2], shape[3]};
}

std::uint64_t bits(double value)
{
    std::uint64_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::size_t mismatches(const std::vector<double>& left, const std::vector<double>& right)
{
    if (left.size() != right.size()) {
        return std::max(left.size(), right.size());
    }
    std::size_t count = 0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        count += bits(left[index]) != bits(right[index]);
    }
    return count;
}

std::size_t statistic_mismatches(
    const phase_b::AdaptiveMedianStatistics& left,
    const phase_b::AdaptiveMedianStatistics& right)
{
    return
        (left.total_outputs != right.total_outputs) +
        (left.finished_at_3x3 != right.finished_at_3x3) +
        (left.expanded_to_5x5 != right.expanded_to_5x5) +
        (left.finished_at_5x5 != right.finished_at_5x5) +
        (left.expanded_to_7x7 != right.expanded_to_7x7) +
        (left.finished_at_7x7 != right.finished_at_7x7) +
        (left.maximum_window_fallback != right.maximum_window_fallback) +
        (left.stage_b_retained_center != right.stage_b_retained_center) +
        (left.stage_b_replaced_with_median != right.stage_b_replaced_with_median) +
        (left.median_computations != right.median_computations);
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 3) {
            throw std::runtime_error(
                "Usage: phase_b_adaptive_cuda_validation.exe [input.npy] [reference.npy]");
        }
        const auto input = load_npy(
            argc >= 2 ? argv[1] : PHASE_B_REFERENCE_INPUT_PATH, "adaptive input");
        const auto reference = load_npy(
            argc == 3 ? argv[2] : PHASE_B_REFERENCE_OUTPUT_PATH, "adaptive reference");
        if (input.array.shape != reference.array.shape) {
            throw std::runtime_error("Input and reference shapes differ.");
        }
        const auto dimensions = dimensions_from(input.array.shape);
        const std::vector<double> original = input.array.data;
        const auto baseline = phase_b::adaptive_median_s3_smax7(input.array.data, dimensions);
        const auto serial = phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(
            input.array.data, dimensions);
        const auto openmp = phase_b::adaptive_median_s3_smax7_openmp(
            input.array.data, dimensions, 4);
        const auto cpu_diagnostic =
            phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
                input.array.data, dimensions);
        const auto cuda_diagnostic =
            phase_b::adaptive_median_s3_smax7_cuda_baseline_diagnostics(
                input.array.data, dimensions);
        const auto split_cuda_diagnostic =
            phase_b::adaptive_median_s3_smax7_cuda_split_diagnostics(
                input.array.data, dimensions);
        const auto split_cuda = phase_b::adaptive_median_s3_smax7_cuda_split(
            input.array.data, dimensions);

        const std::size_t baseline_reference = mismatches(baseline, reference.array.data);
        const std::size_t serial_reference = mismatches(serial, reference.array.data);
        const std::size_t openmp_serial = mismatches(openmp, serial);
        const std::size_t cuda_serial = mismatches(cuda_diagnostic.output, serial);
        const std::size_t split_cuda_serial = mismatches(split_cuda_diagnostic.output, serial);
        const std::size_t split_cuda_baseline = mismatches(
            split_cuda_diagnostic.output, cuda_diagnostic.output);
        const std::size_t split_cuda_diagnostic_difference = mismatches(
            split_cuda.output, split_cuda_diagnostic.output);
        const std::size_t diagnostic_difference = statistic_mismatches(
            cpu_diagnostic.statistics, cuda_diagnostic.statistics);
        const std::size_t split_diagnostic_difference = statistic_mismatches(
            cpu_diagnostic.statistics, split_cuda_diagnostic.statistics);
        const std::size_t input_changes = mismatches(input.array.data, original);

        std::cout << "Compared values: " << input.element_count << '\n'
                  << "Baseline vs reference mismatches: " << baseline_reference << '\n'
                  << "Optimized serial vs reference mismatches: " << serial_reference << '\n'
                  << "OpenMP vs optimized serial mismatches: " << openmp_serial << '\n'
                  << "CUDA vs optimized serial mismatches: " << cuda_serial << '\n'
                  << "Split CUDA vs optimized serial mismatches: " << split_cuda_serial << '\n'
                  << "Split CUDA vs monolithic CUDA mismatches: " << split_cuda_baseline << '\n'
                  << "Split CUDA normal vs diagnostic mismatches: "
                  << split_cuda_diagnostic_difference << '\n'
                  << "CUDA vs CPU diagnostic-counter differences: " << diagnostic_difference << '\n'
                  << "Split CUDA vs CPU diagnostic-counter differences: "
                  << split_diagnostic_difference << '\n'
                  << "Input bit changes: " << input_changes << '\n';
        const bool success = baseline_reference == 0 && serial_reference == 0 &&
            openmp_serial == 0 && cuda_serial == 0 && split_cuda_serial == 0 &&
            split_cuda_baseline == 0 && split_cuda_diagnostic_difference == 0 &&
            diagnostic_difference == 0 &&
            split_diagnostic_difference == 0 && input_changes == 0;
        std::cout << "Validation: " << (success ? "PASS" : "FAIL") << '\n';
        return success ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Adaptive CUDA validation failed: " << error.what() << '\n';
        return 1;
    }
}
