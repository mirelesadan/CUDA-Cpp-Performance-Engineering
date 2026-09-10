#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fixed_median.hpp"
#include "npy.hpp"

#ifndef PHASE_A_REFERENCE_INPUT_PATH
#error "PHASE_A_REFERENCE_INPUT_PATH must be provided by CMake."
#endif

#ifndef PHASE_A_REFERENCE_OUTPUT_PATH
#error "PHASE_A_REFERENCE_OUTPUT_PATH must be provided by CMake."
#endif

namespace {

constexpr std::size_t expected_dimension_count = 4;
constexpr const char* expected_dtype = "<f8";

struct LoadedNpy {
    npy::npy_data<double> array;
    std::string dtype;
    std::size_t element_count;
    std::uintmax_t file_bytes;
};

struct ComparisonSummary {
    std::size_t mismatch_count = 0;
    bool has_first_mismatch = false;
    std::size_t first_mismatch_index = 0;
    double first_actual_value = 0.0;
    double first_expected_value = 0.0;
    std::uint64_t first_actual_bits = 0;
    std::uint64_t first_expected_bits = 0;
    double maximum_absolute_difference = 0.0;
};

std::string shape_as_string(const npy::shape_t& shape)
{
    std::ostringstream text;
    text << '(';

    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            text << ", ";
        }
        text << shape[index];
    }

    text << ')';
    return text.str();
}

std::size_t checked_element_count(const npy::shape_t& shape)
{
    std::size_t count = 1;
    npy::ndarray_len_t libnpy_count = 1;

    for (const npy::ndarray_len_t dimension : shape) {
        if (dimension == 0) {
            throw std::runtime_error("NPY contract error: every dimension must be nonempty.");
        }

        const std::size_t dimension_size = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / dimension_size) {
            throw std::runtime_error("NPY contract error: shape product exceeds std::size_t.");
        }
        if (libnpy_count > std::numeric_limits<npy::ndarray_len_t>::max() / dimension) {
            throw std::runtime_error("NPY contract error: shape product exceeds libnpy's supported range.");
        }

        count *= dimension_size;
        libnpy_count *= dimension;
    }

    return count;
}

void validate_header(const npy::header_t& header, const std::string& role)
{
    const std::string detected_dtype = header.dtype.str();
    if (detected_dtype != expected_dtype) {
        throw std::runtime_error(
            role + " contract error: expected dtype <f8 (little-endian float64), found " + detected_dtype + '.');
    }
    if (header.shape.size() != expected_dimension_count) {
        throw std::runtime_error(
            role + " contract error: expected 4 dimensions, found " + std::to_string(header.shape.size()) + '.');
    }
    if (header.fortran_order) {
        throw std::runtime_error(role + " contract error: expected C order, found Fortran order.");
    }

    checked_element_count(header.shape);
}

LoadedNpy load_npy(const std::filesystem::path& path, const std::string& role)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open " + role + ": " + path.string());
    }

    const npy::header_t header = npy::parse_header(npy::read_header(input));
    validate_header(header, role);
    const std::size_t element_count = checked_element_count(header.shape);

    input.clear();
    input.seekg(0, std::ios::beg);
    npy::npy_data<double> array = npy::read_npy<double>(input);

    if (!input) {
        throw std::runtime_error(role + " data error: the float64 payload is incomplete.");
    }
    if (array.shape != header.shape || array.fortran_order != header.fortran_order) {
        throw std::runtime_error(role + " data error: loaded metadata does not match the validated header.");
    }
    if (array.data.size() != element_count) {
        throw std::runtime_error(role + " data error: loaded element count does not match the shape.");
    }

    return {std::move(array), header.dtype.str(), element_count, std::filesystem::file_size(path)};
}

void validate_matching_arrays(const LoadedNpy& input, const LoadedNpy& reference)
{
    if (input.dtype != reference.dtype) {
        throw std::runtime_error("Reference contract error: input and expected-output dtypes differ.");
    }
    if (input.array.shape != reference.array.shape) {
        throw std::runtime_error("Reference contract error: input and expected-output shapes differ.");
    }
    if (input.element_count != reference.element_count) {
        throw std::runtime_error("Reference contract error: input and expected-output element counts differ.");
    }
}

phase_a::Dimensions4D dimensions_from_shape(const npy::shape_t& shape)
{
    return {
        static_cast<std::size_t>(shape[0]),
        static_cast<std::size_t>(shape[1]),
        static_cast<std::size_t>(shape[2]),
        static_cast<std::size_t>(shape[3]),
    };
}

std::array<std::size_t, 4> coordinate_from_flat_index(
    std::size_t index,
    const phase_a::Dimensions4D& dimensions)
{
    std::array<std::size_t, 4> coordinate{};

    coordinate[3] = index % dimensions.detector_x;
    index /= dimensions.detector_x;
    coordinate[2] = index % dimensions.detector_y;
    index /= dimensions.detector_y;
    coordinate[1] = index % dimensions.scan_x;
    index /= dimensions.scan_x;
    coordinate[0] = index;

    return coordinate;
}

std::uint64_t double_bits(double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t), "Bitwise validation requires 64-bit double.");

    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::string bits_as_hex(std::uint64_t bits)
{
    std::ostringstream text;
    text << "0x" << std::hex << std::setw(16) << std::setfill('0') << bits;
    return text.str();
}

ComparisonSummary compare_bitwise(
    const std::vector<double>& actual,
    const std::vector<double>& expected)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Cannot compare outputs with different element counts.");
    }

    ComparisonSummary summary;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double absolute_difference = std::abs(actual[index] - expected[index]);
        if (absolute_difference > summary.maximum_absolute_difference) {
            summary.maximum_absolute_difference = absolute_difference;
        }

        const std::uint64_t actual_bits = double_bits(actual[index]);
        const std::uint64_t expected_bits = double_bits(expected[index]);
        if (actual_bits != expected_bits) {
            ++summary.mismatch_count;

            if (!summary.has_first_mismatch) {
                summary.has_first_mismatch = true;
                summary.first_mismatch_index = index;
                summary.first_actual_value = actual[index];
                summary.first_expected_value = expected[index];
                summary.first_actual_bits = actual_bits;
                summary.first_expected_bits = expected_bits;
            }
        }
    }

    return summary;
}

bool is_scan_boundary(
    const std::array<std::size_t, 4>& coordinate,
    const phase_a::Dimensions4D& dimensions)
{
    return coordinate[0] == 0 || coordinate[0] + 1 == dimensions.scan_y ||
           coordinate[1] == 0 || coordinate[1] + 1 == dimensions.scan_x;
}

void print_output_check(
    const std::string& label,
    const std::array<std::size_t, 4>& coordinate,
    const phase_a::Dimensions4D& dimensions,
    const std::vector<double>& actual,
    const std::vector<double>& expected)
{
    const std::size_t index = phase_a::flat_index(
        dimensions,
        coordinate[0],
        coordinate[1],
        coordinate[2],
        coordinate[3]);

    std::cout << "  " << label << " (" << coordinate[0] << ", " << coordinate[1] << ", " << coordinate[2]
              << ", " << coordinate[3] << ") -> flat index " << index << " -> C++ " << actual[index]
              << ", Python " << expected[index]
              << ", bitwise match: " << (double_bits(actual[index]) == double_bits(expected[index]) ? "yes" : "no")
              << '\n';
}

void print_top_left_neighborhood(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions)
{
    std::cout << "Top-left reflected 3x3 scan neighborhood at detector (0, 0):\n";

    for (std::ptrdiff_t offset_y = -1; offset_y <= 1; ++offset_y) {
        for (std::ptrdiff_t offset_x = -1; offset_x <= 1; ++offset_x) {
            const std::size_t reflected_y = phase_a::reflect_index(offset_y, dimensions.scan_y);
            const std::size_t reflected_x = phase_a::reflect_index(offset_x, dimensions.scan_x);
            const std::size_t index = phase_a::flat_index(dimensions, reflected_y, reflected_x, 0, 0);

            std::cout << "  raw scan (" << offset_y << ", " << offset_x << ") -> reflected (" << reflected_y
                      << ", " << reflected_x << ") -> " << input[index] << '\n';
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 3) {
            throw std::runtime_error(
                "Usage: phase_a_fixed_median.exe [reference_input.npy] [reference_output_python.npy]");
        }

        const std::filesystem::path input_path = std::filesystem::absolute(
            argc >= 2 ? std::filesystem::path(argv[1]) : std::filesystem::path(PHASE_A_REFERENCE_INPUT_PATH))
                                                       .lexically_normal();
        const std::filesystem::path reference_path = std::filesystem::absolute(
            argc == 3 ? std::filesystem::path(argv[2]) : std::filesystem::path(PHASE_A_REFERENCE_OUTPUT_PATH))
                                                       .lexically_normal();

        const LoadedNpy input = load_npy(input_path, "Phase A input");
        const LoadedNpy reference = load_npy(reference_path, "Phase A expected output");
        validate_matching_arrays(input, reference);

        const phase_a::Dimensions4D dimensions = dimensions_from_shape(input.array.shape);
        const std::vector<double> input_before = input.array.data;
        const std::vector<double> baseline_output =
            phase_a::fixed_median_3x3(input.array.data, dimensions);
        const std::vector<double> median9_output =
            phase_a::fixed_median_3x3_median9(input.array.data, dimensions);
        const std::vector<double> optimized_serial_output =
            phase_a::fixed_median_3x3_median9_direct_addressing(input.array.data, dimensions);
        const int openmp_threads =
            std::max(1, std::min(phase_a::openmp_max_threads(), phase_a::openmp_processor_count()));
        const std::vector<double> openmp_output =
            phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                input.array.data, dimensions, openmp_threads);
        const ComparisonSummary baseline_comparison =
            compare_bitwise(baseline_output, reference.array.data);
        const ComparisonSummary median9_comparison =
            compare_bitwise(median9_output, reference.array.data);
        const ComparisonSummary optimized_serial_comparison =
            compare_bitwise(optimized_serial_output, reference.array.data);
        const ComparisonSummary openmp_comparison =
            compare_bitwise(openmp_output, reference.array.data);
        const ComparisonSummary serial_openmp_comparison =
            compare_bitwise(openmp_output, optimized_serial_output);
        const ComparisonSummary input_comparison =
            compare_bitwise(input.array.data, input_before);

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        std::cout << "Project 1 - Phase A consolidated CPU validation\n"
                  << "Input path: " << input_path.string() << '\n'
                  << "Reference path: " << reference_path.string() << '\n'
                  << "Dtype: " << input.dtype << " (float64)\n"
                  << "Shape: " << shape_as_string(input.array.shape) << '\n'
                  << "Storage order: C-contiguous\n"
                  << "Input elements: " << input.element_count << '\n'
                  << "Input data bytes: " << input.element_count * sizeof(double) << '\n'
                  << "Input NPY file bytes: " << input.file_bytes << '\n'
                  << "Reference metadata match: yes\n"
                  << "OpenMP validation threads: " << openmp_threads << '\n'
                  << "OpenMP maximum threads: " << phase_a::openmp_max_threads() << '\n'
                  << "OpenMP processor count: " << phase_a::openmp_processor_count() << '\n'
                  << "Reflect mapping checks: -1 -> " << phase_a::reflect_index(-1, dimensions.scan_y)
                  << ", 0 -> " << phase_a::reflect_index(0, dimensions.scan_y)
                  << ", N-1 -> "
                  << phase_a::reflect_index(static_cast<std::ptrdiff_t>(dimensions.scan_y - 1), dimensions.scan_y)
                  << ", N -> "
                  << phase_a::reflect_index(static_cast<std::ptrdiff_t>(dimensions.scan_y), dimensions.scan_y)
                  << '\n';

        print_top_left_neighborhood(input.array.data, dimensions);

        const std::array<std::size_t, 4> top_left = {0, 0, 0, 0};
        const std::array<std::size_t, 4> interior = {
            dimensions.scan_y / 2,
            dimensions.scan_x / 2,
            dimensions.detector_y / 2,
            dimensions.detector_x / 2,
        };
        const std::array<std::size_t, 4> bottom_right = {
            dimensions.scan_y - 1,
            dimensions.scan_x - 1,
            dimensions.detector_y - 1,
            dimensions.detector_x - 1,
        };

        std::cout << "Output sanity checks:\n";
        print_output_check("top/left boundary", top_left, dimensions, openmp_output, reference.array.data);
        print_output_check("interior", interior, dimensions, openmp_output, reference.array.data);
        print_output_check("bottom/right boundary", bottom_right, dimensions, openmp_output, reference.array.data);

        std::cout << "Bitwise validation:\n"
                  << "  Elements compared per implementation: " << openmp_output.size() << '\n'
                  << "  Straightforward baseline-vs-reference mismatches: "
                  << baseline_comparison.mismatch_count << '\n'
                  << "  Median-of-nine-vs-reference mismatches: "
                  << median9_comparison.mismatch_count << '\n'
                  << "  Optimized serial-vs-reference mismatches: "
                  << optimized_serial_comparison.mismatch_count << '\n'
                  << "  OpenMP-vs-reference mismatches: "
                  << openmp_comparison.mismatch_count << '\n'
                  << "  Optimized serial-vs-OpenMP mismatches: "
                  << serial_openmp_comparison.mismatch_count << '\n'
                  << "  Input-after-filter mismatches: "
                  << input_comparison.mismatch_count << '\n'
                  << "  Maximum OpenMP-vs-reference absolute difference: "
                  << openmp_comparison.maximum_absolute_difference << '\n';

        if (openmp_comparison.has_first_mismatch) {
            const std::array<std::size_t, 4> coordinate =
                coordinate_from_flat_index(openmp_comparison.first_mismatch_index, dimensions);

            std::cout << "  First mismatch coordinate: (" << coordinate[0] << ", " << coordinate[1] << ", "
                      << coordinate[2] << ", " << coordinate[3] << ")\n"
                      << "  First mismatch flat index: " << openmp_comparison.first_mismatch_index << '\n'
                      << "  C++ value: " << openmp_comparison.first_actual_value << " ("
                      << bits_as_hex(openmp_comparison.first_actual_bits) << ")\n"
                      << "  Python value: " << openmp_comparison.first_expected_value << " ("
                      << bits_as_hex(openmp_comparison.first_expected_bits) << ")\n"
                      << "  On scan boundary: " << (is_scan_boundary(coordinate, dimensions) ? "yes" : "no") << '\n'
                      << "Validation result: FAIL\n";
            return 1;
        }

        if (baseline_comparison.mismatch_count != 0 ||
            median9_comparison.mismatch_count != 0 ||
            optimized_serial_comparison.mismatch_count != 0 ||
            serial_openmp_comparison.mismatch_count != 0 ||
            input_comparison.mismatch_count != 0) {
            std::cout << "Validation result: FAIL (consolidated implementation mismatch)\n";
            return 1;
        }

        std::cout << "  First mismatch: none\n"
                  << "Validation result: PASS\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
