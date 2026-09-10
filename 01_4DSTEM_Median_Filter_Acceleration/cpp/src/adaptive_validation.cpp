#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "adaptive_median.hpp"
#include "npy.hpp"

#ifndef PHASE_B_REFERENCE_INPUT_PATH
#error "PHASE_B_REFERENCE_INPUT_PATH must be provided by CMake."
#endif

#ifndef PHASE_B_REFERENCE_OUTPUT_PATH
#error "PHASE_B_REFERENCE_OUTPUT_PATH must be provided by CMake."
#endif

namespace {

constexpr std::size_t expected_dimension_count = 4;
constexpr const char* expected_dtype = "<f8";

struct LoadedNpy {
    npy::npy_data<double> array;
    std::size_t element_count;
};

std::size_t checked_element_count(const npy::shape_t& shape)
{
    std::size_t count = 1;
    for (const npy::ndarray_len_t dimension : shape) {
        if (dimension == 0) {
            throw std::runtime_error("NPY contract error: every dimension must be nonempty.");
        }
        const std::size_t size = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / size) {
            throw std::runtime_error("NPY contract error: shape product exceeds std::size_t.");
        }
        count *= size;
    }
    return count;
}

LoadedNpy load_npy(const std::filesystem::path& path, const char* role)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("Could not open ") + role + ": " + path.string());
    }

    const npy::header_t header = npy::parse_header(npy::read_header(input));
    if (header.dtype.str() != expected_dtype) {
        throw std::runtime_error(std::string(role) + " must use little-endian float64.");
    }
    if (header.shape.size() != expected_dimension_count) {
        throw std::runtime_error(std::string(role) + " must have exactly four dimensions.");
    }
    if (header.fortran_order) {
        throw std::runtime_error(std::string(role) + " must be C-contiguous.");
    }
    const std::size_t element_count = checked_element_count(header.shape);

    input.clear();
    input.seekg(0, std::ios::beg);
    npy::npy_data<double> array = npy::read_npy<double>(input);
    if (!input || array.data.size() != element_count ||
        array.shape != header.shape || array.fortran_order) {
        throw std::runtime_error(std::string(role) + " payload does not match its header.");
    }
    return {std::move(array), element_count};
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

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 3) {
            throw std::runtime_error(
                "Usage: phase_b_adaptive_median_validation.exe [input.npy] [reference_output.npy]");
        }

        const std::filesystem::path input_path = std::filesystem::absolute(
            argc >= 2 ? std::filesystem::path(argv[1])
                      : std::filesystem::path(PHASE_B_REFERENCE_INPUT_PATH))
                                                       .lexically_normal();
        const std::filesystem::path reference_path = std::filesystem::absolute(
            argc == 3 ? std::filesystem::path(argv[2])
                      : std::filesystem::path(PHASE_B_REFERENCE_OUTPUT_PATH))
                                                       .lexically_normal();

        const LoadedNpy input = load_npy(input_path, "input");
        const LoadedNpy reference = load_npy(reference_path, "reference output");
        if (input.array.shape != reference.array.shape ||
            input.element_count != reference.element_count) {
            throw std::runtime_error("Input and reference output metadata do not match.");
        }

        const std::vector<double> original_input = input.array.data;
        const auto start = std::chrono::steady_clock::now();
        const std::vector<double> actual = phase_b::adaptive_median_s3_smax7(
            input.array.data,
            dimensions_from_shape(input.array.shape));
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();

        std::size_t output_mismatches = 0;
        std::size_t first_mismatch = actual.size();
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (double_bits(actual[index]) != double_bits(reference.array.data[index])) {
                if (first_mismatch == actual.size()) {
                    first_mismatch = index;
                }
                ++output_mismatches;
            }
        }

        std::size_t input_mismatches = 0;
        for (std::size_t index = 0; index < original_input.size(); ++index) {
            input_mismatches +=
                double_bits(original_input[index]) != double_bits(input.array.data[index]);
        }

        std::cout << "Project 1 Phase B correctness-first adaptive median validation\n"
                  << "Shape: (" << input.array.shape[0] << ", " << input.array.shape[1]
                  << ", " << input.array.shape[2] << ", " << input.array.shape[3] << ")\n"
                  << "Elements compared: " << input.element_count << '\n'
                  << "Bitwise mismatches: " << output_mismatches << '\n'
                  << "Input bitwise changes: " << input_mismatches << '\n'
                  << std::fixed << std::setprecision(6)
                  << "Filter time: " << elapsed_ms << " ms\n";

        if (first_mismatch != actual.size()) {
            const std::size_t detector_x = first_mismatch % input.array.shape[3];
            std::size_t remaining = first_mismatch / input.array.shape[3];
            const std::size_t detector_y = remaining % input.array.shape[2];
            remaining /= input.array.shape[2];
            const std::size_t scan_x = remaining % input.array.shape[1];
            const std::size_t scan_y = remaining / input.array.shape[1];
            std::cout << "First mismatch: flat index " << first_mismatch
                      << " at (" << scan_y << ", " << scan_x << ", "
                      << detector_y << ", " << detector_x << ")"
                      << ", actual bits 0x" << std::hex << double_bits(actual[first_mismatch])
                      << ", expected bits 0x"
                      << double_bits(reference.array.data[first_mismatch]) << std::dec << '\n';
        }

        const bool passed = output_mismatches == 0 && input_mismatches == 0;
        std::cout << "Validation result: " << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
