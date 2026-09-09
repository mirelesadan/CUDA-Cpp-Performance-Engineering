#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fixed_median.hpp"
#include "fixed_median_cuda.hpp"
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
    const std::string dtype = header.dtype.str();
    if (dtype != expected_dtype) {
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
    return {std::move(array), dtype, element_count};
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

std::size_t count_bitwise_mismatches(
    const std::vector<double>& actual,
    const std::vector<double>& expected)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Cannot compare outputs with different element counts.");
    }

    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        mismatches += double_bits(actual[index]) != double_bits(expected[index]);
    }
    return mismatches;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 3) {
            throw std::runtime_error(
                "Usage: phase_a_cuda_validation.exe [input.npy] [reference_output.npy]");
        }

        const std::filesystem::path input_path = std::filesystem::absolute(
            argc >= 2 ? std::filesystem::path(argv[1])
                      : std::filesystem::path(PHASE_A_REFERENCE_INPUT_PATH))
                                                       .lexically_normal();
        const std::filesystem::path reference_path = std::filesystem::absolute(
            argc == 3 ? std::filesystem::path(argv[2])
                      : std::filesystem::path(PHASE_A_REFERENCE_OUTPUT_PATH))
                                                       .lexically_normal();

        const LoadedNpy input = load_npy(input_path, "input");
        const LoadedNpy reference = load_npy(reference_path, "reference output");
        if (input.array.shape != reference.array.shape ||
            input.element_count != reference.element_count) {
            throw std::runtime_error("Input and reference output metadata do not match.");
        }

        const phase_a::Dimensions4D dimensions = dimensions_from_shape(input.array.shape);
        const phase_a::CudaDeviceInfo device = phase_a::cuda_device_info();
        const std::vector<double> serial_output =
            phase_a::fixed_median_3x3_median9_direct_addressing(input.array.data, dimensions);
        const phase_a::CudaFilterResult cuda_result =
            phase_a::fixed_median_3x3_cuda_baseline(input.array.data, dimensions);

        const std::size_t reference_mismatches =
            count_bitwise_mismatches(cuda_result.output, reference.array.data);
        const std::size_t serial_mismatches =
            count_bitwise_mismatches(cuda_result.output, serial_output);

        std::cout << "Project 1 Phase A correctness-first CUDA baseline validation\n"
                  << "CUDA device: " << device.name << '\n'
                  << "Compute capability: sm_" << device.compute_major << device.compute_minor << '\n'
                  << "Kernel mapping: one output per thread, 256 threads per one-dimensional block\n"
                  << "Elements compared: " << input.element_count << '\n'
                  << "CUDA-vs-reference bitwise mismatches: " << reference_mismatches << '\n'
                  << "CUDA-vs-optimized-serial bitwise mismatches: " << serial_mismatches << '\n'
                  << "Validation result: "
                  << (reference_mismatches == 0 && serial_mismatches == 0 ? "PASS" : "FAIL")
                  << '\n';

        return reference_mismatches == 0 && serial_mismatches == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
