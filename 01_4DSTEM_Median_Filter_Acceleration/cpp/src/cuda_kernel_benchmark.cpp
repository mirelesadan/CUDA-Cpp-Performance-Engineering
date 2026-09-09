#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fixed_median.hpp"
#include "fixed_median_cuda.hpp"
#include "npy.hpp"

#ifndef PHASE_A_BENCHMARK_INPUT_PATH
#error "PHASE_A_BENCHMARK_INPUT_PATH must be provided by CMake."
#endif

namespace {

constexpr std::size_t expected_dimension_count = 4;
constexpr const char* expected_dtype = "<f8";
constexpr std::size_t warmup_launch_count = 5;
constexpr std::size_t steady_launch_count = 20;
constexpr unsigned int idle_milliseconds = 3000;
constexpr std::size_t post_idle_launch_count = 5;

struct LoadedNpy {
    npy::npy_data<double> array;
    std::string dtype;
    std::size_t element_count;
};

struct Statistics {
    double minimum = 0.0;
    double median = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double coefficient_of_variation_percent = 0.0;
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

LoadedNpy load_npy(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open benchmark input: " + path.string());
    }

    const npy::header_t header = npy::parse_header(npy::read_header(input));
    const std::string dtype = header.dtype.str();
    if (dtype != expected_dtype) {
        throw std::runtime_error("Expected dtype <f8 (little-endian float64), found " + dtype + '.');
    }
    if (header.shape.size() != expected_dimension_count) {
        throw std::runtime_error("Expected exactly four dimensions.");
    }
    if (header.fortran_order) {
        throw std::runtime_error("Expected C order, found Fortran order.");
    }
    const std::size_t element_count = checked_element_count(header.shape);

    input.clear();
    input.seekg(0, std::ios::beg);
    npy::npy_data<double> array = npy::read_npy<double>(input);
    if (!input || array.data.size() != element_count ||
        array.shape != header.shape || array.fortran_order) {
        throw std::runtime_error("Loaded NPY payload does not match its validated header.");
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

Statistics summarize(const std::vector<double>& values)
{
    if (values.empty()) {
        throw std::runtime_error("Cannot summarize an empty timing sequence.");
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    double squared_deviation_sum = 0.0;
    for (const double value : values) {
        const double deviation = value - mean;
        squared_deviation_sum += deviation * deviation;
    }
    const double population_standard_deviation =
        std::sqrt(squared_deviation_sum / static_cast<double>(values.size()));
    return {
        sorted.front(),
        sorted[sorted.size() / 2],
        sorted.back(),
        mean,
        100.0 * population_standard_deviation / mean,
    };
}

void print_sequence(const char* label, const std::vector<double>& values)
{
    std::cout << label << "_ms:";
    for (const double value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 2) {
            throw std::runtime_error(
                "Usage: phase_a_cuda_kernel_benchmark.exe [median_filter_input.npy]");
        }

        const std::filesystem::path input_path = std::filesystem::absolute(
            argc == 2 ? std::filesystem::path(argv[1])
                      : std::filesystem::path(PHASE_A_BENCHMARK_INPUT_PATH))
                                                       .lexically_normal();
        const LoadedNpy input = load_npy(input_path);
        const phase_a::Dimensions4D dimensions = dimensions_from_shape(input.array.shape);
        const phase_a::CudaDeviceInfo device = phase_a::cuda_device_info();

        phase_a::CudaKernelTimingSequence sequence =
            phase_a::benchmark_fixed_median_3x3_cuda_baseline_kernel(
                input.array.data,
                dimensions,
                warmup_launch_count,
                steady_launch_count,
                idle_milliseconds,
                post_idle_launch_count);

        const std::vector<double> serial_output =
            phase_a::fixed_median_3x3_median9_direct_addressing(input.array.data, dimensions);
        const std::size_t mismatches =
            count_bitwise_mismatches(sequence.output, serial_output);
        const Statistics steady = summarize(sequence.steady_kernel_milliseconds);
        const double million_outputs_per_second =
            static_cast<double>(input.element_count) /
            (steady.median / 1000.0) / 1.0e6;

        std::cout << std::fixed << std::setprecision(6)
                  << "Project 1 Phase A CUDA kernel timing audit\n"
                  << "Input path: " << input_path.string() << '\n'
                  << "Dtype: " << input.dtype << " (float64)\n"
                  << "Shape: " << shape_as_string(input.array.shape) << '\n'
                  << "Storage order: C-contiguous\n"
                  << "Input elements: " << input.element_count << '\n'
                  << "CUDA device: " << device.name << '\n'
                  << "Compute capability: sm_" << device.compute_major << device.compute_minor << '\n'
                  << "Kernel mapping: one output per thread, 256 threads per one-dimensional block\n"
                  << "Device allocation, H2D copy, event creation, and runtime setup: before sequences\n"
                  << "Timed boundary: CUDA events around one kernel launch only\n"
                  << "Device buffers: reused for every launch\n"
                  << "Warm-up launches (diagnostic, excluded): " << warmup_launch_count << '\n'
                  << "Steady-state launches: " << steady_launch_count << '\n'
                  << "Idle delay before recovery sequence_ms: " << idle_milliseconds << '\n';
        print_sequence("Warmup_kernel", sequence.warmup_kernel_milliseconds);
        print_sequence("Steady_kernel", sequence.steady_kernel_milliseconds);
        print_sequence("Post_idle_kernel", sequence.post_idle_kernel_milliseconds);
        std::cout << "Steady min_median_max_ms: "
                  << steady.minimum << ", " << steady.median << ", " << steady.maximum << '\n'
                  << "Steady mean_ms: " << steady.mean << '\n'
                  << "Steady coefficient_of_variation_percent: "
                  << steady.coefficient_of_variation_percent << '\n'
                  << "Steady rate_Moutputs_per_s: " << million_outputs_per_second << '\n'
                  << "Optimized-serial-vs-CUDA bitwise mismatches: " << mismatches << '\n'
                  << "Validation result: " << (mismatches == 0 ? "PASS" : "FAIL") << '\n';

        return mismatches == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
