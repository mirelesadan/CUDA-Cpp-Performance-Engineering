#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
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
constexpr std::size_t timed_run_count = 3;

using Clock = std::chrono::steady_clock;
using Coordinate = std::array<std::size_t, 4>;

struct LoadedNpy {
    npy::npy_data<double> array;
    std::string dtype;
    std::size_t element_count;
    std::uintmax_t file_bytes;
};

struct Statistics {
    double minimum = 0.0;
    double median = 0.0;
    double maximum = 0.0;
};

struct CpuSummary {
    std::array<double, timed_run_count> milliseconds{};
    std::array<std::array<double, 3>, timed_run_count> samples{};
    Statistics statistics;
    double million_outputs_per_second = 0.0;
};

struct CudaSummary {
    std::array<double, timed_run_count> host_to_device{};
    std::array<double, timed_run_count> kernel{};
    std::array<double, timed_run_count> device_to_host{};
    std::array<double, timed_run_count> total_gpu_path{};
    std::array<std::array<double, 3>, timed_run_count> samples{};
    Statistics host_to_device_statistics;
    Statistics kernel_statistics;
    Statistics device_to_host_statistics;
    Statistics total_gpu_path_statistics;
    double kernel_million_outputs_per_second = 0.0;
    double total_million_outputs_per_second = 0.0;
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
    return {std::move(array), dtype, element_count, std::filesystem::file_size(path)};
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
    const std::vector<double>& left,
    const std::vector<double>& right)
{
    if (left.size() != right.size()) {
        throw std::runtime_error("Cannot compare outputs with different element counts.");
    }

    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        mismatches += double_bits(left[index]) != double_bits(right[index]);
    }
    return mismatches;
}

std::array<double, 3> extract_samples(
    const std::vector<double>& output,
    const std::array<Coordinate, 3>& coordinates,
    const phase_a::Dimensions4D& dimensions)
{
    std::array<double, 3> samples{};
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        const Coordinate& coordinate = coordinates[index];
        samples[index] = output[phase_a::flat_index(
            dimensions, coordinate[0], coordinate[1], coordinate[2], coordinate[3])];
    }
    return samples;
}

bool samples_match(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right)
{
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (double_bits(left[index]) != double_bits(right[index])) {
            return false;
        }
    }
    return true;
}

Statistics summarize(const std::array<double, timed_run_count>& values)
{
    std::array<double, timed_run_count> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    return {sorted.front(), sorted[sorted.size() / 2], sorted.back()};
}

template <typename FilterCall>
void record_cpu_run(
    FilterCall filter_call,
    const std::array<Coordinate, 3>& coordinates,
    const phase_a::Dimensions4D& dimensions,
    std::size_t run,
    CpuSummary& summary)
{
    const Clock::time_point start = Clock::now();
    std::vector<double> output = filter_call();
    const Clock::time_point end = Clock::now();
    summary.milliseconds[run] =
        std::chrono::duration<double, std::milli>(end - start).count();
    summary.samples[run] = extract_samples(output, coordinates, dimensions);
}

void finish_cpu_summary(CpuSummary& summary, std::size_t output_count)
{
    summary.statistics = summarize(summary.milliseconds);
    summary.million_outputs_per_second =
        static_cast<double>(output_count) / (summary.statistics.median / 1000.0) / 1.0e6;
}

void record_cuda_run(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions,
    const std::array<Coordinate, 3>& coordinates,
    std::size_t run,
    CudaSummary& summary)
{
    const phase_a::CudaFilterResult result =
        phase_a::fixed_median_3x3_cuda_baseline(input, dimensions);
    summary.host_to_device[run] = result.timing.host_to_device;
    summary.kernel[run] = result.timing.kernel;
    summary.device_to_host[run] = result.timing.device_to_host;
    summary.total_gpu_path[run] = result.timing.total_gpu_path;
    summary.samples[run] = extract_samples(result.output, coordinates, dimensions);
}

void finish_cuda_summary(CudaSummary& summary, std::size_t output_count)
{
    summary.host_to_device_statistics = summarize(summary.host_to_device);
    summary.kernel_statistics = summarize(summary.kernel);
    summary.device_to_host_statistics = summarize(summary.device_to_host);
    summary.total_gpu_path_statistics = summarize(summary.total_gpu_path);
    summary.kernel_million_outputs_per_second =
        static_cast<double>(output_count) /
        (summary.kernel_statistics.median / 1000.0) / 1.0e6;
    summary.total_million_outputs_per_second =
        static_cast<double>(output_count) /
        (summary.total_gpu_path_statistics.median / 1000.0) / 1.0e6;
}

void print_cpu_summary(const char* label, const CpuSummary& summary)
{
    std::cout << label << " raw_ms: "
              << summary.milliseconds[0] << ", "
              << summary.milliseconds[1] << ", "
              << summary.milliseconds[2] << '\n'
              << label << " min_median_max_ms: "
              << summary.statistics.minimum << ", "
              << summary.statistics.median << ", "
              << summary.statistics.maximum << '\n'
              << label << " rate_Moutputs_per_s: "
              << summary.million_outputs_per_second << '\n';
}

void print_cuda_component(
    const char* label,
    const std::array<double, timed_run_count>& raw,
    const Statistics& statistics)
{
    std::cout << label << " raw_ms: "
              << raw[0] << ", " << raw[1] << ", " << raw[2] << '\n'
              << label << " min_median_max_ms: "
              << statistics.minimum << ", "
              << statistics.median << ", "
              << statistics.maximum << '\n';
}

int actual_openmp_team_size(int requested_threads)
{
    int actual_threads = 0;
#pragma omp parallel num_threads(requested_threads)
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
    }
    return actual_threads;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 2) {
            throw std::runtime_error(
                "Usage: phase_a_cuda_benchmark.exe [median_filter_input.npy]");
        }

        const std::filesystem::path input_path = std::filesystem::absolute(
            argc == 2 ? std::filesystem::path(argv[1])
                      : std::filesystem::path(PHASE_A_BENCHMARK_INPUT_PATH))
                                                       .lexically_normal();
        const Clock::time_point load_start = Clock::now();
        const LoadedNpy input = load_npy(input_path);
        const Clock::time_point load_end = Clock::now();
        const double load_ms =
            std::chrono::duration<double, std::milli>(load_end - load_start).count();

        const phase_a::Dimensions4D dimensions = dimensions_from_shape(input.array.shape);
        const std::array<Coordinate, 3> coordinates = {{
            {0, 0, 0, 0},
            {dimensions.scan_y / 2, dimensions.scan_x / 2,
             dimensions.detector_y / 2, dimensions.detector_x / 2},
            {dimensions.scan_y - 1, dimensions.scan_x - 1,
             dimensions.detector_y - 1, dimensions.detector_x - 1},
        }};

        omp_set_dynamic(0);
        const int requested_openmp_threads = std::max(
            1,
            std::min(phase_a::openmp_max_threads(), phase_a::openmp_processor_count()));
        const int openmp_threads = actual_openmp_team_size(requested_openmp_threads);
        if (openmp_threads != requested_openmp_threads) {
            throw std::runtime_error("OpenMP runtime did not supply the requested benchmark team.");
        }

        const phase_a::CudaDeviceInfo device = phase_a::cuda_device_info();

        // Exactly one warm-up per path. The retained outputs also support the
        // required full comparisons before any timed performance claim.
        std::vector<double> serial_warmup =
            phase_a::fixed_median_3x3_median9_direct_addressing(input.array.data, dimensions);
        const std::array<double, 3> serial_warmup_samples =
            extract_samples(serial_warmup, coordinates, dimensions);

        std::vector<double> openmp_warmup =
            phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                input.array.data, dimensions, openmp_threads);
        const std::size_t openmp_mismatches =
            count_bitwise_mismatches(serial_warmup, openmp_warmup);
        std::vector<double>().swap(openmp_warmup);

        phase_a::CudaFilterResult cuda_warmup =
            phase_a::fixed_median_3x3_cuda_baseline(input.array.data, dimensions);
        const std::size_t cuda_mismatches =
            count_bitwise_mismatches(serial_warmup, cuda_warmup.output);
        const std::array<double, 3> cuda_warmup_samples =
            extract_samples(cuda_warmup.output, coordinates, dimensions);

        std::cout << "Full elements compared: " << input.element_count << '\n'
                  << "Optimized-serial-vs-OpenMP bitwise mismatches: "
                  << openmp_mismatches << '\n'
                  << "Optimized-serial-vs-CUDA bitwise mismatches: "
                  << cuda_mismatches << '\n';
        if (openmp_mismatches != 0 || cuda_mismatches != 0) {
            std::cout << "Full comparison result: FAIL; timing skipped\n";
            return 1;
        }
        std::cout << "Full comparison result: PASS\n";
        std::vector<double>().swap(serial_warmup);
        std::vector<double>().swap(cuda_warmup.output);

        CpuSummary serial;
        CpuSummary openmp;
        CudaSummary cuda;
        for (std::size_t run = 0; run < timed_run_count; ++run) {
            record_cpu_run(
                [&]() {
                    return phase_a::fixed_median_3x3_median9_direct_addressing(
                        input.array.data, dimensions);
                },
                coordinates,
                dimensions,
                run,
                serial);
            record_cpu_run(
                [&]() {
                    return phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                        input.array.data, dimensions, openmp_threads);
                },
                coordinates,
                dimensions,
                run,
                openmp);
            record_cuda_run(input.array.data, dimensions, coordinates, run, cuda);
        }

        finish_cpu_summary(serial, input.element_count);
        finish_cpu_summary(openmp, input.element_count);
        finish_cuda_summary(cuda, input.element_count);

        bool samples_consistent = samples_match(cuda_warmup_samples, serial_warmup_samples);
        for (std::size_t run = 0; run < timed_run_count; ++run) {
            samples_consistent =
                samples_consistent &&
                samples_match(serial.samples[run], serial_warmup_samples) &&
                samples_match(openmp.samples[run], serial_warmup_samples) &&
                samples_match(cuda.samples[run], cuda_warmup_samples);
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "Project 1 Phase A correctness-first CUDA baseline benchmark\n"
                  << "Input path: " << input_path.string() << '\n'
                  << "Dtype: " << input.dtype << " (float64)\n"
                  << "Shape: " << shape_as_string(input.array.shape) << '\n'
                  << "Storage order: C-contiguous\n"
                  << "Input elements: " << input.element_count << '\n'
                  << "NPY file bytes: " << input.file_bytes << '\n'
                  << "NPY load time_ms: " << load_ms << '\n'
                  << "CUDA device: " << device.name << '\n'
                  << "Compute capability: sm_" << device.compute_major << device.compute_minor << '\n'
                  << "CUDA multiprocessors: " << device.multiprocessor_count << '\n'
                  << "OpenMP threads: " << openmp_threads << '\n'
                  << "Warm-up runs per path: 1\n"
                  << "Timed runs per path: " << timed_run_count << '\n'
                  << "Timed order: optimized serial, OpenMP, CUDA; interleaved by repetition\n";

        print_cpu_summary("Optimized serial", serial);
        print_cpu_summary("Best OpenMP", openmp);
        print_cuda_component(
            "CUDA H2D", cuda.host_to_device, cuda.host_to_device_statistics);
        print_cuda_component(
            "CUDA kernel", cuda.kernel, cuda.kernel_statistics);
        print_cuda_component(
            "CUDA D2H", cuda.device_to_host, cuda.device_to_host_statistics);
        print_cuda_component(
            "CUDA total GPU path", cuda.total_gpu_path, cuda.total_gpu_path_statistics);

        std::cout << "CUDA kernel rate_Moutputs_per_s: "
                  << cuda.kernel_million_outputs_per_second << '\n'
                  << "CUDA total path rate_Moutputs_per_s: "
                  << cuda.total_million_outputs_per_second << '\n'
                  << "CUDA kernel speedup_vs_serial: "
                  << serial.statistics.median / cuda.kernel_statistics.median << "x\n"
                  << "CUDA total path speedup_vs_serial: "
                  << serial.statistics.median / cuda.total_gpu_path_statistics.median << "x\n"
                  << "CUDA kernel speedup_vs_OpenMP: "
                  << openmp.statistics.median / cuda.kernel_statistics.median << "x\n"
                  << "CUDA total path speedup_vs_OpenMP: "
                  << openmp.statistics.median / cuda.total_gpu_path_statistics.median << "x\n"
                  << "GPU event timing includes synchronous H2D, kernel, and synchronous D2H only; "
                     "device/host allocation, event setup, and file I/O are excluded\n"
                  << "CPU filter timing includes internal output allocation and excludes file I/O\n"
                  << "Sanity samples bitwise-consistent across warm-up and timed runs: "
                  << (samples_consistent ? "yes" : "no") << '\n';

        return samples_consistent ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
