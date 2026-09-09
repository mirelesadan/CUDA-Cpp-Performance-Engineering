#include <algorithm>
#include <array>
#include <chrono>
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
constexpr std::size_t pageable_warmup_count = 5;
constexpr std::size_t pageable_timed_count = 20;
constexpr std::size_t residency_warmup_count = 3;
constexpr std::size_t residency_timed_count = 10;
constexpr std::size_t transfer_diagnostic_warmup_count = 5;
constexpr std::size_t transfer_diagnostic_timed_count = 20;
constexpr std::size_t size_cuda_warmup_count = 3;
constexpr std::size_t size_cuda_timed_count = 10;
constexpr std::size_t cpu_timed_count = 3;

using Clock = std::chrono::steady_clock;

struct LoadedNpy {
    npy::npy_data<double> array;
    std::string dtype;
    std::size_t element_count = 0;
};

struct Statistics {
    double minimum = 0.0;
    double median = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double coefficient_of_variation_percent = 0.0;
};

struct CudaPathStatistics {
    Statistics host_to_device;
    Statistics kernel;
    Statistics device_to_host;
    Statistics total;
};

struct CpuTiming {
    int threads = 1;
    std::vector<double> milliseconds;
    Statistics statistics;
};

struct CpuCaseResult {
    CpuTiming serial;
    std::vector<CpuTiming> openmp;
    std::size_t cuda_mismatches = 0;
};

struct SizeResult {
    std::string label;
    phase_a::Dimensions4D dimensions{};
    std::size_t element_count = 0;
    CpuCaseResult cpu;
    CudaPathStatistics cuda;
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

std::size_t checked_element_count(const phase_a::Dimensions4D& dimensions)
{
    const std::array<std::size_t, 4> shape = {
        dimensions.scan_y,
        dimensions.scan_x,
        dimensions.detector_y,
        dimensions.detector_x,
    };
    std::size_t count = 1;
    for (const std::size_t size : shape) {
        if (size == 0 || count > std::numeric_limits<std::size_t>::max() / size) {
            throw std::runtime_error("Invalid synthetic shape.");
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
    if (dtype != expected_dtype || header.shape.size() != expected_dimension_count ||
        header.fortran_order) {
        throw std::runtime_error(
            "Canonical input must be C-order, four-dimensional, little-endian float64.");
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

std::string shape_as_string(const phase_a::Dimensions4D& dimensions)
{
    std::ostringstream text;
    text << '(' << dimensions.scan_y << ", " << dimensions.scan_x << ", "
         << dimensions.detector_y << ", " << dimensions.detector_x << ')';
    return text.str();
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

std::array<std::uint64_t, 3> output_samples(const std::vector<double>& output)
{
    if (output.empty()) {
        throw std::runtime_error("Cannot sample an empty output.");
    }
    return {
        double_bits(output.front()),
        double_bits(output[output.size() / 2]),
        double_bits(output.back()),
    };
}

Statistics summarize(const std::vector<double>& values)
{
    if (values.empty()) {
        throw std::runtime_error("Cannot summarize an empty sequence.");
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
    const double standard_deviation =
        std::sqrt(squared_deviation_sum / static_cast<double>(values.size()));
    return {
        sorted.front(),
        sorted[sorted.size() / 2],
        sorted.back(),
        mean,
        100.0 * standard_deviation / mean,
    };
}

std::vector<double> component_values(
    const std::vector<phase_a::CudaTimingMilliseconds>& runs,
    double phase_a::CudaTimingMilliseconds::* member)
{
    std::vector<double> values;
    values.reserve(runs.size());
    for (const phase_a::CudaTimingMilliseconds& run : runs) {
        values.push_back(run.*member);
    }
    return values;
}

CudaPathStatistics summarize_cuda(
    const std::vector<phase_a::CudaTimingMilliseconds>& runs)
{
    return {
        summarize(component_values(runs, &phase_a::CudaTimingMilliseconds::host_to_device)),
        summarize(component_values(runs, &phase_a::CudaTimingMilliseconds::kernel)),
        summarize(component_values(runs, &phase_a::CudaTimingMilliseconds::device_to_host)),
        summarize(component_values(runs, &phase_a::CudaTimingMilliseconds::total_gpu_path)),
    };
}

template <typename FilterCall>
CpuTiming time_cpu_filter(
    FilterCall filter,
    int threads,
    const std::array<std::uint64_t, 3>& expected_samples)
{
    CpuTiming result;
    result.threads = threads;
    result.milliseconds.reserve(cpu_timed_count);
    for (std::size_t run = 0; run < cpu_timed_count; ++run) {
        const Clock::time_point start = Clock::now();
        std::vector<double> output = filter();
        const Clock::time_point end = Clock::now();
        if (output_samples(output) != expected_samples) {
            throw std::runtime_error("CPU timed-run sanity samples changed.");
        }
        result.milliseconds.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    result.statistics = summarize(result.milliseconds);
    return result;
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

std::vector<int> selected_openmp_counts()
{
    const int available = std::max(
        1,
        std::min(phase_a::openmp_max_threads(), phase_a::openmp_processor_count()));
    std::vector<int> counts;
    for (const int requested : {4, 8, available}) {
        const int count = std::min(requested, available);
        if (std::find(counts.begin(), counts.end(), count) == counts.end()) {
            if (actual_openmp_team_size(count) != count) {
                throw std::runtime_error("OpenMP runtime did not supply the requested team.");
            }
            counts.push_back(count);
        }
    }
    return counts;
}

CpuCaseResult benchmark_cpu_case(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions,
    const std::vector<double>& cuda_output,
    const std::vector<int>& openmp_counts)
{
    CpuCaseResult result;
    std::vector<double> serial_warmup =
        phase_a::fixed_median_3x3_median9_direct_addressing(input, dimensions);
    result.cuda_mismatches = count_bitwise_mismatches(cuda_output, serial_warmup);
    if (result.cuda_mismatches != 0) {
        throw std::runtime_error("CUDA output does not match optimized serial output.");
    }
    const std::array<std::uint64_t, 3> samples = output_samples(serial_warmup);

    for (const int threads : openmp_counts) {
        std::vector<double> openmp_warmup =
            phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                input, dimensions, threads);
        if (count_bitwise_mismatches(openmp_warmup, serial_warmup) != 0) {
            throw std::runtime_error("OpenMP output does not match optimized serial output.");
        }
    }
    std::vector<double>().swap(serial_warmup);

    result.serial = time_cpu_filter(
        [&]() {
            return phase_a::fixed_median_3x3_median9_direct_addressing(input, dimensions);
        },
        1,
        samples);
    for (const int threads : openmp_counts) {
        result.openmp.push_back(time_cpu_filter(
            [&]() {
                return phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                    input, dimensions, threads);
            },
            threads,
            samples));
    }
    return result;
}

const CpuTiming& best_openmp(const CpuCaseResult& result)
{
    return *std::min_element(
        result.openmp.begin(),
        result.openmp.end(),
        [](const CpuTiming& left, const CpuTiming& right) {
            return left.statistics.median < right.statistics.median;
        });
}

std::vector<double> make_synthetic_input(const phase_a::Dimensions4D& dimensions)
{
    const std::size_t count = checked_element_count(dimensions);
    std::vector<double> input(count);
    for (std::size_t index = 0; index < count; ++index) {
        input[index] = static_cast<double>(
            static_cast<long long>((index * 37 + (index / 97) * 13) % 1009) - 504);
    }
    return input;
}

void print_sequence(const std::string& label, const std::vector<double>& values)
{
    std::cout << label << "_ms:";
    for (const double value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}

void print_statistics(const std::string& label, const Statistics& statistics)
{
    std::cout << label << " min_median_max_ms: "
              << statistics.minimum << ", " << statistics.median << ", "
              << statistics.maximum << '\n'
              << label << " mean_ms: " << statistics.mean << '\n'
              << label << " coefficient_of_variation_percent: "
              << statistics.coefficient_of_variation_percent << '\n';
}

void print_cpu_timing(const std::string& label, const CpuTiming& timing)
{
    print_sequence(label, timing.milliseconds);
    print_statistics(label, timing.statistics);
}

std::vector<double> add_sequences(
    const std::vector<double>& left,
    const std::vector<double>& right)
{
    if (left.size() != right.size()) {
        throw std::runtime_error("Cannot add timing sequences with different lengths.");
    }
    std::vector<double> result(left.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] + right[index];
    }
    return result;
}

std::vector<double> transfer_fraction_values(
    const std::vector<phase_a::CudaTimingMilliseconds>& runs)
{
    std::vector<double> fractions;
    fractions.reserve(runs.size());
    for (const phase_a::CudaTimingMilliseconds& run : runs) {
        fractions.push_back(
            100.0 * (run.host_to_device + run.device_to_host) /
            run.total_gpu_path);
    }
    return fractions;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 2) {
            throw std::runtime_error(
                "Usage: phase_a_cuda_transfer_characterization.exe [median_filter_input.npy]");
        }
        const std::filesystem::path input_path = std::filesystem::absolute(
            argc == 2 ? std::filesystem::path(argv[1])
                      : std::filesystem::path(PHASE_A_BENCHMARK_INPUT_PATH))
                                                       .lexically_normal();
        const LoadedNpy canonical = load_npy(input_path);
        const phase_a::Dimensions4D canonical_dimensions =
            dimensions_from_shape(canonical.array.shape);
        const phase_a::CudaDeviceInfo device = phase_a::cuda_device_info();
        omp_set_dynamic(0);
        const std::vector<int> openmp_counts = selected_openmp_counts();
        const std::vector<std::size_t> residency_iterations = {1, 2, 5, 10};

        phase_a::CudaTransferCharacterizationResult characterization =
            phase_a::characterize_fixed_median_3x3_cuda_transfers(
                canonical.array.data,
                canonical_dimensions,
                pageable_warmup_count,
                pageable_timed_count,
                residency_iterations,
                residency_warmup_count,
                residency_timed_count,
                transfer_diagnostic_warmup_count,
                transfer_diagnostic_timed_count);
        const CudaPathStatistics canonical_cuda =
            summarize_cuda(characterization.pageable_path_runs);
        const CpuCaseResult canonical_cpu = benchmark_cpu_case(
            canonical.array.data,
            canonical_dimensions,
            characterization.output,
            openmp_counts);
        const CpuTiming& canonical_best_openmp = best_openmp(canonical_cpu);

        std::vector<SizeResult> size_results;
        const std::vector<std::pair<std::string, phase_a::Dimensions4D>> synthetic_shapes = {
            {"tiny", {5, 5, 8, 8}},
            {"small", {10, 10, 16, 16}},
            {"medium", {20, 20, 32, 32}},
            {"crossover-low", {25, 20, 50, 50}},
            {"crossover-high", {25, 20, 64, 64}},
            {"large", {40, 25, 64, 64}},
            {"larger", {50, 25, 80, 100}},
        };
        for (const auto& item : synthetic_shapes) {
            std::vector<double> input = make_synthetic_input(item.second);
            phase_a::CudaPageablePathBenchmarkResult cuda =
                phase_a::benchmark_fixed_median_3x3_cuda_pageable_path(
                    input,
                    item.second,
                    size_cuda_warmup_count,
                    size_cuda_timed_count);
            CpuCaseResult cpu = benchmark_cpu_case(
                input,
                item.second,
                cuda.output,
                openmp_counts);
            size_results.push_back({
                item.first,
                item.second,
                input.size(),
                std::move(cpu),
                summarize_cuda(cuda.timed_runs),
            });
        }
        size_results.push_back({
            "canonical",
            canonical_dimensions,
            canonical.element_count,
            canonical_cpu,
            canonical_cuda,
        });

        const std::vector<double> pageable_h2d = component_values(
            characterization.pageable_path_runs,
            &phase_a::CudaTimingMilliseconds::host_to_device);
        const std::vector<double> pageable_kernel = component_values(
            characterization.pageable_path_runs,
            &phase_a::CudaTimingMilliseconds::kernel);
        const std::vector<double> pageable_d2h = component_values(
            characterization.pageable_path_runs,
            &phase_a::CudaTimingMilliseconds::device_to_host);
        const std::vector<double> pageable_total = component_values(
            characterization.pageable_path_runs,
            &phase_a::CudaTimingMilliseconds::total_gpu_path);

        std::cout << std::fixed << std::setprecision(6)
                  << "Project 1 Phase A CUDA transfer and residency characterization\n"
                  << "Input path: " << input_path.string() << '\n'
                  << "Dtype: " << canonical.dtype << " (float64)\n"
                  << "Canonical shape: " << shape_as_string(canonical_dimensions) << '\n'
                  << "Canonical elements: " << canonical.element_count << '\n'
                  << "CUDA device: " << device.name << '\n'
                  << "Compute capability: sm_" << device.compute_major << device.compute_minor << '\n'
                  << "Kernel: retained one-output-per-thread baseline, 256 threads/block\n"
                  << "Device allocations and event creation: persistent and outside timing\n"
                  << "Pageable warm-ups/timed runs: " << pageable_warmup_count << "/"
                  << pageable_timed_count << '\n'
                  << "Residency warm-ups/timed runs per count: "
                  << residency_warmup_count << "/" << residency_timed_count << '\n'
                  << "CPU warm-ups/timed runs per path: 1/" << cpu_timed_count << '\n'
                  << "OpenMP candidate thread counts:";
        for (const int threads : openmp_counts) {
            std::cout << ' ' << threads;
        }
        std::cout << '\n';

        print_sequence("Pageable_H2D", pageable_h2d);
        print_sequence("Pageable_kernel", pageable_kernel);
        print_sequence("Pageable_D2H", pageable_d2h);
        print_sequence("Pageable_total", pageable_total);
        print_statistics("Pageable H2D", canonical_cuda.host_to_device);
        print_statistics("Pageable kernel", canonical_cuda.kernel);
        print_statistics("Pageable D2H", canonical_cuda.device_to_host);
        print_statistics("Pageable total", canonical_cuda.total);
        std::cout << "Pageable median transfer_fraction_percent: "
                  << summarize(
                         transfer_fraction_values(
                             characterization.pageable_path_runs))
                         .median
                  << '\n';

        print_cpu_timing("Optimized_serial", canonical_cpu.serial);
        for (const CpuTiming& timing : canonical_cpu.openmp) {
            print_cpu_timing(
                "OpenMP_" + std::to_string(timing.threads),
                timing);
        }
        std::cout << "Canonical CUDA-vs-optimized-serial bitwise mismatches: "
                  << canonical_cpu.cuda_mismatches << '\n'
                  << "Fresh best OpenMP threads: " << canonical_best_openmp.threads << '\n'
                  << "Fresh pageable CUDA speedup_vs_serial: "
                  << canonical_cpu.serial.statistics.median /
                         canonical_cuda.total.median
                  << "x\n"
                  << "Fresh pageable CUDA speedup_vs_best_OpenMP: "
                  << canonical_best_openmp.statistics.median /
                         canonical_cuda.total.median
                  << "x\n";

        for (const phase_a::CudaResidencyBenchmarkResult& residency :
             characterization.residency_results) {
            const CudaPathStatistics statistics = summarize_cuda(residency.timed_runs);
            const std::vector<double> total_values = component_values(
                residency.timed_runs,
                &phase_a::CudaTimingMilliseconds::total_gpu_path);
            const Statistics transfer_fraction =
                summarize(transfer_fraction_values(residency.timed_runs));
            const double effective_ms =
                statistics.total.median /
                static_cast<double>(residency.iteration_count);
            print_sequence(
                "Residency_" + std::to_string(residency.iteration_count) + "_total",
                total_values);
            std::cout << "Residency iterations: " << residency.iteration_count << '\n'
                      << "Residency median H2D/kernel-total/D2H/total_ms: "
                      << statistics.host_to_device.median << ", "
                      << statistics.kernel.median << ", "
                      << statistics.device_to_host.median << ", "
                      << statistics.total.median << '\n'
                      << "Residency effective median_ms_per_filter: " << effective_ms << '\n'
                      << "Residency median transfer_fraction_percent: "
                      << transfer_fraction.median << '\n'
                      << "Residency speedup_vs_serial: "
                      << canonical_cpu.serial.statistics.median / effective_ms << "x\n"
                      << "Residency speedup_vs_best_OpenMP: "
                      << canonical_best_openmp.statistics.median / effective_ms << "x\n";
        }

        print_sequence(
            "Diagnostic_pageable_H2D",
            characterization.diagnostic_pageable_h2d_milliseconds);
        print_sequence(
            "Diagnostic_pageable_D2H",
            characterization.diagnostic_pageable_d2h_milliseconds);
        print_statistics(
            "Diagnostic pageable H2D",
            summarize(characterization.diagnostic_pageable_h2d_milliseconds));
        print_statistics(
            "Diagnostic pageable D2H",
            summarize(characterization.diagnostic_pageable_d2h_milliseconds));
        if (characterization.pinned_memory_error.empty()) {
            print_sequence(
                "Diagnostic_pinned_H2D",
                characterization.diagnostic_pinned_h2d_milliseconds);
            print_sequence(
                "Diagnostic_pinned_D2H",
                characterization.diagnostic_pinned_d2h_milliseconds);
            const Statistics pinned_h2d =
                summarize(characterization.diagnostic_pinned_h2d_milliseconds);
            const Statistics pinned_d2h =
                summarize(characterization.diagnostic_pinned_d2h_milliseconds);
            const Statistics diagnostic_pageable_h2d =
                summarize(characterization.diagnostic_pageable_h2d_milliseconds);
            const Statistics diagnostic_pageable_d2h =
                summarize(characterization.diagnostic_pageable_d2h_milliseconds);
            print_statistics("Diagnostic pinned H2D", pinned_h2d);
            print_statistics("Diagnostic pinned D2H", pinned_d2h);
            const std::vector<double> pageable_transfer_total = add_sequences(
                characterization.diagnostic_pageable_h2d_milliseconds,
                characterization.diagnostic_pageable_d2h_milliseconds);
            const std::vector<double> pinned_transfer_total = add_sequences(
                characterization.diagnostic_pinned_h2d_milliseconds,
                characterization.diagnostic_pinned_d2h_milliseconds);
            const Statistics pageable_transfer = summarize(pageable_transfer_total);
            const Statistics pinned_transfer = summarize(pinned_transfer_total);
            std::cout << "Diagnostic pageable transfer-pair median_ms: "
                      << pageable_transfer.median << '\n'
                      << "Diagnostic pinned transfer-pair median_ms: "
                      << pinned_transfer.median << '\n'
                      << "Pinned transfer-pair speedup: "
                      << pageable_transfer.median / pinned_transfer.median << "x\n"
                      << "Pinned transfer-pair runtime_reduction_percent: "
                      << 100.0 *
                             (pageable_transfer.median - pinned_transfer.median) /
                             pageable_transfer.median
                      << '\n';
        }
        else {
            std::cout << "Pinned-memory diagnostic unavailable: "
                      << characterization.pinned_memory_error << '\n';
        }

        std::cout << "Crossover results (median milliseconds)\n";
        for (const SizeResult& size : size_results) {
            const CpuTiming& best = best_openmp(size.cpu);
            const double serial_ms = size.cpu.serial.statistics.median;
            const double openmp_ms = best.statistics.median;
            const double cuda_ms = size.cuda.total.median;
            const char* preferred = serial_ms <= openmp_ms && serial_ms <= cuda_ms
                ? "serial"
                : (openmp_ms <= cuda_ms ? "OpenMP" : "CUDA");
            std::cout << "Crossover case: " << size.label
                      << "; shape=" << shape_as_string(size.dimensions)
                      << "; elements=" << size.element_count
                      << "; serial_ms=" << serial_ms
                      << "; best_OpenMP_threads=" << best.threads
                      << "; best_OpenMP_ms=" << openmp_ms
                      << "; pageable_CUDA_total_ms=" << cuda_ms
                      << "; preferred=" << preferred
                      << "; CUDA_mismatches=" << size.cpu.cuda_mismatches
                      << '\n';
        }
        std::cout << "Validation result: PASS\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
