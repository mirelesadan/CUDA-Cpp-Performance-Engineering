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

struct TimingSummary {
    std::array<double, timed_run_count> run_ms{};
    std::array<std::array<double, 3>, timed_run_count> samples{};
    double minimum_ms = 0.0;
    double median_ms = 0.0;
    double maximum_ms = 0.0;
    double million_outputs_per_second = 0.0;
};

struct ParallelResult {
    int requested_threads = 0;
    int actual_threads = 0;
    std::array<double, 3> warmup_samples{};
    TimingSummary timing;
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
        const std::size_t dimension_size = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / dimension_size) {
            throw std::runtime_error("NPY contract error: shape product exceeds std::size_t.");
        }
        count *= dimension_size;
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
        throw std::runtime_error(
            "Expected 4 dimensions, found " + std::to_string(header.shape.size()) + '.');
    }
    if (header.fortran_order) {
        throw std::runtime_error("Expected C order, found Fortran order.");
    }
    const std::size_t element_count = checked_element_count(header.shape);

    input.clear();
    input.seekg(0, std::ios::beg);
    npy::npy_data<double> array = npy::read_npy<double>(input);
    if (!input || array.data.size() != element_count || array.shape != header.shape || array.fortran_order) {
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

double elapsed_milliseconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
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
    const std::array<double, 3>& actual,
    const std::array<double, 3>& expected)
{
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (double_bits(actual[index]) != double_bits(expected[index])) {
            return false;
        }
    }
    return true;
}

std::size_t count_bitwise_mismatches(
    const std::vector<double>& baseline,
    const std::vector<double>& candidate)
{
    if (baseline.size() != candidate.size()) {
        throw std::runtime_error("Cannot compare outputs with different element counts.");
    }

    std::size_t mismatch_count = 0;
    for (std::size_t index = 0; index < baseline.size(); ++index) {
        mismatch_count += double_bits(baseline[index]) != double_bits(candidate[index]);
    }
    return mismatch_count;
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

std::vector<int> default_thread_counts(int maximum_threads)
{
    std::vector<int> counts;
    for (int count = 1; count < maximum_threads; count *= 2) {
        counts.push_back(count);
        if (count > std::numeric_limits<int>::max() / 2) {
            break;
        }
    }
    counts.push_back(maximum_threads);
    return counts;
}

std::vector<int> parse_thread_counts(const std::string& specification, int maximum_threads)
{
    std::vector<int> counts;
    std::istringstream stream(specification);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            throw std::runtime_error("Thread list contains an empty entry.");
        }
        std::size_t consumed = 0;
        const int count = std::stoi(token, &consumed);
        if (consumed != token.size() || count < 1 || count > maximum_threads) {
            throw std::runtime_error(
                "Each thread count must be between 1 and " + std::to_string(maximum_threads) + '.');
        }
        if (std::find(counts.begin(), counts.end(), count) == counts.end()) {
            counts.push_back(count);
        }
    }
    if (counts.empty()) {
        throw std::runtime_error("At least one OpenMP thread count is required.");
    }
    return counts;
}

template <typename FilterCall>
void record_timed_run(
    FilterCall filter_call,
    const std::array<Coordinate, 3>& coordinates,
    const phase_a::Dimensions4D& dimensions,
    std::size_t run,
    TimingSummary& summary)
{
    const Clock::time_point filter_start = Clock::now();
    std::vector<double> output = filter_call();
    const Clock::time_point filter_end = Clock::now();

    summary.run_ms[run] = elapsed_milliseconds(filter_start, filter_end);
    summary.samples[run] = extract_samples(output, coordinates, dimensions);
}

void finish_summary(TimingSummary& summary, std::size_t output_count)
{
    std::array<double, timed_run_count> sorted_ms = summary.run_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    summary.minimum_ms = sorted_ms.front();
    summary.median_ms = sorted_ms[sorted_ms.size() / 2];
    summary.maximum_ms = sorted_ms.back();
    summary.million_outputs_per_second =
        static_cast<double>(output_count) / (summary.median_ms / 1000.0) / 1.0e6;
}

void print_summary(
    const std::string& label,
    const TimingSummary& summary,
    double serial_median_ms,
    int efficiency_threads)
{
    const double speedup = serial_median_ms / summary.median_ms;
    std::cout << label << " timed run 1_ms: " << summary.run_ms[0] << '\n'
              << label << " timed run 2_ms: " << summary.run_ms[1] << '\n'
              << label << " timed run 3_ms: " << summary.run_ms[2] << '\n'
              << label << " minimum_ms: " << summary.minimum_ms << '\n'
              << label << " median_ms: " << summary.median_ms << '\n'
              << label << " maximum_ms: " << summary.maximum_ms << '\n'
              << label << " output rate_Moutputs_per_s: " << summary.million_outputs_per_second << '\n'
              << label << " speedup_vs_serial: " << speedup << "x\n"
              << label << " parallel_efficiency: "
              << speedup / static_cast<double>(efficiency_threads) << '\n';
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        std::filesystem::path input_path = PHASE_A_BENCHMARK_INPUT_PATH;
        std::string thread_specification;
        bool input_path_set = false;

        for (int argument = 1; argument < argc; ++argument) {
            const std::string value = argv[argument];
            if (value == "--threads") {
                if (++argument >= argc) {
                    throw std::runtime_error("--threads requires a comma-separated value.");
                }
                thread_specification = argv[argument];
            }
            else if (value.rfind("--threads=", 0) == 0) {
                thread_specification = value.substr(std::string("--threads=").size());
            }
            else if (!input_path_set) {
                input_path = value;
                input_path_set = true;
            }
            else {
                throw std::runtime_error(
                    "Usage: phase_a_baseline_benchmark.exe [input.npy] [--threads 1,2,4,...]");
            }
        }
        input_path = std::filesystem::absolute(input_path).lexically_normal();

        omp_set_dynamic(0);
        const int openmp_max_threads = phase_a::openmp_max_threads();
        const int openmp_processors = phase_a::openmp_processor_count();
        const int available_threads = std::max(1, std::min(openmp_max_threads, openmp_processors));
        const std::vector<int> thread_counts = thread_specification.empty()
            ? default_thread_counts(available_threads)
            : parse_thread_counts(thread_specification, available_threads);

        const Clock::time_point load_start = Clock::now();
        const LoadedNpy input = load_npy(input_path);
        const Clock::time_point load_end = Clock::now();
        const double load_ms = elapsed_milliseconds(load_start, load_end);

        const phase_a::Dimensions4D dimensions = dimensions_from_shape(input.array.shape);
        const std::array<Coordinate, 3> coordinates = {{
            {0, 0, 0, 0},
            {dimensions.scan_y / 2, dimensions.scan_x / 2,
             dimensions.detector_y / 2, dimensions.detector_x / 2},
            {dimensions.scan_y - 1, dimensions.scan_x - 1,
             dimensions.detector_y - 1, dimensions.detector_x - 1},
        }};

        std::vector<double> serial_warmup =
            phase_a::fixed_median_3x3_median9_direct_addressing(input.array.data, dimensions);
        const std::array<double, 3> serial_warmup_samples =
            extract_samples(serial_warmup, coordinates, dimensions);

        std::vector<ParallelResult> parallel_results;
        parallel_results.reserve(thread_counts.size());
        std::vector<double> validation_output;
        for (const int requested_threads : thread_counts) {
            const int actual_threads = actual_openmp_team_size(requested_threads);
            if (actual_threads != requested_threads) {
                throw std::runtime_error(
                    "OpenMP runtime supplied " + std::to_string(actual_threads) +
                    " threads when " + std::to_string(requested_threads) + " were requested.");
            }

            std::vector<double> warmup =
                phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                    input.array.data, dimensions, requested_threads);
            parallel_results.push_back({
                requested_threads,
                actual_threads,
                extract_samples(warmup, coordinates, dimensions),
                {},
            });
            if (requested_threads == thread_counts.back()) {
                validation_output = std::move(warmup);
            }
        }

        const std::size_t canonical_mismatches =
            count_bitwise_mismatches(serial_warmup, validation_output);
        std::cout << "Input elements compared: " << input.element_count << '\n'
                  << "Input serial-vs-OpenMP bitwise mismatches: " << canonical_mismatches << '\n'
                  << "OpenMP validation threads: " << thread_counts.back() << '\n';
        if (canonical_mismatches != 0) {
            std::cout << "Input comparison result: FAIL; timing skipped\n";
            return 1;
        }
        std::cout << "Input comparison result: PASS\n";
        std::vector<double>().swap(serial_warmup);
        std::vector<double>().swap(validation_output);

        TimingSummary serial;
        for (std::size_t run = 0; run < timed_run_count; ++run) {
            record_timed_run(
                [&]() {
                    return phase_a::fixed_median_3x3_median9_direct_addressing(
                        input.array.data, dimensions);
                },
                coordinates,
                dimensions,
                run,
                serial);

            for (ParallelResult& result : parallel_results) {
                record_timed_run(
                    [&]() {
                        return phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                            input.array.data, dimensions, result.requested_threads);
                    },
                    coordinates,
                    dimensions,
                    run,
                    result.timing);
            }
        }
        finish_summary(serial, input.element_count);
        for (ParallelResult& result : parallel_results) {
            finish_summary(result.timing, input.element_count);
        }

        bool samples_consistent = true;
        for (std::size_t run = 0; run < timed_run_count; ++run) {
            samples_consistent =
                samples_consistent && samples_match(serial.samples[run], serial_warmup_samples);
        }
        for (const ParallelResult& result : parallel_results) {
            samples_consistent =
                samples_consistent && samples_match(result.warmup_samples, serial_warmup_samples);
            for (std::size_t run = 0; run < timed_run_count; ++run) {
                samples_consistent =
                    samples_consistent && samples_match(result.timing.samples[run], result.warmup_samples);
            }
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "Project 1 Phase A portable OpenMP scaling benchmark\n"
                  << "Input path: " << input_path.string() << '\n'
                  << "Dtype: " << input.dtype << " (float64)\n"
                  << "Shape: " << shape_as_string(input.array.shape) << '\n'
                  << "Storage order: C-contiguous\n"
                  << "Input elements: " << input.element_count << '\n'
                  << "NPY file bytes: " << input.file_bytes << '\n'
                  << "NPY load time_ms: " << load_ms << '\n'
                  << "OpenMP processor count: " << openmp_processors << '\n'
                  << "OpenMP maximum threads: " << openmp_max_threads << '\n'
                  << "OpenMP dynamic teams: disabled\n"
                  << "Warm-up runs per configuration: 1\n"
                  << "Timed runs per configuration: " << timed_run_count << '\n'
                  << "Timed order: serial then ascending OpenMP counts, interleaved by repetition\n";

        print_summary("Optimized serial", serial, serial.median_ms, 1);
        for (const ParallelResult& result : parallel_results) {
            const std::string label =
                "OpenMP " + std::to_string(result.actual_threads) + " thread";
            print_summary(label, result.timing, serial.median_ms, result.actual_threads);
        }

        std::cout << "Timing boundary: filter call only; internal output allocation included; file I/O excluded\n"
                  << "Sanity samples bitwise-consistent across warm-ups and timed runs: "
                  << (samples_consistent ? "yes" : "no") << '\n';

        return samples_consistent ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
