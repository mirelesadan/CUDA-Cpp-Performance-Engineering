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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "adaptive_median.hpp"
#include "adaptive_median_detail.hpp"
#include "npy.hpp"

#ifndef PHASE_B_BENCHMARK_INPUT_PATH
#error "PHASE_B_BENCHMARK_INPUT_PATH must be provided by CMake."
#endif

namespace {

constexpr std::size_t expected_dimension_count = 4;
constexpr const char* expected_dtype = "<f8";
constexpr std::size_t default_timed_runs = 7;
constexpr std::size_t scaling_timed_runs = 5;

struct LoadedNpy {
    npy::npy_data<double> array;
    std::size_t element_count;
};

struct Slice {
    std::size_t start;
    std::size_t count;
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

LoadedNpy load_npy(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open benchmark input: " + path.string());
    }

    const npy::header_t header = npy::parse_header(npy::read_header(input));
    if (header.dtype.str() != expected_dtype) {
        throw std::runtime_error("Benchmark input must use little-endian float64.");
    }
    if (header.shape.size() != expected_dimension_count) {
        throw std::runtime_error("Benchmark input must have exactly four dimensions.");
    }
    if (header.fortran_order) {
        throw std::runtime_error("Benchmark input must be C-contiguous.");
    }
    const std::size_t element_count = checked_element_count(header.shape);

    input.clear();
    input.seekg(0, std::ios::beg);
    npy::npy_data<double> array = npy::read_npy<double>(input);
    if (!input || array.data.size() != element_count ||
        array.shape != header.shape || array.fortran_order) {
        throw std::runtime_error("Benchmark payload does not match its header.");
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

Slice centered_slice(std::size_t length, std::size_t maximum_count)
{
    const std::size_t count = std::min(length, maximum_count);
    return {(length - count) / 2, count};
}

std::vector<double> extract_historical_profile_subset(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& source_dimensions,
    phase_a::Dimensions4D& subset_dimensions,
    std::array<Slice, 4>& slices)
{
    slices = {
        centered_slice(source_dimensions.scan_y, 64),
        centered_slice(source_dimensions.scan_x, 64),
        centered_slice(source_dimensions.detector_y, 8),
        centered_slice(source_dimensions.detector_x, 8),
    };
    subset_dimensions = {
        slices[0].count,
        slices[1].count,
        slices[2].count,
        slices[3].count,
    };
    if (subset_dimensions.scan_y < 7 || subset_dimensions.scan_x < 7) {
        throw std::runtime_error("Benchmark subset cannot support the sMax=7 contract.");
    }

    const std::size_t subset_count =
        subset_dimensions.scan_y * subset_dimensions.scan_x *
        subset_dimensions.detector_y * subset_dimensions.detector_x;
    std::vector<double> subset(subset_count);
    for (std::size_t scan_y = 0; scan_y < subset_dimensions.scan_y; ++scan_y) {
        for (std::size_t scan_x = 0; scan_x < subset_dimensions.scan_x; ++scan_x) {
            for (std::size_t detector_y = 0; detector_y < subset_dimensions.detector_y;
                 ++detector_y) {
                for (std::size_t detector_x = 0; detector_x < subset_dimensions.detector_x;
                     ++detector_x) {
                    subset[phase_a::flat_index(
                        subset_dimensions, scan_y, scan_x, detector_y, detector_x)] =
                        input[phase_a::flat_index(
                            source_dimensions,
                            scan_y + slices[0].start,
                            scan_x + slices[1].start,
                            detector_y + slices[2].start,
                            detector_x + slices[3].start)];
                }
            }
        }
    }
    return subset;
}

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

double percentage(std::size_t value, std::size_t total)
{
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(value) / static_cast<double>(total);
}

struct TimingSummary {
    double median_ms;
    double minimum_ms;
    double maximum_ms;
    double mean_ms;
    double coefficient_of_variation;
};

struct ParallelResult {
    int requested_threads;
    int actual_threads;
    std::size_t bitwise_mismatches;
    std::vector<double> timings_ms;
    TimingSummary summary{};
};

TimingSummary summarize_timings(const std::vector<double>& timings_ms)
{
    std::vector<double> sorted = timings_ms;
    std::sort(sorted.begin(), sorted.end());
    const double mean_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) /
        static_cast<double>(timings_ms.size());
    double squared_deviation_sum = 0.0;
    for (const double timing : timings_ms) {
        const double deviation = timing - mean_ms;
        squared_deviation_sum += deviation * deviation;
    }
    return {
        sorted[sorted.size() / 2],
        sorted.front(),
        sorted.back(),
        mean_ms,
        100.0 * std::sqrt(squared_deviation_sum / timings_ms.size()) / mean_ms,
    };
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

std::vector<int> scaling_thread_counts(int available_threads)
{
    std::vector<int> counts;
    for (const int count : {1, 2, 4, 8, 16, 20}) {
        if (count <= available_threads) {
            counts.push_back(count);
        }
    }
    return counts;
}

std::size_t count_bitwise_mismatches(
    const std::vector<double>& left,
    const std::vector<double>& right)
{
    if (left.size() != right.size()) {
        return std::max(left.size(), right.size());
    }
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        mismatches += double_bits(left[index]) != double_bits(right[index]);
    }
    return mismatches;
}

std::size_t count_statistic_mismatches(
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

void print_timing_results(
    const char* label,
    const std::vector<double>& timings_ms,
    const TimingSummary& summary,
    std::size_t output_count)
{
    std::cout << label << " raw times (ms):";
    for (const double timing : timings_ms) {
        std::cout << ' ' << timing;
    }
    std::cout << '\n'
              << label << " median/min/max (ms): " << summary.median_ms << " / "
              << summary.minimum_ms << " / " << summary.maximum_ms << '\n'
              << label << " mean (ms): " << summary.mean_ms << '\n'
              << label << " CV (%): " << summary.coefficient_of_variation << '\n'
              << label << " throughput (Moutput/s): "
              << static_cast<double>(output_count) / (summary.median_ms * 1000.0) << '\n';
}

std::size_t parse_positive_count(const std::string& text)
{
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char character) {
            return character >= '0' && character <= '9';
        })) {
        throw std::runtime_error("Run count must contain only decimal digits.");
    }
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(text, &consumed);
    if (consumed != text.size() || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Run count must be a positive std::size_t value.");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        std::filesystem::path input_path = PHASE_B_BENCHMARK_INPUT_PATH;
        bool full_scaling_workload = false;
        enum class ProfileImplementation { none, heap, stack, specialized, direct_gather };
        ProfileImplementation profile_implementation = ProfileImplementation::none;
        std::size_t profile_runs = 0;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string value = argv[argument];
            if (value == "--profile-runs" || value == "--profile-stack-runs" ||
                value == "--profile-specialized-runs" ||
                value == "--profile-direct-gather-runs") {
                if (++argument >= argc || profile_implementation != ProfileImplementation::none) {
                    throw std::runtime_error(
                        "A profiling option requires one positive count and may appear once.");
                }
                profile_runs = parse_positive_count(argv[argument]);
                if (value == "--profile-runs") {
                    profile_implementation = ProfileImplementation::heap;
                }
                else if (value == "--profile-stack-runs") {
                    profile_implementation = ProfileImplementation::stack;
                }
                else if (value == "--profile-specialized-runs") {
                    profile_implementation = ProfileImplementation::specialized;
                }
                else {
                    profile_implementation = ProfileImplementation::direct_gather;
                }
            }
            else if (value == "--full-scaling") {
                full_scaling_workload = true;
            }
            else if (input_path != std::filesystem::path(PHASE_B_BENCHMARK_INPUT_PATH)) {
                throw std::runtime_error(
                    "Usage: phase_b_adaptive_median_benchmark.exe [input.npy] "
                    "[--full-scaling] "
                    "[--profile-runs N | --profile-stack-runs N | "
                    "--profile-specialized-runs N | --profile-direct-gather-runs N]");
            }
            else {
                input_path = value;
            }
        }
        input_path = std::filesystem::absolute(input_path).lexically_normal();

        LoadedNpy loaded = load_npy(input_path);
        const phase_a::Dimensions4D source_dimensions = dimensions_from_shape(loaded.array.shape);
        phase_a::Dimensions4D dimensions{};
        std::array<Slice, 4> slices{};
        std::vector<double> input;
        if (full_scaling_workload) {
            dimensions = source_dimensions;
            slices = {{{0, dimensions.scan_y}, {0, dimensions.scan_x},
                       {0, dimensions.detector_y}, {0, dimensions.detector_x}}};
            input = std::move(loaded.array.data);
        }
        else {
            input = extract_historical_profile_subset(
                loaded.array.data, source_dimensions, dimensions, slices);
        }
        const std::vector<double> original_input = input;

        if (profile_implementation != ProfileImplementation::none) {
            std::vector<double> profile_output;
            const auto run_profile_filter = [&] {
                if (profile_implementation == ProfileImplementation::heap) {
                    return phase_b::adaptive_median_s3_smax7(input, dimensions);
                }
                if (profile_implementation == ProfileImplementation::stack) {
                    return phase_b::adaptive_median_s3_smax7_stack(input, dimensions);
                }
                if (profile_implementation == ProfileImplementation::specialized) {
                    return phase_b::adaptive_median_s3_smax7_specialized_3x3(input, dimensions);
                }
                return phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(
                    input, dimensions);
            };
            profile_output = run_profile_filter();
            volatile double checksum = 0.0;
            for (std::size_t run = 0; run < profile_runs; ++run) {
                profile_output = run_profile_filter();
                checksum += profile_output[run % profile_output.size()];
            }
            const char* implementation =
                profile_implementation == ProfileImplementation::heap ? "heap" :
                profile_implementation == ProfileImplementation::stack ? "stack" :
                profile_implementation == ProfileImplementation::specialized ?
                    "specialized-3x3" : "direct-3x3-gather";
            std::cout << "Phase B adaptive profiling workload complete\n"
                      << "Implementation: " << implementation << '\n'
                      << "Profile filter calls: " << profile_runs << '\n'
                      << "Checksum: " << checksum << '\n';
            return 0;
        }

        if (full_scaling_workload) {
            omp_set_dynamic(0);
            const int openmp_max_threads = phase_a::openmp_max_threads();
            const int openmp_processors = phase_a::openmp_processor_count();
            const int available_threads = std::max(
                1, std::min(openmp_max_threads, openmp_processors));
            const std::vector<int> thread_counts = scaling_thread_counts(available_threads);
            if (thread_counts.empty()) {
                throw std::runtime_error("No requested OpenMP thread count is available.");
            }

            std::vector<double> serial_output =
                phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(
                    input, dimensions);
            std::vector<ParallelResult> parallel_results;
            parallel_results.reserve(thread_counts.size());
            bool correctness_passed = true;
            for (const int requested_threads : thread_counts) {
                const int actual_threads = actual_openmp_team_size(requested_threads);
                if (actual_threads != requested_threads) {
                    throw std::runtime_error(
                        "OpenMP runtime supplied " + std::to_string(actual_threads) +
                        " threads when " + std::to_string(requested_threads) +
                        " were requested.");
                }
                const std::vector<double> openmp_output =
                    phase_b::adaptive_median_s3_smax7_openmp(
                        input, dimensions, requested_threads);
                const std::size_t mismatches =
                    count_bitwise_mismatches(serial_output, openmp_output);
                correctness_passed = correctness_passed && mismatches == 0;
                parallel_results.push_back(
                    {requested_threads, actual_threads, mismatches, {}, {}});
            }

            phase_b::AdaptiveMedianDiagnosticResult serial_diagnostic =
                phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
                    input, dimensions);
            phase_b::AdaptiveMedianDiagnosticResult openmp_diagnostic =
                phase_b::adaptive_median_s3_smax7_openmp_diagnostics(
                    input, dimensions, thread_counts.back());
            const std::size_t serial_diagnostic_mismatches =
                count_bitwise_mismatches(serial_output, serial_diagnostic.output);
            const std::size_t openmp_diagnostic_mismatches =
                count_bitwise_mismatches(serial_output, openmp_diagnostic.output);
            const std::size_t statistic_mismatches = count_statistic_mismatches(
                serial_diagnostic.statistics, openmp_diagnostic.statistics);
            const phase_b::AdaptiveMedianStatistics statistics =
                serial_diagnostic.statistics;
            correctness_passed = correctness_passed &&
                serial_diagnostic_mismatches == 0 &&
                openmp_diagnostic_mismatches == 0 &&
                statistic_mismatches == 0;
            std::vector<double>().swap(serial_output);
            std::vector<double>().swap(serial_diagnostic.output);
            std::vector<double>().swap(openmp_diagnostic.output);

            std::vector<double> serial_timings_ms;
            serial_timings_ms.reserve(scaling_timed_runs);
            for (ParallelResult& result : parallel_results) {
                result.timings_ms.reserve(scaling_timed_runs);
            }
            const auto measure_filter = [](const auto& filter) {
                const auto start = std::chrono::steady_clock::now();
                const std::vector<double> output = filter();
                const auto stop = std::chrono::steady_clock::now();
                volatile double checksum = output[output.size() / 2];
                static_cast<void>(checksum);
                return std::chrono::duration<double, std::milli>(stop - start).count();
            };

            for (std::size_t run = 0; run < scaling_timed_runs; ++run) {
                serial_timings_ms.push_back(measure_filter([&] {
                    return phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(
                        input, dimensions);
                }));
                for (ParallelResult& result : parallel_results) {
                    result.timings_ms.push_back(measure_filter([&] {
                        return phase_b::adaptive_median_s3_smax7_openmp(
                            input, dimensions, result.requested_threads);
                    }));
                }
            }

            const TimingSummary serial_summary = summarize_timings(serial_timings_ms);
            for (ParallelResult& result : parallel_results) {
                result.summary = summarize_timings(result.timings_ms);
            }
            const std::size_t input_changes = count_bitwise_mismatches(input, original_input);
            correctness_passed = correctness_passed && input_changes == 0;

            std::cout << std::fixed << std::setprecision(6)
                      << "Project 1 Phase B portable OpenMP scaling benchmark\n"
                      << "Input path: " << input_path.string() << '\n'
                      << "Benchmark shape: (" << dimensions.scan_y << ", "
                      << dimensions.scan_x << ", " << dimensions.detector_y << ", "
                      << dimensions.detector_x << ")\n"
                      << "Output elements: " << input.size() << '\n'
                      << "Detector-coordinate planes: "
                      << dimensions.detector_y * dimensions.detector_x << '\n'
                      << "OpenMP processor count: " << openmp_processors << '\n'
                      << "OpenMP maximum threads: " << openmp_max_threads << '\n'
                      << "OpenMP dynamic teams: disabled\n"
                      << "Warm-up filter calls per implementation: 1\n"
                      << "Timed filter calls per implementation: "
                      << scaling_timed_runs << '\n'
                      << "Timed order: serial then ascending OpenMP counts per repetition\n";
            print_timing_results(
                "Optimized serial", serial_timings_ms, serial_summary, input.size());
            for (const ParallelResult& result : parallel_results) {
                const std::string label =
                    "OpenMP " + std::to_string(result.actual_threads) + " thread";
                print_timing_results(
                    label.c_str(), result.timings_ms, result.summary, input.size());
                const double speedup = serial_summary.median_ms / result.summary.median_ms;
                std::cout << label << " speedup vs serial: " << speedup << "x\n"
                          << label << " parallel efficiency (%): "
                          << 100.0 * speedup / result.actual_threads << '\n'
                          << label << " bitwise mismatches: "
                          << result.bitwise_mismatches << '\n';
            }
            std::cout << "Serial diagnostic-vs-normal mismatches: "
                      << serial_diagnostic_mismatches << '\n'
                      << "OpenMP diagnostic-vs-serial mismatches: "
                      << openmp_diagnostic_mismatches << '\n'
                      << "Serial-vs-OpenMP statistic field mismatches: "
                      << statistic_mismatches << '\n'
                      << "Input bitwise changes: " << input_changes << '\n'
                      << "Finished at 3x3: " << statistics.finished_at_3x3 << " ("
                      << percentage(statistics.finished_at_3x3, statistics.total_outputs)
                      << "%)\n"
                      << "Expanded to 5x5: " << statistics.expanded_to_5x5 << " ("
                      << percentage(statistics.expanded_to_5x5, statistics.total_outputs)
                      << "%)\n"
                      << "Median computations: " << statistics.median_computations << '\n'
                      << "Validation result: "
                      << (correctness_passed ? "PASS" : "FAIL") << '\n';
            return correctness_passed ? 0 : 1;
        }

        std::vector<double> baseline_output =
            phase_b::adaptive_median_s3_smax7_specialized_3x3(input, dimensions);
        std::vector<double> candidate_output =
            phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(input, dimensions);
        std::vector<double> baseline_timings_ms;
        std::vector<double> candidate_timings_ms;
        baseline_timings_ms.reserve(default_timed_runs);
        candidate_timings_ms.reserve(default_timed_runs);

        const auto time_baseline = [&] {
            const auto start = std::chrono::steady_clock::now();
            std::vector<double> run_output =
                phase_b::adaptive_median_s3_smax7_specialized_3x3(input, dimensions);
            const auto stop = std::chrono::steady_clock::now();
            baseline_timings_ms.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
            baseline_output = std::move(run_output);
        };
        const auto time_candidate = [&] {
            const auto start = std::chrono::steady_clock::now();
            std::vector<double> run_output =
                phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(
                    input, dimensions);
            const auto stop = std::chrono::steady_clock::now();
            candidate_timings_ms.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
            candidate_output = std::move(run_output);
        };

        for (std::size_t run = 0; run < default_timed_runs; ++run) {
            if (run % 2 == 0) {
                time_baseline();
                time_candidate();
            }
            else {
                time_candidate();
                time_baseline();
            }
        }

        const TimingSummary baseline_summary = summarize_timings(baseline_timings_ms);
        const TimingSummary candidate_summary = summarize_timings(candidate_timings_ms);
        const phase_b::AdaptiveMedianDiagnosticResult baseline_diagnostic =
            phase_b::adaptive_median_s3_smax7_specialized_3x3_diagnostics(input, dimensions);
        const phase_b::AdaptiveMedianDiagnosticResult candidate_diagnostic =
            phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
                input, dimensions);
        omp_set_dynamic(0);
        const int validation_available_threads = std::max(
            1,
            std::min(phase_a::openmp_max_threads(), phase_a::openmp_processor_count()));
        const std::vector<int> validation_thread_counts =
            scaling_thread_counts(validation_available_threads);
        std::vector<std::pair<int, std::size_t>> openmp_validation_mismatches;
        for (const int threads : validation_thread_counts) {
            const std::vector<double> openmp_output = phase_b::adaptive_median_s3_smax7_openmp(
                input, dimensions, threads);
            openmp_validation_mismatches.emplace_back(
                threads, count_bitwise_mismatches(candidate_output, openmp_output));
        }
        const phase_b::AdaptiveMedianDiagnosticResult openmp_diagnostic =
            phase_b::adaptive_median_s3_smax7_openmp_diagnostics(
                input, dimensions, validation_thread_counts.back());
        const std::size_t openmp_diagnostic_mismatches =
            count_bitwise_mismatches(candidate_output, openmp_diagnostic.output);
        const std::size_t openmp_statistic_mismatches = count_statistic_mismatches(
            candidate_diagnostic.statistics, openmp_diagnostic.statistics);
        const bool openmp_validation_passed = std::all_of(
            openmp_validation_mismatches.begin(),
            openmp_validation_mismatches.end(),
            [](const std::pair<int, std::size_t>& result) { return result.second == 0; });
        const std::size_t baseline_candidate_mismatches =
            count_bitwise_mismatches(baseline_output, candidate_output);
        const std::size_t baseline_diagnostic_mismatches =
            count_bitwise_mismatches(baseline_output, baseline_diagnostic.output);
        const std::size_t candidate_diagnostic_mismatches =
            count_bitwise_mismatches(candidate_output, candidate_diagnostic.output);
        const std::size_t statistic_mismatches = count_statistic_mismatches(
            baseline_diagnostic.statistics, candidate_diagnostic.statistics);
        const std::size_t input_changes = count_bitwise_mismatches(input, original_input);

        const phase_b::AdaptiveMedianStatistics& statistics = baseline_diagnostic.statistics;
        const double speedup = baseline_summary.median_ms / candidate_summary.median_ms;
        const double runtime_reduction =
            100.0 * (baseline_summary.median_ms - candidate_summary.median_ms) /
            baseline_summary.median_ms;
        std::cout << std::fixed << std::setprecision(6)
                  << "Project 1 Phase B direct padded-row 3x3 gathering A/B benchmark\n"
                  << "Input path: " << input_path.string() << '\n'
                  << "Source shape: (" << source_dimensions.scan_y << ", "
                  << source_dimensions.scan_x << ", " << source_dimensions.detector_y
                  << ", " << source_dimensions.detector_x << ")\n"
                  << "Centered slices: [" << slices[0].start << ":"
                  << slices[0].start + slices[0].count << ", " << slices[1].start << ":"
                  << slices[1].start + slices[1].count << ", " << slices[2].start << ":"
                  << slices[2].start + slices[2].count << ", " << slices[3].start << ":"
                  << slices[3].start + slices[3].count << "]\n"
                  << "Benchmark shape: (" << dimensions.scan_y << ", " << dimensions.scan_x
                  << ", " << dimensions.detector_y << ", " << dimensions.detector_x << ")\n"
                  << "Output elements: " << input.size() << '\n'
                  << "Warm-up filter calls per implementation: 1\n"
                  << "Timed filter calls per implementation: "
                  << baseline_timings_ms.size() << '\n';
        print_timing_results(
            "Specialized 3x3 baseline",
            baseline_timings_ms,
            baseline_summary,
            input.size());
        print_timing_results(
            "Direct padded-row 3x3 candidate",
            candidate_timings_ms,
            candidate_summary,
            input.size());
        std::cout << "Candidate speedup: " << speedup << "x\n"
                  << "Candidate runtime reduction (%): " << runtime_reduction << '\n'
                  << "Baseline-vs-candidate bitwise mismatches: "
                  << baseline_candidate_mismatches << '\n'
                  << "Baseline diagnostic-vs-timed bitwise mismatches: "
                  << baseline_diagnostic_mismatches << '\n'
                  << "Candidate diagnostic-vs-timed bitwise mismatches: "
                  << candidate_diagnostic_mismatches << '\n'
                  << "Adaptive-statistic field mismatches: " << statistic_mismatches << '\n'
                  << "OpenMP validation mismatches by thread count:";
        for (const auto& result : openmp_validation_mismatches) {
            std::cout << " " << result.first << ":" << result.second;
        }
        std::cout << '\n'
                  << "OpenMP diagnostic/statistic mismatches: "
                  << openmp_diagnostic_mismatches << "/"
                  << openmp_statistic_mismatches << '\n'
                  << "Input bitwise changes: " << input_changes << '\n'
                  << "Finished at 3x3: " << statistics.finished_at_3x3 << " ("
                  << percentage(statistics.finished_at_3x3, statistics.total_outputs) << "%)\n"
                  << "Expanded to 5x5: " << statistics.expanded_to_5x5 << " ("
                  << percentage(statistics.expanded_to_5x5, statistics.total_outputs) << "%)\n"
                  << "Finished at 5x5: " << statistics.finished_at_5x5 << " ("
                  << percentage(statistics.finished_at_5x5, statistics.total_outputs) << "%)\n"
                  << "Expanded to 7x7: " << statistics.expanded_to_7x7 << " ("
                  << percentage(statistics.expanded_to_7x7, statistics.total_outputs) << "%)\n"
                  << "Finished at 7x7: " << statistics.finished_at_7x7 << " ("
                  << percentage(statistics.finished_at_7x7, statistics.total_outputs) << "%)\n"
                  << "Maximum-window fallback: " << statistics.maximum_window_fallback << " ("
                  << percentage(statistics.maximum_window_fallback, statistics.total_outputs)
                  << "%)\n"
                  << "Stage B retained center: " << statistics.stage_b_retained_center << " ("
                  << percentage(statistics.stage_b_retained_center, statistics.total_outputs)
                  << "%)\n"
                  << "Stage B replaced with median: "
                  << statistics.stage_b_replaced_with_median << " ("
                  << percentage(statistics.stage_b_replaced_with_median, statistics.total_outputs)
                  << "%)\n"
                  << "Median computations: " << statistics.median_computations << '\n'
                  << "Validation result: "
                  << (baseline_candidate_mismatches == 0 &&
                              baseline_diagnostic_mismatches == 0 &&
                              candidate_diagnostic_mismatches == 0 &&
                              statistic_mismatches == 0 &&
                              openmp_validation_passed &&
                              openmp_diagnostic_mismatches == 0 &&
                              openmp_statistic_mismatches == 0 &&
                              input_changes == 0
                          ? "PASS"
                          : "FAIL")
                  << '\n';

        return baseline_candidate_mismatches == 0 &&
                baseline_diagnostic_mismatches == 0 &&
                candidate_diagnostic_mismatches == 0 &&
                statistic_mismatches == 0 &&
                openmp_validation_passed &&
                openmp_diagnostic_mismatches == 0 &&
                openmp_statistic_mismatches == 0 &&
                input_changes == 0
            ? 0
            : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
