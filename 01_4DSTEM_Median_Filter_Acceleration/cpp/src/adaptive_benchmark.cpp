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

#include "adaptive_median.hpp"
#include "npy.hpp"

#ifndef PHASE_B_BENCHMARK_INPUT_PATH
#error "PHASE_B_BENCHMARK_INPUT_PATH must be provided by CMake."
#endif

namespace {

constexpr std::size_t expected_dimension_count = 4;
constexpr const char* expected_dtype = "<f8";
constexpr std::size_t default_timed_runs = 7;

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
        std::size_t profile_runs = 0;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string value = argv[argument];
            if (value == "--profile-runs") {
                if (++argument >= argc || profile_runs != 0) {
                    throw std::runtime_error("--profile-runs requires one positive count.");
                }
                profile_runs = parse_positive_count(argv[argument]);
            }
            else if (input_path != std::filesystem::path(PHASE_B_BENCHMARK_INPUT_PATH)) {
                throw std::runtime_error(
                    "Usage: phase_b_adaptive_median_benchmark.exe [input.npy] [--profile-runs N]");
            }
            else {
                input_path = value;
            }
        }
        input_path = std::filesystem::absolute(input_path).lexically_normal();

        const LoadedNpy loaded = load_npy(input_path);
        const phase_a::Dimensions4D source_dimensions = dimensions_from_shape(loaded.array.shape);
        phase_a::Dimensions4D dimensions{};
        std::array<Slice, 4> slices{};
        const std::vector<double> input = extract_historical_profile_subset(
            loaded.array.data, source_dimensions, dimensions, slices);
        const std::vector<double> original_input = input;

        std::vector<double> output = phase_b::adaptive_median_s3_smax7(input, dimensions);
        if (profile_runs != 0) {
            volatile double checksum = 0.0;
            for (std::size_t run = 0; run < profile_runs; ++run) {
                output = phase_b::adaptive_median_s3_smax7(input, dimensions);
                checksum += output[run % output.size()];
            }
            std::cout << "Phase B adaptive profiling workload complete\n"
                      << "Profile filter calls: " << profile_runs << '\n'
                      << "Checksum: " << checksum << '\n';
            return 0;
        }

        std::vector<double> timings_ms;
        timings_ms.reserve(default_timed_runs);
        for (std::size_t run = 0; run < default_timed_runs; ++run) {
            const auto start = std::chrono::steady_clock::now();
            std::vector<double> run_output =
                phase_b::adaptive_median_s3_smax7(input, dimensions);
            const auto stop = std::chrono::steady_clock::now();
            timings_ms.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
            output = std::move(run_output);
        }

        const std::vector<double> sorted_timings = [&timings_ms] {
            std::vector<double> sorted = timings_ms;
            std::sort(sorted.begin(), sorted.end());
            return sorted;
        }();
        const double median_ms = sorted_timings[sorted_timings.size() / 2];
        const double mean_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) /
            static_cast<double>(timings_ms.size());
        double squared_deviation_sum = 0.0;
        for (const double timing : timings_ms) {
            const double deviation = timing - mean_ms;
            squared_deviation_sum += deviation * deviation;
        }
        const double coefficient_of_variation =
            100.0 * std::sqrt(squared_deviation_sum / timings_ms.size()) / mean_ms;

        const phase_b::AdaptiveMedianDiagnosticResult diagnostic =
            phase_b::adaptive_median_s3_smax7_diagnostics(input, dimensions);
        std::size_t diagnostic_mismatches = 0;
        for (std::size_t index = 0; index < output.size(); ++index) {
            diagnostic_mismatches +=
                double_bits(output[index]) != double_bits(diagnostic.output[index]);
        }
        std::size_t input_changes = 0;
        for (std::size_t index = 0; index < input.size(); ++index) {
            input_changes += double_bits(input[index]) != double_bits(original_input[index]);
        }

        const phase_b::AdaptiveMedianStatistics& statistics = diagnostic.statistics;
        std::cout << std::fixed << std::setprecision(6)
                  << "Project 1 Phase B native adaptive-median benchmark\n"
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
                  << "Warm-up filter calls: 1\n"
                  << "Timed filter calls: " << timings_ms.size() << '\n'
                  << "Raw times (ms):";
        for (const double timing : timings_ms) {
            std::cout << ' ' << timing;
        }
        std::cout << "\nMedian/min/max (ms): " << median_ms << " / "
                  << sorted_timings.front() << " / " << sorted_timings.back() << '\n'
                  << "Mean (ms): " << mean_ms << '\n'
                  << "CV (%): " << coefficient_of_variation << '\n'
                  << "Throughput (Moutput/s): "
                  << static_cast<double>(input.size()) / (median_ms * 1000.0) << '\n'
                  << "Diagnostic-vs-timed bitwise mismatches: " << diagnostic_mismatches << '\n'
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
                  << (diagnostic_mismatches == 0 && input_changes == 0 ? "PASS" : "FAIL")
                  << '\n';

        return diagnostic_mismatches == 0 && input_changes == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
