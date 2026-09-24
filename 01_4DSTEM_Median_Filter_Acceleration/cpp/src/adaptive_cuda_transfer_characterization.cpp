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

#include "adaptive_median_cuda_transfer_experiment.hpp"
#include "adaptive_median_detail.hpp"
#include "fixed_median.hpp"
#include "npy.hpp"

#ifndef PHASE_B_BENCHMARK_INPUT_PATH
#error "PHASE_B_BENCHMARK_INPUT_PATH must be provided by CMake."
#endif
#ifndef PHASE_B_REFERENCE_INPUT_PATH
#error "PHASE_B_REFERENCE_INPUT_PATH must be provided by CMake."
#endif
#ifndef PHASE_B_REFERENCE_OUTPUT_PATH
#error "PHASE_B_REFERENCE_OUTPUT_PATH must be provided by CMake."
#endif
#ifndef PHASE_B_LOCAL_REFERENCE_INPUT_PATH
#error "PHASE_B_LOCAL_REFERENCE_INPUT_PATH must be provided by CMake."
#endif
#ifndef PHASE_B_LOCAL_REFERENCE_OUTPUT_PATH
#error "PHASE_B_LOCAL_REFERENCE_OUTPUT_PATH must be provided by CMake."
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Workload {
    std::vector<double> input;
    phase_a::Dimensions4D dimensions;
};

struct Statistics {
    double minimum;
    double median;
    double maximum;
    double mean;
    double cv_percent;
};

Workload load_npy(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Could not open input: " + path.string());
    const auto header = npy::parse_header(npy::read_header(stream));
    if (header.dtype.str() != "<f8" || header.shape.size() != 4 || header.fortran_order) {
        throw std::runtime_error("Input must be C-order, four-dimensional float64.");
    }
    std::size_t count = 1;
    for (const auto dimension : header.shape) {
        if (dimension == 0 || count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("Input shape is invalid.");
        }
        count *= dimension;
    }
    stream.clear();
    stream.seekg(0);
    auto array = npy::read_npy<double>(stream);
    if (!stream || array.data.size() != count || array.shape != header.shape) {
        throw std::runtime_error("Input payload does not match its header.");
    }
    return {std::move(array.data), {header.shape[0], header.shape[1],
                                    header.shape[2], header.shape[3]}};
}

bool same_shape(const phase_a::Dimensions4D& left, const phase_a::Dimensions4D& right)
{
    return left.scan_y == right.scan_y && left.scan_x == right.scan_x &&
           left.detector_y == right.detector_y && left.detector_x == right.detector_x;
}

bool exact(const std::vector<double>& left, const std::vector<double>& right)
{
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(double)) == 0;
}

std::size_t counter_differences(const phase_b::AdaptiveMedianStatistics& left,
                                const phase_b::AdaptiveMedianStatistics& right)
{
    return (left.total_outputs != right.total_outputs) +
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

Workload centered_subset(const Workload& source)
{
    const auto& large = source.dimensions;
    Workload subset;
    subset.dimensions = {
        std::min<std::size_t>(large.scan_y, 64),
        std::min<std::size_t>(large.scan_x, 35),
        std::min<std::size_t>(large.detector_y, 8),
        std::min<std::size_t>(large.detector_x, 8),
    };
    const auto& small = subset.dimensions;
    const std::size_t start_y = (large.scan_y - small.scan_y) / 2;
    const std::size_t start_dy = (large.detector_y - small.detector_y) / 2;
    const std::size_t start_dx = (large.detector_x - small.detector_x) / 2;
    subset.input.resize(small.scan_y * small.scan_x * small.detector_y * small.detector_x);
    for (std::size_t sy = 0; sy < small.scan_y; ++sy) {
        for (std::size_t sx = 0; sx < small.scan_x; ++sx) {
            for (std::size_t dy = 0; dy < small.detector_y; ++dy) {
                for (std::size_t dx = 0; dx < small.detector_x; ++dx) {
                    subset.input[phase_a::flat_index(small, sy, sx, dy, dx)] =
                        source.input[phase_a::flat_index(
                            large, sy + start_y, sx, dy + start_dy, dx + start_dx)];
                }
            }
        }
    }
    return subset;
}

Statistics summarize(const std::vector<double>& values)
{
    if (values.empty()) throw std::runtime_error("Cannot summarize empty timing series.");
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0.0;
    for (const double value : values) variance += (value - mean) * (value - mean);
    const double standard_deviation = std::sqrt(variance / values.size());
    return {sorted.front(), sorted[sorted.size() / 2], sorted.back(), mean,
            mean == 0.0 ? 0.0 : 100.0 * standard_deviation / mean};
}

void print_series(const std::string& label, const std::vector<double>& values)
{
    const auto stats = summarize(values);
    std::cout << label << " raw ms:";
    for (double value : values) std::cout << ' ' << value;
    std::cout << "\n" << label << " min/median/max/mean/CV(%): "
              << stats.minimum << " / " << stats.median << " / " << stats.maximum
              << " / " << stats.mean << " / " << stats.cv_percent << '\n';
}

template <typename Sample>
std::vector<double> values(const std::vector<Sample>& samples, double Sample::* field)
{
    std::vector<double> result;
    result.reserve(samples.size());
    for (const auto& sample : samples) result.push_back(sample.*field);
    return result;
}

std::vector<double> resident_values(
    const phase_b::detail::AdaptiveResidentSeries& series,
    double phase_b::detail::AdaptiveResidentSample::* field,
    bool per_filter)
{
    auto result = values(series.samples, field);
    if (per_filter) {
        for (double& value : result) value /= series.repetition_count;
    }
    return result;
}

template <typename Filter>
std::vector<double> time_cpu(Filter filter, const std::vector<double>& expected)
{
    if (!exact(filter(), expected)) throw std::runtime_error("CPU warm-up output differs.");
    std::vector<double> result;
    for (int run = 0; run < 3; ++run) {
        const auto start = Clock::now();
        auto output = filter();
        const auto stop = Clock::now();
        if (!exact(output, expected)) throw std::runtime_error("CPU timed output differs.");
        result.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return result;
}

std::vector<double> validate_workload(
    const Workload& workload,
    const std::vector<double>* python_reference,
    const char* label)
{
    const auto input_snapshot = workload.input;
    auto cpu = phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
        workload.input, workload.dimensions);
    auto cuda = phase_b::adaptive_median_s3_smax7_cuda_split_balanced_diagnostics(
        workload.input, workload.dimensions);
    const bool pass = exact(cpu.output, cuda.output) &&
        (python_reference == nullptr || exact(cpu.output, *python_reference)) &&
        counter_differences(cpu.statistics, cuda.statistics) == 0 &&
        exact(input_snapshot, workload.input);
    if (!pass) throw std::runtime_error(std::string(label) + " CPU/CUDA/reference mismatch.");
    phase_b::detail::validate_adaptive_cuda_transfer_modes(
        workload.input, cuda.output, workload.dimensions);
    if (!exact(input_snapshot, workload.input)) {
        throw std::runtime_error(std::string(label) + " input changed in transfer modes.");
    }
    std::cout << label << " exact output/counters/pageable/pinned/resident: PASS ("
              << workload.input.size() << " outputs)\n";
    return std::move(cuda.output);
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 2) {
            throw std::runtime_error(
                "Usage: phase_b_cuda_transfer.exe [canonical_input.npy | --validate-only]");
        }
        const bool validate_only = argc == 2 && std::string(argv[1]) == "--validate-only";
        std::cout << std::fixed << std::setprecision(6);
        const auto public_input = load_npy(PHASE_B_REFERENCE_INPUT_PATH);
        const auto public_expected = load_npy(PHASE_B_REFERENCE_OUTPUT_PATH);
        if (!same_shape(public_input.dimensions, public_expected.dimensions)) {
            throw std::runtime_error("Public adaptive fixture shapes differ.");
        }
        validate_workload(public_input, &public_expected.input, "Public fixture");

        const std::filesystem::path local_input_path = PHASE_B_LOCAL_REFERENCE_INPUT_PATH;
        const std::filesystem::path local_expected_path = PHASE_B_LOCAL_REFERENCE_OUTPUT_PATH;
        if (std::filesystem::exists(local_input_path) &&
            std::filesystem::exists(local_expected_path)) {
            const auto local_input = load_npy(local_input_path);
            const auto local_expected = load_npy(local_expected_path);
            if (!same_shape(local_input.dimensions, local_expected.dimensions)) {
                throw std::runtime_error("Local adaptive fixture shapes differ.");
            }
            validate_workload(local_input, &local_expected.input, "Local fixture");
        }
        else {
            std::cout << "Local ignored fixture unavailable; public fixture remains runnable.\n";
        }

        const Workload canonical = load_npy(
            argc == 2 && !validate_only ? argv[1] : PHASE_B_BENCHMARK_INPUT_PATH);
        const Workload subset = centered_subset(canonical);
        validate_workload(subset, nullptr, "Representative subset");
        const auto expected = validate_workload(canonical, nullptr, "Canonical workload");
        if (validate_only) {
            std::cout << "Transfer-mode correctness: PASS; timing intentionally skipped.\n";
            return 0;
        }
        const auto input_snapshot = canonical.input;

        const auto serial_ms = time_cpu([&]() {
            return phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(
                canonical.input, canonical.dimensions);
        }, expected);
        const auto openmp_ms = time_cpu([&]() {
            return phase_b::adaptive_median_s3_smax7_openmp(
                canonical.input, canonical.dimensions, 16);
        }, expected);
        print_series("Optimized serial CPU", serial_ms);
        print_series("OpenMP-16 CPU", openmp_ms);
        std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << '\n';

        const auto experiment = phase_b::detail::benchmark_adaptive_cuda_transfer_residency(
            canonical.input, expected, canonical.dimensions,
            7, 20, 5, {1, 2, 5, 10, 20});
        if (!exact(input_snapshot, canonical.input)) {
            throw std::runtime_error("Canonical host input changed during benchmark.");
        }
        using OneShot = phase_b::detail::AdaptiveTransferOneShotSample;
        using Transfer = phase_b::detail::AdaptiveTransferPairSample;
        using Resident = phase_b::detail::AdaptiveResidentSample;
        const auto& pageable = experiment.pageable_one_shot;
        const auto& pinned = experiment.pinned_staging_one_shot;
        print_series("Pageable one-shot native wall", values(pageable, &OneShot::native_wall_ms));
        print_series("Pageable one-shot GPU path", values(pageable, &OneShot::gpu_path_ms));
        print_series("Pageable host preparation", values(pageable, &OneShot::host_preparation_ms));
        print_series("Pageable cudaMalloc", values(pageable, &OneShot::device_allocation_ms));
        print_series("Pageable event setup", values(pageable, &OneShot::event_setup_ms));
        print_series("Pageable H2D", values(pageable, &OneShot::host_to_device_ms));
        print_series("Pageable plane minimum", values(pageable, &OneShot::plane_minimum_ms));
        print_series("Pageable balanced common", values(pageable, &OneShot::common_3x3_ms));
        print_series("Pageable fallback", values(pageable, &OneShot::fallback_ms));
        print_series("Pageable D2H", values(pageable, &OneShot::device_to_host_ms));
        print_series("Pageable cudaFree", values(pageable, &OneShot::device_free_ms));
        print_series("Pinned-staged one-shot native wall", values(pinned, &OneShot::native_wall_ms));
        print_series("Pinned-staged one-shot GPU path", values(pinned, &OneShot::gpu_path_ms));
        print_series("Pinned host-to-staging copy", values(pinned, &OneShot::host_to_pinned_ms));
        print_series("Pinned staging-to-host copy", values(pinned, &OneShot::pinned_to_host_ms));
        std::cout << "Pinned staging allocation, once outside calls (ms): "
                  << experiment.pinned_buffer_allocation_ms << '\n'
                  << "Pinned staging release, once outside calls (ms): "
                  << experiment.pinned_buffer_free_ms << '\n';
        print_series("Transfer-only pageable H2D", values(experiment.transfer_pairs, &Transfer::pageable_h2d_ms));
        print_series("Transfer-only pinned H2D", values(experiment.transfer_pairs, &Transfer::pinned_h2d_ms));
        print_series("Transfer-only pageable D2H", values(experiment.transfer_pairs, &Transfer::pageable_d2h_ms));
        print_series("Transfer-only pinned D2H", values(experiment.transfer_pairs, &Transfer::pinned_d2h_ms));

        const auto serial_median = summarize(serial_ms).median;
        const auto openmp_median = summarize(openmp_ms).median;
        const auto one_shot_wall = summarize(values(pageable, &OneShot::native_wall_ms)).median;
        const auto one_shot_gpu = summarize(values(pageable, &OneShot::gpu_path_ms)).median;
        std::cout << "One-shot native-wall speedup vs serial/OpenMP-16: "
                  << serial_median / one_shot_wall << " / " << openmp_median / one_shot_wall << '\n'
                  << "One-shot GPU-path speedup vs serial/OpenMP-16 (different timing boundary): "
                  << serial_median / one_shot_gpu << " / " << openmp_median / one_shot_gpu << '\n';
        for (const auto& series : experiment.resident) {
            const auto total = resident_values(series, &Resident::total_gpu_path_ms, false);
            const auto effective = resident_values(series, &Resident::total_gpu_path_ms, true);
            const auto device = resident_values(series, &Resident::repeated_device_ms, true);
            const auto h2d = resident_values(series, &Resident::initial_h2d_ms, false);
            const auto d2h = resident_values(series, &Resident::final_d2h_ms, false);
            std::vector<double> transfer_fraction;
            for (const auto& sample : series.samples) {
                transfer_fraction.push_back(100.0 *
                    (sample.initial_h2d_ms + sample.final_d2h_ms) / sample.total_gpu_path_ms);
            }
            const std::string prefix = "Resident-" + std::to_string(series.repetition_count);
            print_series(prefix + " total GPU path", total);
            print_series(prefix + " initial H2D", h2d);
            print_series(prefix + " complete device/filter", device);
            print_series(prefix + " final D2H", d2h);
            print_series(prefix + " effective/filter", effective);
            print_series(prefix + " transfer fraction %", transfer_fraction);
            const double median = summarize(effective).median;
            std::cout << prefix << " effective speedup vs one-shot GPU path/serial/OpenMP-16: "
                      << one_shot_gpu / median << " / " << serial_median / median
                      << " / " << openmp_median / median << '\n';
        }
        const std::size_t resident_bytes = experiment.input_device_bytes +
            experiment.output_device_bytes + experiment.plane_minima_device_bytes +
            experiment.fallback_flags_device_bytes;
        std::cout << "Resident device bytes input/output/minima/flags/total: "
                  << experiment.input_device_bytes << '/' << experiment.output_device_bytes
                  << '/' << experiment.plane_minima_device_bytes << '/'
                  << experiment.fallback_flags_device_bytes << '/' << resident_bytes << '\n';
        std::cout << "Validation and characterization: PASS\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Adaptive transfer characterization failed: " << error.what() << '\n';
        return 1;
    }
}
