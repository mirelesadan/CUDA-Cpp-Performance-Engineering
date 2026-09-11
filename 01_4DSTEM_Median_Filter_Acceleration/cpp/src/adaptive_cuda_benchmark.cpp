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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "adaptive_median_cuda.hpp"
#include "adaptive_median_detail.hpp"
#include "fixed_median.hpp"
#include "npy.hpp"

#ifndef PHASE_B_BENCHMARK_INPUT_PATH
#error "PHASE_B_BENCHMARK_INPUT_PATH must be provided by CMake."
#endif

namespace {

struct LoadedNpy {
    npy::npy_data<double> array;
    std::size_t element_count;
};

struct Summary {
    double median;
    double minimum;
    double maximum;
    double mean;
    double cv_percent;
};

LoadedNpy load_npy(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Could not open benchmark input: " + path.string());
    }
    const auto header = npy::parse_header(npy::read_header(stream));
    if (header.dtype.str() != "<f8" || header.shape.size() != 4 || header.fortran_order) {
        throw std::runtime_error("Benchmark input must be C-order four-dimensional float64.");
    }
    std::size_t count = 1;
    for (const auto dimension : header.shape) {
        if (dimension == 0 || count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("Benchmark input has invalid dimensions.");
        }
        count *= dimension;
    }
    stream.clear();
    stream.seekg(0);
    auto array = npy::read_npy<double>(stream);
    if (!stream || array.data.size() != count) {
        throw std::runtime_error("Benchmark payload does not match its header.");
    }
    return {std::move(array), count};
}

phase_a::Dimensions4D dimensions_from(const npy::shape_t& shape)
{
    return {shape[0], shape[1], shape[2], shape[3]};
}

std::vector<double> extract_subset(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& source,
    phase_a::Dimensions4D& subset)
{
    const std::array<std::size_t, 4> counts = {
        std::min<std::size_t>(source.scan_y, 64),
        std::min<std::size_t>(source.scan_x, 35),
        std::min<std::size_t>(source.detector_y, 8),
        std::min<std::size_t>(source.detector_x, 8),
    };
    const std::array<std::size_t, 4> starts = {
        (source.scan_y - counts[0]) / 2,
        0,
        (source.detector_y - counts[2]) / 2,
        (source.detector_x - counts[3]) / 2,
    };
    subset = {counts[0], counts[1], counts[2], counts[3]};
    std::vector<double> result(counts[0] * counts[1] * counts[2] * counts[3]);
    for (std::size_t sy = 0; sy < counts[0]; ++sy) {
        for (std::size_t sx = 0; sx < counts[1]; ++sx) {
            for (std::size_t dy = 0; dy < counts[2]; ++dy) {
                for (std::size_t dx = 0; dx < counts[3]; ++dx) {
                    result[phase_a::flat_index(subset, sy, sx, dy, dx)] = input[
                        phase_a::flat_index(
                            source, sy + starts[0], sx + starts[1],
                            dy + starts[2], dx + starts[3])];
                }
            }
        }
    }
    return result;
}

std::uint64_t bits(double value)
{
    std::uint64_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::size_t mismatches(const std::vector<double>& left, const std::vector<double>& right)
{
    if (left.size() != right.size()) {
        return std::max(left.size(), right.size());
    }
    std::size_t count = 0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        count += bits(left[index]) != bits(right[index]);
    }
    return count;
}

std::size_t statistic_mismatches(
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

Summary summarize(const std::vector<double>& values)
{
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squared = 0.0;
    for (const double value : values) {
        const double deviation = value - mean;
        squared += deviation * deviation;
    }
    return {sorted[sorted.size() / 2], sorted.front(), sorted.back(), mean,
            100.0 * std::sqrt(squared / values.size()) / mean};
}

void print_series(const char* label, const std::vector<double>& values)
{
    const Summary summary = summarize(values);
    std::cout << label << " raw (ms):";
    for (const double value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n' << label << " median/min/max/mean/CV(%): "
              << summary.median << " / " << summary.minimum << " / "
              << summary.maximum << " / " << summary.mean << " / "
              << summary.cv_percent << '\n';
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc > 2) {
            throw std::runtime_error("Usage: phase_b_adaptive_cuda_benchmark.exe [canonical_input.npy]");
        }
        const auto canonical = load_npy(argc == 2 ? argv[1] : PHASE_B_BENCHMARK_INPUT_PATH);
        const auto dimensions = dimensions_from(canonical.array.shape);
        phase_a::Dimensions4D subset_dimensions{};
        const auto subset = extract_subset(canonical.array.data, dimensions, subset_dimensions);

        const auto subset_cpu =
            phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
                subset, subset_dimensions);
        const auto subset_cuda = phase_b::adaptive_median_s3_smax7_cuda_baseline_diagnostics(
            subset, subset_dimensions);
        const auto canonical_cpu =
            phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
                canonical.array.data, dimensions);
        const auto canonical_cuda = phase_b::adaptive_median_s3_smax7_cuda_baseline_diagnostics(
            canonical.array.data, dimensions);
        const std::size_t subset_output_mismatches = mismatches(subset_cpu.output, subset_cuda.output);
        const std::size_t subset_counter_mismatches = statistic_mismatches(
            subset_cpu.statistics, subset_cuda.statistics);
        const std::size_t canonical_output_mismatches = mismatches(
            canonical_cpu.output, canonical_cuda.output);
        const std::size_t canonical_counter_mismatches = statistic_mismatches(
            canonical_cpu.statistics, canonical_cuda.statistics);
        if (subset_output_mismatches != 0 || subset_counter_mismatches != 0 ||
            canonical_output_mismatches != 0 || canonical_counter_mismatches != 0) {
            throw std::runtime_error("CUDA correctness or diagnostic counters differ from optimized CPU.");
        }

        const auto kernel_sequence =
            phase_b::benchmark_adaptive_median_s3_smax7_cuda_kernels(
                canonical.array.data, dimensions, 5, 7);
        if (mismatches(kernel_sequence.output, canonical_cpu.output) != 0) {
            throw std::runtime_error("Stable kernel-sequence output differs from optimized CPU.");
        }

        for (int warmup = 0; warmup < 5; ++warmup) {
            const auto ignored = phase_b::adaptive_median_s3_smax7_cuda_baseline(
                canonical.array.data, dimensions);
        }

        std::vector<double> h2d, minima, adaptive, d2h, total, wall;
        std::vector<double> final_timed_output;
        for (int run = 0; run < 7; ++run) {
            auto result = phase_b::adaptive_median_s3_smax7_cuda_baseline(
                canonical.array.data, dimensions);
            h2d.push_back(result.timing.host_to_device);
            minima.push_back(result.timing.plane_minimum_kernel);
            adaptive.push_back(result.timing.adaptive_filter_kernel);
            d2h.push_back(result.timing.device_to_host);
            total.push_back(result.timing.total_gpu_path);
            wall.push_back(result.timing.native_wall);
            final_timed_output = std::move(result.output);
        }
        if (mismatches(final_timed_output, canonical_cpu.output) != 0) {
            throw std::runtime_error("Final timed CUDA output differs from optimized CPU.");
        }

        const auto configuration = phase_b::adaptive_median_cuda_kernel_configuration(dimensions);
        const Summary adaptive_summary = summarize(
            kernel_sequence.adaptive_filter_kernel_milliseconds);
        const Summary total_summary = summarize(total);
        std::cout << std::fixed << std::setprecision(6)
                  << "Canonical shape: (" << dimensions.scan_y << ',' << dimensions.scan_x
                  << ',' << dimensions.detector_y << ',' << dimensions.detector_x << ")\n"
                  << "Canonical outputs: " << canonical.element_count << '\n'
                  << "Subset shape: (" << subset_dimensions.scan_y << ',' << subset_dimensions.scan_x
                  << ',' << subset_dimensions.detector_y << ',' << subset_dimensions.detector_x << ")\n"
                  << "Subset output/counter mismatches: " << subset_output_mismatches << " / "
                  << subset_counter_mismatches << '\n'
                  << "Canonical output/counter mismatches: " << canonical_output_mismatches << " / "
                  << canonical_counter_mismatches << '\n'
                  << "Threads/block, plane-min blocks, adaptive blocks: "
                  << configuration.threads_per_block << " / "
                  << configuration.plane_minimum_blocks << " / "
                  << configuration.adaptive_filter_blocks << '\n';
        print_series("H2D", h2d);
        print_series("Steady plane minimum kernel", kernel_sequence.plane_minimum_kernel_milliseconds);
        print_series("Steady adaptive filter kernel", kernel_sequence.adaptive_filter_kernel_milliseconds);
        print_series("Sparse one-shot plane minimum kernel", minima);
        print_series("Sparse one-shot adaptive filter kernel", adaptive);
        print_series("D2H", d2h);
        print_series("Total GPU path", total);
        print_series("Native one-shot wall", wall);
        std::cout << "Adaptive kernel throughput (Moutput/s): "
                  << canonical.element_count / (adaptive_summary.median * 1000.0) << '\n'
                  << "Kernel-only speedup vs serial 3318.3376 ms: "
                  << 3318.3376 / adaptive_summary.median << '\n'
                  << "Kernel-only speedup vs OpenMP-16 426.5593 ms: "
                  << 426.5593 / adaptive_summary.median << '\n'
                  << "One-shot GPU-path speedup vs serial 3318.3376 ms: "
                  << 3318.3376 / total_summary.median << '\n'
                  << "One-shot GPU-path speedup vs OpenMP-16 426.5593 ms: "
                  << 426.5593 / total_summary.median << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Adaptive CUDA benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
