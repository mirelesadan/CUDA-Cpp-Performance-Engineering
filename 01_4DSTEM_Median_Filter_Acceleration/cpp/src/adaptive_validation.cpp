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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "adaptive_median.hpp"
#include "adaptive_median_detail.hpp"
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

std::size_t count_bitwise_mismatches(
    const std::vector<double>& left,
    const std::vector<double>& right)
{
    if (left.size() != right.size()) {
        throw std::runtime_error("Cannot compare arrays with different element counts.");
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

struct SelectorVerification {
    std::size_t permutations = 0;
    std::size_t permutation_mismatches = 0;
    std::size_t duplicate_cases = 0;
    std::size_t duplicate_mismatches = 0;
};

double sorted_median(std::array<double, 9> values)
{
    std::sort(values.begin(), values.end());
    return values[4];
}

SelectorVerification verify_median_of_nine()
{
    SelectorVerification result;
    std::array<double, 9> permutation = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    do {
        std::array<double, 9> candidate = permutation;
        const double expected = sorted_median(permutation);
        const double actual = phase_b::detail::median_of_nine_in_place(candidate.data());
        ++result.permutations;
        result.permutation_mismatches += double_bits(actual) != double_bits(expected);
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    const std::array<std::array<double, 9>, 10> duplicate_inputs = {{
        {{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}},
        {{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
        {{-2.0, -2.0, -2.0, 3.0, 3.0, 3.0, 9.0, 9.0, 9.0}},
        {{7.0, 1.0, 7.0, 1.0, 7.0, 1.0, 7.0, 1.0, 7.0}},
        {{-5.0, 2.0, -5.0, 2.0, -5.0, 2.0, 8.0, 8.0, 8.0}},
        {{10.0, -1.0, -1.0, -1.0, 10.0, 10.0, 3.0, 3.0, 3.0}},
        {{2.5, 2.5, -4.5, -4.5, 0.0, 0.0, 9.5, 9.5, 9.5}},
        {{100.0, 0.0, 0.0, 0.0, 0.0, -100.0, -100.0, -100.0, -100.0}},
        {{-3.0, 6.0, -3.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
        {{8.0, 8.0, 8.0, -8.0, -8.0, -8.0, 0.25, 0.25, 0.25}},
    }};
    for (const std::array<double, 9>& input : duplicate_inputs) {
        std::array<double, 9> candidate = input;
        std::array<double, 9> reference = input;
        std::nth_element(reference.begin(), reference.begin() + 4, reference.end());
        const double actual = phase_b::detail::median_of_nine_in_place(candidate.data());
        ++result.duplicate_cases;
        result.duplicate_mismatches += double_bits(actual) != double_bits(reference[4]);
    }
    return result;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const SelectorVerification selector_verification = verify_median_of_nine();
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
        const phase_a::Dimensions4D dimensions = dimensions_from_shape(input.array.shape);
        omp_set_dynamic(0);
        const int available_threads = std::max(
            1,
            std::min(phase_a::openmp_max_threads(), phase_a::openmp_processor_count()));
        std::vector<int> openmp_thread_counts;
        for (const int count : {1, 2, 4, 8, 16, 20}) {
            if (count <= available_threads) {
                openmp_thread_counts.push_back(count);
            }
        }
        if (openmp_thread_counts.empty() || openmp_thread_counts.back() != available_threads) {
            openmp_thread_counts.push_back(available_threads);
        }
        const auto baseline_start = std::chrono::steady_clock::now();
        const std::vector<double> baseline = phase_b::adaptive_median_s3_smax7(
            input.array.data,
            dimensions);
        const auto baseline_stop = std::chrono::steady_clock::now();
        const auto stack_start = std::chrono::steady_clock::now();
        const std::vector<double> stack = phase_b::adaptive_median_s3_smax7_stack(
            input.array.data,
            dimensions);
        const auto stack_stop = std::chrono::steady_clock::now();
        const auto specialized_start = std::chrono::steady_clock::now();
        const std::vector<double> specialized =
            phase_b::adaptive_median_s3_smax7_specialized_3x3(
                input.array.data,
                dimensions);
        const auto specialized_stop = std::chrono::steady_clock::now();
        const auto direct_gather_start = std::chrono::steady_clock::now();
        const std::vector<double> direct_gather =
            phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather(
                input.array.data,
                dimensions);
        const auto direct_gather_stop = std::chrono::steady_clock::now();

        const phase_b::AdaptiveMedianDiagnosticResult baseline_diagnostic =
            phase_b::adaptive_median_s3_smax7_diagnostics(input.array.data, dimensions);
        const phase_b::AdaptiveMedianDiagnosticResult stack_diagnostic =
            phase_b::adaptive_median_s3_smax7_stack_diagnostics(input.array.data, dimensions);
        const phase_b::AdaptiveMedianDiagnosticResult specialized_diagnostic =
            phase_b::adaptive_median_s3_smax7_specialized_3x3_diagnostics(
                input.array.data,
                dimensions);
        const phase_b::AdaptiveMedianDiagnosticResult direct_gather_diagnostic =
            phase_b::detail::adaptive_median_s3_smax7_direct_3x3_gather_diagnostics(
                input.array.data,
                dimensions);

        struct OpenMpValidation {
            int threads;
            std::size_t reference_mismatches;
            std::size_t serial_mismatches;
            std::size_t diagnostic_mismatches;
            std::size_t statistic_mismatches;
        };
        std::vector<OpenMpValidation> openmp_validations;
        for (const int threads : openmp_thread_counts) {
            const std::vector<double> openmp_output = phase_b::adaptive_median_s3_smax7_openmp(
                input.array.data, dimensions, threads);
            const phase_b::AdaptiveMedianDiagnosticResult openmp_diagnostic =
                phase_b::adaptive_median_s3_smax7_openmp_diagnostics(
                    input.array.data, dimensions, threads);
            openmp_validations.push_back({
                threads,
                count_bitwise_mismatches(openmp_output, reference.array.data),
                count_bitwise_mismatches(openmp_output, direct_gather),
                count_bitwise_mismatches(openmp_output, openmp_diagnostic.output),
                count_statistic_mismatches(
                    direct_gather_diagnostic.statistics, openmp_diagnostic.statistics),
            });
        }

        bool rejected_zero_threads = false;
        bool rejected_negative_threads = false;
        try {
            static_cast<void>(phase_b::adaptive_median_s3_smax7_openmp(
                input.array.data, dimensions, 0));
        }
        catch (const std::invalid_argument&) {
            rejected_zero_threads = true;
        }
        try {
            static_cast<void>(phase_b::adaptive_median_s3_smax7_openmp(
                input.array.data, dimensions, -1));
        }
        catch (const std::invalid_argument&) {
            rejected_negative_threads = true;
        }

        const std::size_t baseline_reference_mismatches =
            count_bitwise_mismatches(baseline, reference.array.data);
        const std::size_t stack_reference_mismatches =
            count_bitwise_mismatches(stack, reference.array.data);
        const std::size_t baseline_stack_mismatches =
            count_bitwise_mismatches(baseline, stack);
        const std::size_t specialized_reference_mismatches =
            count_bitwise_mismatches(specialized, reference.array.data);
        const std::size_t stack_specialized_mismatches =
            count_bitwise_mismatches(stack, specialized);
        const std::size_t baseline_diagnostic_mismatches =
            count_bitwise_mismatches(baseline, baseline_diagnostic.output);
        const std::size_t stack_diagnostic_mismatches =
            count_bitwise_mismatches(stack, stack_diagnostic.output);
        const std::size_t statistic_mismatches = count_statistic_mismatches(
            baseline_diagnostic.statistics, stack_diagnostic.statistics);
        const std::size_t specialized_diagnostic_mismatches =
            count_bitwise_mismatches(specialized, specialized_diagnostic.output);
        const std::size_t specialized_statistic_mismatches = count_statistic_mismatches(
            stack_diagnostic.statistics, specialized_diagnostic.statistics);
        const std::size_t direct_gather_reference_mismatches =
            count_bitwise_mismatches(direct_gather, reference.array.data);
        const std::size_t specialized_direct_gather_mismatches =
            count_bitwise_mismatches(specialized, direct_gather);
        const std::size_t direct_gather_diagnostic_mismatches =
            count_bitwise_mismatches(direct_gather, direct_gather_diagnostic.output);
        const std::size_t direct_gather_statistic_mismatches = count_statistic_mismatches(
            specialized_diagnostic.statistics, direct_gather_diagnostic.statistics);

        std::size_t first_mismatch = direct_gather.size();
        for (std::size_t index = 0; index < direct_gather.size(); ++index) {
            if (double_bits(direct_gather[index]) != double_bits(reference.array.data[index])) {
                if (first_mismatch == direct_gather.size()) {
                    first_mismatch = index;
                }
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
                  << "Median-of-nine distinct permutations checked: "
                  << selector_verification.permutations << '\n'
                  << "Median-of-nine permutation mismatches: "
                  << selector_verification.permutation_mismatches << '\n'
                  << "Median-of-nine duplicate cases checked: "
                  << selector_verification.duplicate_cases << '\n'
                  << "Median-of-nine duplicate mismatches: "
                  << selector_verification.duplicate_mismatches << '\n'
                  << "Elements compared: " << input.element_count << '\n'
                  << "Heap baseline-vs-reference mismatches: "
                  << baseline_reference_mismatches << '\n'
                  << "Stack candidate-vs-reference mismatches: "
                  << stack_reference_mismatches << '\n'
                  << "Heap baseline-vs-stack candidate mismatches: "
                  << baseline_stack_mismatches << '\n'
                  << "Specialized-vs-reference mismatches: "
                  << specialized_reference_mismatches << '\n'
                  << "Stack baseline-vs-specialized mismatches: "
                  << stack_specialized_mismatches << '\n'
                  << "Heap diagnostic-vs-normal mismatches: "
                  << baseline_diagnostic_mismatches << '\n'
                  << "Stack diagnostic-vs-normal mismatches: "
                  << stack_diagnostic_mismatches << '\n'
                  << "Specialized diagnostic-vs-normal mismatches: "
                  << specialized_diagnostic_mismatches << '\n'
                  << "Heap-vs-stack statistic field mismatches: " << statistic_mismatches << '\n'
                  << "Stack-vs-specialized statistic field mismatches: "
                  << specialized_statistic_mismatches << '\n'
                  << "Direct-gather-vs-reference mismatches: "
                  << direct_gather_reference_mismatches << '\n'
                  << "Specialized baseline-vs-direct-gather mismatches: "
                  << specialized_direct_gather_mismatches << '\n'
                  << "Direct-gather diagnostic-vs-normal mismatches: "
                  << direct_gather_diagnostic_mismatches << '\n'
                  << "Specialized-vs-direct-gather statistic field mismatches: "
                  << direct_gather_statistic_mismatches << '\n'
                  << "OpenMP maximum threads: " << available_threads << '\n';
        for (const OpenMpValidation& validation : openmp_validations) {
            std::cout << "OpenMP-" << validation.threads
                      << " reference/serial/diagnostic/statistic mismatches: "
                      << validation.reference_mismatches << "/"
                      << validation.serial_mismatches << "/"
                      << validation.diagnostic_mismatches << "/"
                      << validation.statistic_mismatches << '\n';
        }
        std::cout << "Rejected zero/negative OpenMP thread counts: "
                  << (rejected_zero_threads ? "yes" : "no") << "/"
                  << (rejected_negative_threads ? "yes" : "no") << '\n'
                  << "Input bitwise changes: " << input_mismatches << '\n'
                  << std::fixed << std::setprecision(6)
                  << "Heap filter time: "
                  << std::chrono::duration<double, std::milli>(
                         baseline_stop - baseline_start).count() << " ms\n"
                  << "Stack filter time: "
                  << std::chrono::duration<double, std::milli>(
                         stack_stop - stack_start).count() << " ms\n"
                  << "Specialized filter time: "
                  << std::chrono::duration<double, std::milli>(
                         specialized_stop - specialized_start).count() << " ms\n"
                  << "Direct-gather filter time: "
                  << std::chrono::duration<double, std::milli>(
                         direct_gather_stop - direct_gather_start).count() << " ms\n";

        if (first_mismatch != direct_gather.size()) {
            const std::size_t detector_x = first_mismatch % input.array.shape[3];
            std::size_t remaining = first_mismatch / input.array.shape[3];
            const std::size_t detector_y = remaining % input.array.shape[2];
            remaining /= input.array.shape[2];
            const std::size_t scan_x = remaining % input.array.shape[1];
            const std::size_t scan_y = remaining / input.array.shape[1];
            std::cout << "First mismatch: flat index " << first_mismatch
                      << " at (" << scan_y << ", " << scan_x << ", "
                      << detector_y << ", " << detector_x << ")"
                      << ", direct-gather bits 0x" << std::hex
                      << double_bits(direct_gather[first_mismatch])
                      << ", expected bits 0x"
                      << double_bits(reference.array.data[first_mismatch]) << std::dec << '\n';
        }

        const bool openmp_passed = std::all_of(
            openmp_validations.begin(),
            openmp_validations.end(),
            [](const OpenMpValidation& validation) {
                return validation.reference_mismatches == 0 &&
                    validation.serial_mismatches == 0 &&
                    validation.diagnostic_mismatches == 0 &&
                    validation.statistic_mismatches == 0;
            });
        const bool passed =
            selector_verification.permutations == 362880 &&
            selector_verification.permutation_mismatches == 0 &&
            selector_verification.duplicate_mismatches == 0 &&
            baseline_reference_mismatches == 0 &&
            stack_reference_mismatches == 0 &&
            baseline_stack_mismatches == 0 &&
            specialized_reference_mismatches == 0 &&
            stack_specialized_mismatches == 0 &&
            baseline_diagnostic_mismatches == 0 &&
            stack_diagnostic_mismatches == 0 &&
            specialized_diagnostic_mismatches == 0 &&
            statistic_mismatches == 0 &&
            specialized_statistic_mismatches == 0 &&
            direct_gather_reference_mismatches == 0 &&
            specialized_direct_gather_mismatches == 0 &&
            direct_gather_diagnostic_mismatches == 0 &&
            direct_gather_statistic_mismatches == 0 &&
            openmp_passed &&
            rejected_zero_threads &&
            rejected_negative_threads &&
            input_mismatches == 0;
        std::cout << "Validation result: " << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
