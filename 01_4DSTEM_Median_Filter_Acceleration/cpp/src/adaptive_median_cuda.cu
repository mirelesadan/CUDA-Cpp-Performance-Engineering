#include "adaptive_median_cuda.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace phase_b {
namespace {

constexpr unsigned int threads_per_block = 256;

void check_cuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result));
    }
}

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* message)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

std::size_t checked_element_count(const phase_a::Dimensions4D& dimensions)
{
    const std::array<std::size_t, 4> sizes = {
        dimensions.scan_y,
        dimensions.scan_x,
        dimensions.detector_y,
        dimensions.detector_x,
    };
    std::size_t count = 1;
    for (const std::size_t size : sizes) {
        if (size == 0) {
            throw std::invalid_argument("Adaptive CUDA median requires four nonempty dimensions.");
        }
        count = checked_multiply(count, size, "The adaptive CUDA shape exceeds std::size_t.");
    }
    checked_multiply(count, sizeof(double), "The adaptive CUDA byte count exceeds std::size_t.");
    return count;
}

void validate_input(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions)
{
    if (input.size() != checked_element_count(dimensions)) {
        throw std::invalid_argument("Input element count does not match the supplied 4D dimensions.");
    }
    if (!std::all_of(input.begin(), input.end(), [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Adaptive CUDA median requires finite float64 values.");
    }
}

unsigned int checked_block_count(std::size_t work_items)
{
    const std::size_t blocks = (work_items + threads_per_block - 1) / threads_per_block;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("Adaptive CUDA grid exceeds the one-dimensional grid limit.");
    }
    return static_cast<unsigned int>(blocks);
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
    {
        check_cuda(cudaMalloc(&pointer_, checked_multiply(count, sizeof(T), "CUDA buffer size overflow.")),
                   "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return static_cast<T*>(pointer_); }

private:
    void* pointer_ = nullptr;
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event()
    {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

void record(const Event& event, const char* operation)
{
    check_cuda(cudaEventRecord(event.get()), operation);
}

double elapsed(const Event& start, const Event& stop)
{
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    return static_cast<double>(milliseconds);
}

__device__ void compare_swap(double& left, double& right)
{
    if (right < left) {
        const double temporary = left;
        left = right;
        right = temporary;
    }
}

__device__ double median_of_nine(double* values)
{
    compare_swap(values[1], values[2]);
    compare_swap(values[4], values[5]);
    compare_swap(values[7], values[8]);
    compare_swap(values[0], values[1]);
    compare_swap(values[3], values[4]);
    compare_swap(values[6], values[7]);
    compare_swap(values[1], values[2]);
    compare_swap(values[4], values[5]);
    compare_swap(values[7], values[8]);
    compare_swap(values[0], values[3]);
    compare_swap(values[5], values[8]);
    compare_swap(values[4], values[7]);
    compare_swap(values[3], values[6]);
    compare_swap(values[1], values[4]);
    compare_swap(values[2], values[5]);
    compare_swap(values[4], values[7]);
    compare_swap(values[4], values[2]);
    compare_swap(values[6], values[4]);
    compare_swap(values[4], values[2]);
    return values[4];
}

__device__ double median_by_insertion_sort(double* values, int count)
{
    for (int index = 1; index < count; ++index) {
        const double value = values[index];
        int position = index;
        while (position > 0 && value < values[position - 1]) {
            values[position] = values[position - 1];
            --position;
        }
        values[position] = value;
    }
    return values[count / 2];
}

__global__ void detector_plane_minimum_kernel(
    const double* input,
    double* plane_minima,
    std::size_t plane_count,
    std::size_t scan_y_size,
    std::size_t scan_x_size)
{
    const std::size_t plane_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (plane_index >= plane_count) {
        return;
    }

    double minimum = input[plane_index];
    const std::size_t scan_position_count = scan_y_size * scan_x_size;
    for (std::size_t scan_position = 1; scan_position < scan_position_count; ++scan_position) {
        const double value = input[scan_position * plane_count + plane_index];
        if (value < minimum) {
            minimum = value;
        }
    }
    plane_minima[plane_index] = minimum;
}

enum Outcome : std::uint8_t {
    finish_3_retain = 0,
    finish_3_replace = 1,
    finish_5_retain = 2,
    finish_5_replace = 3,
    finish_7_retain = 4,
    finish_7_replace = 5,
    maximum_fallback = 6,
};

__global__ void adaptive_median_s3_smax7_kernel(
    const double* input,
    const double* plane_minima,
    double* output,
    std::uint8_t* outcomes,
    std::size_t element_count,
    std::size_t scan_y_size,
    std::size_t scan_x_size,
    std::size_t detector_y_size,
    std::size_t detector_x_size)
{
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= element_count) {
        return;
    }

    const std::size_t plane_count = detector_y_size * detector_x_size;
    std::size_t remaining = output_index;
    const std::size_t detector_x = remaining % detector_x_size;
    remaining /= detector_x_size;
    const std::size_t detector_y = remaining % detector_y_size;
    remaining /= detector_y_size;
    const std::size_t scan_x = remaining % scan_x_size;
    const std::size_t scan_y = remaining / scan_x_size;
    const std::size_t plane_index = detector_y * detector_x_size + detector_x;
    const double plane_minimum = plane_minima[plane_index];
    const double center = input[output_index];
    double result = center;
    Outcome outcome = maximum_fallback;

    double window[49];
    for (int window_size = 3; window_size <= 7; window_size += 2) {
        const int radius = window_size / 2;
        int count = 0;
        double local_minimum = CUDART_INF;
        double local_maximum = -CUDART_INF;

        for (int offset_y = -radius; offset_y <= radius; ++offset_y) {
            const long long neighbor_y = static_cast<long long>(scan_y) + offset_y;
            for (int offset_x = -radius; offset_x <= radius; ++offset_x) {
                const long long neighbor_x = static_cast<long long>(scan_x) + offset_x;
                double value = plane_minimum;
                if (neighbor_y >= 0 && neighbor_y < static_cast<long long>(scan_y_size) &&
                    neighbor_x >= 0 && neighbor_x < static_cast<long long>(scan_x_size)) {
                    const std::size_t neighbor_scan_position =
                        static_cast<std::size_t>(neighbor_y) * scan_x_size +
                        static_cast<std::size_t>(neighbor_x);
                    value = input[neighbor_scan_position * plane_count + plane_index];
                }
                window[count++] = value;
                if (value < local_minimum) {
                    local_minimum = value;
                }
                if (local_maximum < value) {
                    local_maximum = value;
                }
            }
        }

        const double local_median = window_size == 3
            ? median_of_nine(window)
            : median_by_insertion_sort(window, count);
        if (local_minimum < local_median && local_median < local_maximum) {
            const bool retain_center = local_minimum < center && center < local_maximum;
            result = retain_center ? center : local_median;
            if (window_size == 3) {
                outcome = retain_center ? finish_3_retain : finish_3_replace;
            }
            else if (window_size == 5) {
                outcome = retain_center ? finish_5_retain : finish_5_replace;
            }
            else {
                outcome = retain_center ? finish_7_retain : finish_7_replace;
            }
            break;
        }
    }

    output[output_index] = result;
    if (outcomes != nullptr) {
        outcomes[output_index] = static_cast<std::uint8_t>(outcome);
    }
}

void launch_plane_minimum(
    const double* input,
    double* plane_minima,
    const phase_a::Dimensions4D& dimensions,
    unsigned int blocks)
{
    detector_plane_minimum_kernel<<<blocks, threads_per_block>>>(
        input,
        plane_minima,
        dimensions.detector_y * dimensions.detector_x,
        dimensions.scan_y,
        dimensions.scan_x);
    check_cuda(cudaGetLastError(), "detector_plane_minimum_kernel launch");
}

void launch_adaptive_filter(
    const double* input,
    const double* plane_minima,
    double* output,
    std::uint8_t* outcomes,
    std::size_t element_count,
    const phase_a::Dimensions4D& dimensions,
    unsigned int blocks)
{
    adaptive_median_s3_smax7_kernel<<<blocks, threads_per_block>>>(
        input,
        plane_minima,
        output,
        outcomes,
        element_count,
        dimensions.scan_y,
        dimensions.scan_x,
        dimensions.detector_y,
        dimensions.detector_x);
    check_cuda(cudaGetLastError(), "adaptive_median_s3_smax7_kernel launch");
}

AdaptiveMedianStatistics aggregate_outcomes(const std::vector<std::uint8_t>& outcomes)
{
    AdaptiveMedianStatistics statistics;
    statistics.total_outputs = outcomes.size();
    for (const std::uint8_t outcome : outcomes) {
        switch (outcome) {
        case finish_3_retain:
            ++statistics.finished_at_3x3;
            ++statistics.stage_b_retained_center;
            ++statistics.median_computations;
            break;
        case finish_3_replace:
            ++statistics.finished_at_3x3;
            ++statistics.stage_b_replaced_with_median;
            ++statistics.median_computations;
            break;
        case finish_5_retain:
            ++statistics.expanded_to_5x5;
            ++statistics.finished_at_5x5;
            ++statistics.stage_b_retained_center;
            statistics.median_computations += 2;
            break;
        case finish_5_replace:
            ++statistics.expanded_to_5x5;
            ++statistics.finished_at_5x5;
            ++statistics.stage_b_replaced_with_median;
            statistics.median_computations += 2;
            break;
        case finish_7_retain:
            ++statistics.expanded_to_5x5;
            ++statistics.expanded_to_7x7;
            ++statistics.finished_at_7x7;
            ++statistics.stage_b_retained_center;
            statistics.median_computations += 3;
            break;
        case finish_7_replace:
            ++statistics.expanded_to_5x5;
            ++statistics.expanded_to_7x7;
            ++statistics.finished_at_7x7;
            ++statistics.stage_b_replaced_with_median;
            statistics.median_computations += 3;
            break;
        case maximum_fallback:
            ++statistics.expanded_to_5x5;
            ++statistics.expanded_to_7x7;
            ++statistics.maximum_window_fallback;
            statistics.median_computations += 3;
            break;
        default:
            throw std::runtime_error("CUDA adaptive diagnostic returned an invalid outcome.");
        }
    }
    return statistics;
}

} // namespace

AdaptiveCudaKernelConfiguration adaptive_median_cuda_kernel_configuration(
    const phase_a::Dimensions4D& dimensions)
{
    const std::size_t element_count = checked_element_count(dimensions);
    const std::size_t plane_count = checked_multiply(
        dimensions.detector_y, dimensions.detector_x, "Detector-plane count overflow.");
    return {
        threads_per_block,
        checked_block_count(plane_count),
        checked_block_count(element_count),
    };
}

AdaptiveCudaResult adaptive_median_s3_smax7_cuda_baseline(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions)
{
    const auto wall_start = std::chrono::steady_clock::now();
    validate_input(input, dimensions);
    const AdaptiveCudaKernelConfiguration configuration =
        adaptive_median_cuda_kernel_configuration(dimensions);
    const std::size_t element_count = input.size();
    const std::size_t plane_count = dimensions.detector_y * dimensions.detector_x;
    const std::size_t bytes = element_count * sizeof(double);

    AdaptiveCudaResult result;
    result.output.resize(element_count);
    {
        DeviceBuffer<double> device_input(element_count);
        DeviceBuffer<double> device_output(element_count);
        DeviceBuffer<double> device_plane_minima(plane_count);
        Event total_start;
        Event h2d_stop;
        Event minimum_stop;
        Event adaptive_stop;
        Event total_stop;

        record(total_start, "cudaEventRecord before adaptive H2D");
        check_cuda(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice),
                   "adaptive pageable cudaMemcpy H2D");
        record(h2d_stop, "cudaEventRecord after adaptive H2D");

        launch_plane_minimum(
            device_input.get(), device_plane_minima.get(), dimensions,
            configuration.plane_minimum_blocks);
        record(minimum_stop, "cudaEventRecord after plane-minimum kernel");

        launch_adaptive_filter(
            device_input.get(), device_plane_minima.get(), device_output.get(), nullptr,
            element_count, dimensions, configuration.adaptive_filter_blocks);
        record(adaptive_stop, "cudaEventRecord after adaptive-filter kernel");

        check_cuda(cudaMemcpy(result.output.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost),
                   "adaptive pageable cudaMemcpy D2H");
        record(total_stop, "cudaEventRecord after adaptive D2H");
        check_cuda(cudaEventSynchronize(total_stop.get()), "cudaEventSynchronize adaptive one-shot path");

        result.timing.host_to_device = elapsed(total_start, h2d_stop);
        result.timing.plane_minimum_kernel = elapsed(h2d_stop, minimum_stop);
        result.timing.adaptive_filter_kernel = elapsed(minimum_stop, adaptive_stop);
        result.timing.device_to_host = elapsed(adaptive_stop, total_stop);
        result.timing.total_gpu_path = elapsed(total_start, total_stop);
    }
    const auto wall_stop = std::chrono::steady_clock::now();
    result.timing.native_wall =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    return result;
}

AdaptiveMedianDiagnosticResult adaptive_median_s3_smax7_cuda_baseline_diagnostics(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions)
{
    validate_input(input, dimensions);
    const AdaptiveCudaKernelConfiguration configuration =
        adaptive_median_cuda_kernel_configuration(dimensions);
    const std::size_t element_count = input.size();
    const std::size_t plane_count = dimensions.detector_y * dimensions.detector_x;
    const std::size_t bytes = element_count * sizeof(double);

    AdaptiveMedianDiagnosticResult result;
    result.output.resize(element_count);
    std::vector<std::uint8_t> outcomes(element_count);
    DeviceBuffer<double> device_input(element_count);
    DeviceBuffer<double> device_output(element_count);
    DeviceBuffer<double> device_plane_minima(plane_count);
    DeviceBuffer<std::uint8_t> device_outcomes(element_count);

    check_cuda(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice),
               "diagnostic adaptive cudaMemcpy H2D");
    launch_plane_minimum(
        device_input.get(), device_plane_minima.get(), dimensions,
        configuration.plane_minimum_blocks);
    launch_adaptive_filter(
        device_input.get(), device_plane_minima.get(), device_output.get(), device_outcomes.get(),
        element_count, dimensions, configuration.adaptive_filter_blocks);
    check_cuda(cudaMemcpy(result.output.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost),
               "diagnostic adaptive output cudaMemcpy D2H");
    check_cuda(cudaMemcpy(outcomes.data(), device_outcomes.get(), outcomes.size(), cudaMemcpyDeviceToHost),
               "diagnostic adaptive outcome cudaMemcpy D2H");
    result.statistics = aggregate_outcomes(outcomes);
    return result;
}

AdaptiveCudaKernelBenchmarkResult benchmark_adaptive_median_s3_smax7_cuda_kernels(
    const std::vector<double>& input,
    const phase_a::Dimensions4D& dimensions,
    std::size_t warmup_launch_count,
    std::size_t timed_launch_count)
{
    validate_input(input, dimensions);
    if (timed_launch_count == 0) {
        throw std::invalid_argument("Adaptive CUDA kernel benchmark requires timed launches.");
    }
    const AdaptiveCudaKernelConfiguration configuration =
        adaptive_median_cuda_kernel_configuration(dimensions);
    const std::size_t element_count = input.size();
    const std::size_t plane_count = dimensions.detector_y * dimensions.detector_x;
    const std::size_t bytes = element_count * sizeof(double);

    AdaptiveCudaKernelBenchmarkResult result;
    result.output.resize(element_count);
    result.plane_minimum_kernel_milliseconds.reserve(timed_launch_count);
    result.adaptive_filter_kernel_milliseconds.reserve(timed_launch_count);
    DeviceBuffer<double> device_input(element_count);
    DeviceBuffer<double> device_output(element_count);
    DeviceBuffer<double> device_plane_minima(plane_count);
    Event start;
    Event stop;

    check_cuda(cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice),
               "kernel benchmark adaptive cudaMemcpy H2D");
    for (std::size_t warmup = 0; warmup < warmup_launch_count; ++warmup) {
        launch_plane_minimum(
            device_input.get(), device_plane_minima.get(), dimensions,
            configuration.plane_minimum_blocks);
        launch_adaptive_filter(
            device_input.get(), device_plane_minima.get(), device_output.get(), nullptr,
            element_count, dimensions, configuration.adaptive_filter_blocks);
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after adaptive warm-up");

    for (std::size_t run = 0; run < timed_launch_count; ++run) {
        record(start, "cudaEventRecord before timed plane-minimum kernel");
        launch_plane_minimum(
            device_input.get(), device_plane_minima.get(), dimensions,
            configuration.plane_minimum_blocks);
        record(stop, "cudaEventRecord after timed plane-minimum kernel");
        check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize plane-minimum kernel");
        result.plane_minimum_kernel_milliseconds.push_back(elapsed(start, stop));

        record(start, "cudaEventRecord before timed adaptive-filter kernel");
        launch_adaptive_filter(
            device_input.get(), device_plane_minima.get(), device_output.get(), nullptr,
            element_count, dimensions, configuration.adaptive_filter_blocks);
        record(stop, "cudaEventRecord after timed adaptive-filter kernel");
        check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize adaptive-filter kernel");
        result.adaptive_filter_kernel_milliseconds.push_back(elapsed(start, stop));
    }
    check_cuda(cudaMemcpy(result.output.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost),
               "kernel benchmark adaptive cudaMemcpy D2H");
    return result;
}

} // namespace phase_b
