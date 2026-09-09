#include "fixed_median_cuda.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace phase_a {
namespace {

constexpr int cuda_device_index = 0;
constexpr unsigned int threads_per_block = 256;

void check_cuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result));
    }
}

std::size_t checked_element_count(const Dimensions4D& dimensions)
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
            throw std::invalid_argument("CUDA median requires four nonempty dimensions.");
        }
        if (count > std::numeric_limits<std::size_t>::max() / size) {
            throw std::overflow_error("The 4D shape product exceeds std::size_t.");
        }
        count *= size;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::overflow_error("CUDA median byte count exceeds std::size_t.");
    }

    return count;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes)
    {
        check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    double* get()
    {
        return static_cast<double*>(pointer_);
    }

private:
    void* pointer_ = nullptr;
};

class Event {
public:
    Event()
    {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    ~Event()
    {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const
    {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

double elapsed_milliseconds(const Event& start, const Event& end)
{
    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
        "cudaEventElapsedTime");
    return static_cast<double>(milliseconds);
}

__device__ std::size_t reflect_index_device(long long index, std::size_t size)
{
    const long long extent = static_cast<long long>(size);
    while (index < 0 || index >= extent) {
        if (index < 0) {
            index = -(index + 1);
        }
        else {
            index = extent - 1 - (index - extent);
        }
    }
    return static_cast<std::size_t>(index);
}

__device__ void compare_swap_device(double& left, double& right)
{
    if (right < left) {
        const double temporary = left;
        left = right;
        right = temporary;
    }
}

__device__ double median_of_nine_device(double* values)
{
    // The same fixed 19-comparator selection network used by the optimized CPU path.
    compare_swap_device(values[1], values[2]);
    compare_swap_device(values[4], values[5]);
    compare_swap_device(values[7], values[8]);
    compare_swap_device(values[0], values[1]);
    compare_swap_device(values[3], values[4]);
    compare_swap_device(values[6], values[7]);
    compare_swap_device(values[1], values[2]);
    compare_swap_device(values[4], values[5]);
    compare_swap_device(values[7], values[8]);
    compare_swap_device(values[0], values[3]);
    compare_swap_device(values[5], values[8]);
    compare_swap_device(values[4], values[7]);
    compare_swap_device(values[3], values[6]);
    compare_swap_device(values[1], values[4]);
    compare_swap_device(values[2], values[5]);
    compare_swap_device(values[4], values[7]);
    compare_swap_device(values[4], values[2]);
    compare_swap_device(values[6], values[4]);
    compare_swap_device(values[4], values[2]);
    return values[4];
}

__global__ void fixed_median_3x3_kernel(
    const double* input,
    double* output,
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

    std::size_t remaining = output_index;
    const std::size_t detector_x = remaining % detector_x_size;
    remaining /= detector_x_size;
    const std::size_t detector_y = remaining % detector_y_size;
    remaining /= detector_y_size;
    const std::size_t scan_x = remaining % scan_x_size;
    const std::size_t scan_y = remaining / scan_x_size;

    const std::size_t detector_plane_stride = detector_y_size * detector_x_size;
    const std::size_t scan_y_stride = scan_x_size * detector_plane_stride;
    const std::size_t detector_offset = detector_y * detector_x_size + detector_x;

    double neighborhood[9];
    std::size_t neighborhood_index = 0;
    for (long long offset_y = -1; offset_y <= 1; ++offset_y) {
        const std::size_t reflected_y = reflect_index_device(
            static_cast<long long>(scan_y) + offset_y,
            scan_y_size);
        const std::size_t reflected_scan_y_base = reflected_y * scan_y_stride;

        for (long long offset_x = -1; offset_x <= 1; ++offset_x) {
            const std::size_t reflected_x = reflect_index_device(
                static_cast<long long>(scan_x) + offset_x,
                scan_x_size);
            const std::size_t input_scan_base =
                reflected_scan_y_base + reflected_x * detector_plane_stride;
            neighborhood[neighborhood_index] = input[input_scan_base + detector_offset];
            ++neighborhood_index;
        }
    }

    output[output_index] = median_of_nine_device(neighborhood);
}

void record_and_check(const Event& event, const char* operation)
{
    check_cuda(cudaEventRecord(event.get()), operation);
}

void synchronize_and_check(const Event& event, const char* operation)
{
    check_cuda(cudaEventSynchronize(event.get()), operation);
}

double time_baseline_kernel_launch(
    const double* input,
    double* output,
    std::size_t element_count,
    const Dimensions4D& dimensions,
    unsigned int block_count,
    const Event& start,
    const Event& end)
{
    record_and_check(start, "cudaEventRecord before baseline kernel");
    fixed_median_3x3_kernel<<<block_count, threads_per_block>>>(
        input,
        output,
        element_count,
        dimensions.scan_y,
        dimensions.scan_x,
        dimensions.detector_y,
        dimensions.detector_x);
    check_cuda(cudaGetLastError(), "fixed_median_3x3_kernel launch");
    record_and_check(end, "cudaEventRecord after baseline kernel");
    synchronize_and_check(end, "cudaEventSynchronize after baseline kernel");
    return elapsed_milliseconds(start, end);
}

} // namespace

CudaDeviceInfo cuda_device_info()
{
    check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, cuda_device_index),
        "cudaGetDeviceProperties");
    return {
        properties.name,
        properties.major,
        properties.minor,
        properties.multiProcessorCount,
        properties.totalGlobalMem,
    };
}

CudaFilterResult fixed_median_3x3_cuda_baseline(
    const std::vector<double>& input,
    const Dimensions4D& dimensions)
{
    const std::size_t element_count = checked_element_count(dimensions);
    if (input.size() != element_count) {
        throw std::invalid_argument("Input element count does not match the supplied 4D dimensions.");
    }
    if (dimensions.scan_y > static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
        dimensions.scan_x > static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
        throw std::overflow_error("Scan dimensions are too large for signed reflected indexing.");
    }

    check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, cuda_device_index),
        "cudaGetDeviceProperties");

    const std::size_t block_count =
        (element_count + threads_per_block - 1) / threads_per_block;
    if (block_count > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::overflow_error("CUDA baseline requires more one-dimensional blocks than the device supports.");
    }

    const std::size_t bytes = element_count * sizeof(double);
    std::vector<double> output(element_count);
    DeviceBuffer device_input(bytes);
    DeviceBuffer device_output(bytes);
    Event start;
    Event end;
    CudaTimingMilliseconds timing;

    record_and_check(start, "cudaEventRecord before H2D");
    check_cuda(
        cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
    record_and_check(end, "cudaEventRecord after H2D");
    synchronize_and_check(end, "cudaEventSynchronize after H2D");
    timing.host_to_device = elapsed_milliseconds(start, end);

    record_and_check(start, "cudaEventRecord before kernel");
    fixed_median_3x3_kernel<<<static_cast<unsigned int>(block_count), threads_per_block>>>(
        device_input.get(),
        device_output.get(),
        element_count,
        dimensions.scan_y,
        dimensions.scan_x,
        dimensions.detector_y,
        dimensions.detector_x);
    check_cuda(cudaGetLastError(), "fixed_median_3x3_kernel launch");
    record_and_check(end, "cudaEventRecord after kernel");
    synchronize_and_check(end, "cudaEventSynchronize after kernel");
    timing.kernel = elapsed_milliseconds(start, end);

    record_and_check(start, "cudaEventRecord before D2H");
    check_cuda(
        cudaMemcpy(output.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
    record_and_check(end, "cudaEventRecord after D2H");
    synchronize_and_check(end, "cudaEventSynchronize after D2H");
    timing.device_to_host = elapsed_milliseconds(start, end);
    timing.total_gpu_path =
        timing.host_to_device + timing.kernel + timing.device_to_host;

    return {std::move(output), timing};
}

CudaKernelTimingSequence benchmark_fixed_median_3x3_cuda_baseline_kernel(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    std::size_t warmup_launch_count,
    std::size_t timed_launch_count,
    unsigned int idle_milliseconds,
    std::size_t post_idle_launch_count)
{
    if (warmup_launch_count == 0 || timed_launch_count == 0) {
        throw std::invalid_argument(
            "CUDA kernel benchmark requires warm-up and timed launches.");
    }

    const std::size_t element_count = checked_element_count(dimensions);
    if (input.size() != element_count) {
        throw std::invalid_argument("Input element count does not match the supplied 4D dimensions.");
    }
    if (dimensions.scan_y > static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
        dimensions.scan_x > static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
        throw std::overflow_error("Scan dimensions are too large for signed reflected indexing.");
    }

    check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, cuda_device_index),
        "cudaGetDeviceProperties");
    const std::size_t block_count =
        (element_count + threads_per_block - 1) / threads_per_block;
    if (block_count > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::overflow_error(
            "CUDA baseline requires more one-dimensional blocks than the device supports.");
    }

    const std::size_t bytes = element_count * sizeof(double);
    CudaKernelTimingSequence result;
    result.output.resize(element_count);
    result.warmup_kernel_milliseconds.reserve(warmup_launch_count);
    result.steady_kernel_milliseconds.reserve(timed_launch_count);
    result.post_idle_kernel_milliseconds.reserve(post_idle_launch_count);

    DeviceBuffer device_input(bytes);
    DeviceBuffer device_output(bytes);
    Event start;
    Event end;
    check_cuda(
        cudaMemcpy(device_input.get(), input.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D before kernel benchmark");

    const unsigned int launch_block_count = static_cast<unsigned int>(block_count);
    for (std::size_t launch = 0; launch < warmup_launch_count; ++launch) {
        result.warmup_kernel_milliseconds.push_back(time_baseline_kernel_launch(
            device_input.get(),
            device_output.get(),
            element_count,
            dimensions,
            launch_block_count,
            start,
            end));
    }
    for (std::size_t launch = 0; launch < timed_launch_count; ++launch) {
        result.steady_kernel_milliseconds.push_back(time_baseline_kernel_launch(
            device_input.get(),
            device_output.get(),
            element_count,
            dimensions,
            launch_block_count,
            start,
            end));
    }

    if (idle_milliseconds != 0 && post_idle_launch_count != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(idle_milliseconds));
        for (std::size_t launch = 0; launch < post_idle_launch_count; ++launch) {
            result.post_idle_kernel_milliseconds.push_back(time_baseline_kernel_launch(
                device_input.get(),
                device_output.get(),
                element_count,
                dimensions,
                launch_block_count,
                start,
                end));
        }
    }

    check_cuda(
        cudaMemcpy(result.output.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H after kernel benchmark");
    return result;
}

} // namespace phase_a
