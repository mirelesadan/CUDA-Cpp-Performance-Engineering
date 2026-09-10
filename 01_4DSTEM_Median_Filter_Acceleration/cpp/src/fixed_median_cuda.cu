#include "fixed_median_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t bytes)
    {
        check_cuda(cudaMallocHost(&pointer_, bytes), "cudaMallocHost");
    }

    ~PinnedHostBuffer()
    {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
        }
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

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

double time_copy(
    void* destination,
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind direction,
    const Event& start,
    const Event& end,
    const char* operation)
{
    record_and_check(start, "cudaEventRecord before transfer");
    check_cuda(cudaMemcpy(destination, source, bytes, direction), operation);
    record_and_check(end, "cudaEventRecord after transfer");
    synchronize_and_check(end, "cudaEventSynchronize after transfer");
    return elapsed_milliseconds(start, end);
}

CudaTimingMilliseconds time_pageable_path(
    const double* host_input,
    double* host_output,
    double* device_input,
    double* device_output,
    std::size_t bytes,
    std::size_t element_count,
    const Dimensions4D& dimensions,
    unsigned int block_count,
    const Event& start,
    const Event& end)
{
    CudaTimingMilliseconds timing;
    timing.host_to_device = time_copy(
        device_input,
        host_input,
        bytes,
        cudaMemcpyHostToDevice,
        start,
        end,
        "pageable cudaMemcpy H2D");
    timing.kernel = time_baseline_kernel_launch(
        device_input,
        device_output,
        element_count,
        dimensions,
        block_count,
        start,
        end);
    timing.device_to_host = time_copy(
        host_output,
        device_output,
        bytes,
        cudaMemcpyDeviceToHost,
        start,
        end,
        "pageable cudaMemcpy D2H");
    timing.total_gpu_path =
        timing.host_to_device + timing.kernel + timing.device_to_host;
    return timing;
}

CudaTimingMilliseconds time_resident_path(
    const double* host_input,
    double* host_output,
    double* device_input,
    double* device_output,
    std::size_t bytes,
    std::size_t element_count,
    const Dimensions4D& dimensions,
    unsigned int block_count,
    std::size_t iteration_count,
    const Event& start,
    const Event& end)
{
    CudaTimingMilliseconds timing;
    timing.host_to_device = time_copy(
        device_input,
        host_input,
        bytes,
        cudaMemcpyHostToDevice,
        start,
        end,
        "resident-path pageable cudaMemcpy H2D");

    record_and_check(start, "cudaEventRecord before resident kernel sequence");
    // Every operation reads the same resident input and overwrites the same output;
    // this amortizes transfers without changing the filter applied by each launch.
    for (std::size_t iteration = 0; iteration < iteration_count; ++iteration) {
        fixed_median_3x3_kernel<<<block_count, threads_per_block>>>(
            device_input,
            device_output,
            element_count,
            dimensions.scan_y,
            dimensions.scan_x,
            dimensions.detector_y,
            dimensions.detector_x);
        check_cuda(cudaGetLastError(), "resident fixed_median_3x3_kernel launch");
    }
    record_and_check(end, "cudaEventRecord after resident kernel sequence");
    synchronize_and_check(end, "cudaEventSynchronize after resident kernel sequence");
    timing.kernel = elapsed_milliseconds(start, end);

    timing.device_to_host = time_copy(
        host_output,
        device_output,
        bytes,
        cudaMemcpyDeviceToHost,
        start,
        end,
        "resident-path pageable cudaMemcpy D2H");
    timing.total_gpu_path =
        timing.host_to_device + timing.kernel + timing.device_to_host;
    return timing;
}

struct BenchmarkLaunch {
    std::size_t element_count = 0;
    std::size_t bytes = 0;
    unsigned int block_count = 0;
};

BenchmarkLaunch prepare_benchmark_launch(
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
        throw std::overflow_error(
            "CUDA transfer benchmark exceeds the one-dimensional grid limit.");
    }
    return {
        element_count,
        element_count * sizeof(double),
        static_cast<unsigned int>(block_count),
    };
}

} // namespace

struct CudaMedianBuffer::Impl {
    explicit Impl(const double* host_input, const Dimensions4D& input_dimensions)
        : dimensions(input_dimensions)
    {
        if (host_input == nullptr) {
            throw std::invalid_argument("CUDA resident input pointer must not be null.");
        }

        element_count = checked_element_count(dimensions);
        if (dimensions.scan_y >
                static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
            dimensions.scan_x >
                static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
            throw std::overflow_error(
                "Scan dimensions are too large for signed reflected indexing.");
        }

        check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
        cudaDeviceProp properties{};
        check_cuda(
            cudaGetDeviceProperties(&properties, cuda_device_index),
            "cudaGetDeviceProperties");
        const std::size_t required_blocks =
            (element_count + threads_per_block - 1) / threads_per_block;
        if (required_blocks >
            static_cast<std::size_t>(properties.maxGridSize[0])) {
            throw std::overflow_error(
                "CUDA resident buffer requires more one-dimensional blocks than the device supports.");
        }

        block_count = static_cast<unsigned int>(required_blocks);
        bytes = element_count * sizeof(double);
        device_input = std::make_unique<DeviceBuffer>(bytes);
        device_output = std::make_unique<DeviceBuffer>(bytes);
        check_cuda(
            cudaMemcpy(
                device_input->get(),
                host_input,
                bytes,
                cudaMemcpyHostToDevice),
            "resident CUDA buffer upload");
    }

    Dimensions4D dimensions;
    std::size_t element_count = 0;
    std::size_t bytes = 0;
    unsigned int block_count = 0;
    std::unique_ptr<DeviceBuffer> device_input;
    std::unique_ptr<DeviceBuffer> device_output;
    bool output_ready = false;
};

CudaMedianBuffer::CudaMedianBuffer(
    const double* host_input,
    const Dimensions4D& dimensions)
    : impl_(std::make_unique<Impl>(host_input, dimensions))
{
}

CudaMedianBuffer::~CudaMedianBuffer() = default;

void CudaMedianBuffer::filter()
{
    check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
    fixed_median_3x3_kernel<<<impl_->block_count, threads_per_block>>>(
        impl_->device_input->get(),
        impl_->device_output->get(),
        impl_->element_count,
        impl_->dimensions.scan_y,
        impl_->dimensions.scan_x,
        impl_->dimensions.detector_y,
        impl_->dimensions.detector_x);
    check_cuda(cudaGetLastError(), "resident fixed_median_3x3_kernel launch");
    check_cuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize after resident fixed_median_3x3_kernel");
    impl_->output_ready = true;
}

void CudaMedianBuffer::download(double* host_output) const
{
    if (host_output == nullptr) {
        throw std::invalid_argument("CUDA resident output pointer must not be null.");
    }
    if (!impl_->output_ready) {
        throw std::logic_error("filter() must be called before download().");
    }

    check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
    check_cuda(
        cudaMemcpy(
            host_output,
            impl_->device_output->get(),
            impl_->bytes,
            cudaMemcpyDeviceToHost),
        "resident CUDA buffer download");
}

Dimensions4D CudaMedianBuffer::dimensions() const
{
    return impl_->dimensions;
}

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

CudaPageablePathBenchmarkResult benchmark_fixed_median_3x3_cuda_pageable_path(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    std::size_t warmup_run_count,
    std::size_t timed_run_count)
{
    if (warmup_run_count == 0 || timed_run_count == 0) {
        throw std::invalid_argument(
            "CUDA pageable-path benchmark requires warm-up and timed runs.");
    }

    const BenchmarkLaunch launch = prepare_benchmark_launch(input, dimensions);
    CudaPageablePathBenchmarkResult result;
    result.output.resize(launch.element_count);
    result.timed_runs.reserve(timed_run_count);
    DeviceBuffer device_input(launch.bytes);
    DeviceBuffer device_output(launch.bytes);
    Event start;
    Event end;

    for (std::size_t run = 0; run < warmup_run_count; ++run) {
        static_cast<void>(time_pageable_path(
            input.data(),
            result.output.data(),
            device_input.get(),
            device_output.get(),
            launch.bytes,
            launch.element_count,
            dimensions,
            launch.block_count,
            start,
            end));
    }
    for (std::size_t run = 0; run < timed_run_count; ++run) {
        result.timed_runs.push_back(time_pageable_path(
            input.data(),
            result.output.data(),
            device_input.get(),
            device_output.get(),
            launch.bytes,
            launch.element_count,
            dimensions,
            launch.block_count,
            start,
            end));
    }
    return result;
}

CudaTransferCharacterizationResult characterize_fixed_median_3x3_cuda_transfers(
    const std::vector<double>& input,
    const Dimensions4D& dimensions,
    std::size_t pageable_warmup_count,
    std::size_t pageable_timed_count,
    const std::vector<std::size_t>& residency_iteration_counts,
    std::size_t residency_warmup_count,
    std::size_t residency_timed_count,
    std::size_t transfer_diagnostic_warmup_count,
    std::size_t transfer_diagnostic_timed_count)
{
    if (pageable_warmup_count == 0 || pageable_timed_count == 0 ||
        residency_warmup_count == 0 || residency_timed_count == 0 ||
        transfer_diagnostic_warmup_count == 0 ||
        transfer_diagnostic_timed_count == 0) {
        throw std::invalid_argument(
            "CUDA transfer characterization requires nonzero warm-up and timed counts.");
    }
    if (residency_iteration_counts.empty()) {
        throw std::invalid_argument(
            "CUDA transfer characterization requires residency iteration counts.");
    }
    for (const std::size_t iteration_count : residency_iteration_counts) {
        if (iteration_count == 0) {
            throw std::invalid_argument("Residency iteration counts must be nonzero.");
        }
    }

    const BenchmarkLaunch launch = prepare_benchmark_launch(input, dimensions);
    CudaTransferCharacterizationResult result;
    result.output.resize(launch.element_count);
    result.pageable_path_runs.reserve(pageable_timed_count);
    result.residency_results.reserve(residency_iteration_counts.size());
    for (const std::size_t iteration_count : residency_iteration_counts) {
        result.residency_results.push_back({iteration_count, {}});
        result.residency_results.back().timed_runs.reserve(residency_timed_count);
    }

    DeviceBuffer device_input(launch.bytes);
    DeviceBuffer device_output(launch.bytes);
    Event start;
    Event end;

    for (std::size_t run = 0; run < pageable_warmup_count; ++run) {
        static_cast<void>(time_pageable_path(
            input.data(),
            result.output.data(),
            device_input.get(),
            device_output.get(),
            launch.bytes,
            launch.element_count,
            dimensions,
            launch.block_count,
            start,
            end));
    }
    for (std::size_t run = 0; run < pageable_timed_count; ++run) {
        result.pageable_path_runs.push_back(time_pageable_path(
            input.data(),
            result.output.data(),
            device_input.get(),
            device_output.get(),
            launch.bytes,
            launch.element_count,
            dimensions,
            launch.block_count,
            start,
            end));
    }

    for (CudaResidencyBenchmarkResult& residency : result.residency_results) {
        for (std::size_t run = 0; run < residency_warmup_count; ++run) {
            static_cast<void>(time_resident_path(
                input.data(),
                result.output.data(),
                device_input.get(),
                device_output.get(),
                launch.bytes,
                launch.element_count,
                dimensions,
                launch.block_count,
                residency.iteration_count,
                start,
                end));
        }
    }
    for (std::size_t run = 0; run < residency_timed_count; ++run) {
        for (CudaResidencyBenchmarkResult& residency : result.residency_results) {
            residency.timed_runs.push_back(time_resident_path(
                input.data(),
                result.output.data(),
                device_input.get(),
                device_output.get(),
                launch.bytes,
                launch.element_count,
                dimensions,
                launch.block_count,
                residency.iteration_count,
                start,
                end));
        }
    }

    result.diagnostic_pageable_h2d_milliseconds.reserve(
        transfer_diagnostic_timed_count);
    result.diagnostic_pageable_d2h_milliseconds.reserve(
        transfer_diagnostic_timed_count);
    result.diagnostic_pinned_h2d_milliseconds.reserve(
        transfer_diagnostic_timed_count);
    result.diagnostic_pinned_d2h_milliseconds.reserve(
        transfer_diagnostic_timed_count);

    const auto time_pageable_transfers = [&](bool retain) {
        const double host_to_device = time_copy(
            device_input.get(),
            input.data(),
            launch.bytes,
            cudaMemcpyHostToDevice,
            start,
            end,
            "diagnostic pageable cudaMemcpy H2D");
        const double device_to_host = time_copy(
            result.output.data(),
            device_output.get(),
            launch.bytes,
            cudaMemcpyDeviceToHost,
            start,
            end,
            "diagnostic pageable cudaMemcpy D2H");
        if (retain) {
            result.diagnostic_pageable_h2d_milliseconds.push_back(host_to_device);
            result.diagnostic_pageable_d2h_milliseconds.push_back(device_to_host);
        }
    };

    try {
        PinnedHostBuffer pinned_input(launch.bytes);
        PinnedHostBuffer pinned_output(launch.bytes);
        std::copy(input.begin(), input.end(), pinned_input.get());

        const auto time_pinned_transfers = [&](bool retain) {
            const double host_to_device = time_copy(
                device_input.get(),
                pinned_input.get(),
                launch.bytes,
                cudaMemcpyHostToDevice,
                start,
                end,
                "diagnostic pinned cudaMemcpy H2D");
            const double device_to_host = time_copy(
                pinned_output.get(),
                device_output.get(),
                launch.bytes,
                cudaMemcpyDeviceToHost,
                start,
                end,
                "diagnostic pinned cudaMemcpy D2H");
            if (retain) {
                result.diagnostic_pinned_h2d_milliseconds.push_back(host_to_device);
                result.diagnostic_pinned_d2h_milliseconds.push_back(device_to_host);
            }
        };

        for (std::size_t run = 0; run < transfer_diagnostic_warmup_count; ++run) {
            if (run % 2 == 0) {
                time_pageable_transfers(false);
                time_pinned_transfers(false);
            }
            else {
                time_pinned_transfers(false);
                time_pageable_transfers(false);
            }
        }
        for (std::size_t run = 0; run < transfer_diagnostic_timed_count; ++run) {
            if (run % 2 == 0) {
                time_pageable_transfers(true);
                time_pinned_transfers(true);
            }
            else {
                time_pinned_transfers(true);
                time_pageable_transfers(true);
            }
        }
    }
    catch (const std::exception& error) {
        result.pinned_memory_error = error.what();
        for (std::size_t run = 0; run < transfer_diagnostic_warmup_count; ++run) {
            time_pageable_transfers(false);
        }
        for (std::size_t run = 0; run < transfer_diagnostic_timed_count; ++run) {
            time_pageable_transfers(true);
        }
    }

    return result;
}

} // namespace phase_a
