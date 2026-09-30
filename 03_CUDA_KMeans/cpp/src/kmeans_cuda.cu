#include "kmeans_cuda.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace kmeans {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Project 2 requires IEEE-754 binary32 float");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "Project 2 requires IEEE-754 binary64 double");
static_assert(sizeof(int) == 4, "CUDA labels and counts require 32-bit int");

constexpr std::size_t max_n = 1u << 20;
constexpr std::size_t max_d = 32;
constexpr std::size_t max_k = 32;
constexpr float max_magnitude = 1024.0f;
constexpr int block_size = 256;
constexpr int reduction_tile_samples = 4096;
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point first, Clock::time_point last) {
    return std::chrono::duration<double, std::milli>(last - first).count();
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + operation + ": " +
                                 cudaGetErrorString(status));
    }
}

void validate(const std::vector<float>& input, std::size_t n,
              std::size_t d, std::size_t k, std::size_t max_updates) {
    if (n < 2 || n > max_n || d < 1 || d > max_d ||
        k < 2 || k > max_k || k > n) {
        throw std::invalid_argument(
            "require 2 <= K <= min(N,32), K <= N <= 2^20, 1 <= D <= 32");
    }
    if (input.size() != n * d) {
        throw std::invalid_argument("input size must equal N * D");
    }
    if (max_updates < 1 || max_updates > 100) {
        throw std::invalid_argument("update cap must be between 1 and 100");
    }
    for (const float value : input) {
        if (!std::isfinite(value) || std::fabs(value) > max_magnitude) {
            throw std::invalid_argument(
                "input must be finite with absolute coordinates <= 1024");
        }
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)),
              "allocation");
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

class EventPair {
public:
    explicit EventPair(bool enabled) {
        if (enabled) {
            check(cudaEventCreate(&start_), "event creation (start)");
            try {
                check(cudaEventCreate(&stop_), "event creation (stop)");
            } catch (...) {
                cudaEventDestroy(start_);
                start_ = nullptr;
                throw;
            }
        }
    }

    ~EventPair() {
        if (stop_ != nullptr) cudaEventDestroy(stop_);
        if (start_ != nullptr) cudaEventDestroy(start_);
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    bool enabled() const { return start_ != nullptr; }
    cudaEvent_t start() const { return start_; }
    cudaEvent_t stop() const { return stop_; }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

class EventTriplet {
public:
    EventTriplet() {
        check(cudaEventCreate(&first_), "event creation (first)");
        try {
            check(cudaEventCreate(&middle_), "event creation (middle)");
            check(cudaEventCreate(&last_), "event creation (last)");
        } catch (...) {
            if (middle_ != nullptr) cudaEventDestroy(middle_);
            cudaEventDestroy(first_);
            throw;
        }
    }
    ~EventTriplet() {
        cudaEventDestroy(last_);
        cudaEventDestroy(middle_);
        cudaEventDestroy(first_);
    }
    EventTriplet(const EventTriplet&) = delete;
    EventTriplet& operator=(const EventTriplet&) = delete;
    cudaEvent_t first() const { return first_; }
    cudaEvent_t middle() const { return middle_; }
    cudaEvent_t last() const { return last_; }

private:
    cudaEvent_t first_ = nullptr;
    cudaEvent_t middle_ = nullptr;
    cudaEvent_t last_ = nullptr;
};

double event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed time");
    return static_cast<double>(milliseconds);
}

template <typename Action>
void stage(EventPair& events, CudaStageTimings* timings,
           double CudaStageTimings::*field, Action&& action) {
    if (events.enabled()) {
        check(cudaEventRecord(events.start()), "event record (start)");
    }
    action();
    if (events.enabled()) {
        check(cudaEventRecord(events.stop()), "event record (stop)");
        check(cudaEventSynchronize(events.stop()), "event synchronization");
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, events.start(), events.stop()),
              "event elapsed time");
        timings->*field += static_cast<double>(milliseconds);
    }
}

__global__ void initialize_centroids(const float* input, float* centroids,
                                     int n, int d, int k) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= k * d) return;
    const int cluster = element / d;
    const int feature = element % d;
    const int row = ((2 * cluster + 1) * n) / (2 * k);
    centroids[element] = input[row * d + feature];
}

__global__ void assign_samples(const float* input, const float* centroids,
                               int* labels, const int* previous_labels,
                               int* changed, int n, int d, int k) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= n) return;

    const float* const sample_row = input + sample * d;
    float best_distance = CUDART_INF_F;
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster) {
        const float* const centroid_row = centroids + cluster * d;
        float distance = 0.0f;
        for (int feature = 0; feature < d; ++feature) {
            // Explicit RN operations forbid FMA contraction/reassociation.
            // Preserve ordered binary32 subtract, square, then add.
            const float difference =
                __fsub_rn(sample_row[feature], centroid_row[feature]);
            const float square = __fmul_rn(difference, difference);
            distance = __fadd_rn(distance, square);
        }
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
        }
    }
    labels[sample] = best_cluster;
    if (previous_labels != nullptr && best_cluster != previous_labels[sample]) {
        atomicExch(changed, 1);  // Integer flag only; no unordered FP sums.
    }
}

__global__ void count_clusters(const int* labels, int* counts, int n, int k) {
    const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) return;
    int count = 0;
    for (int sample = 0; sample < n; ++sample) {
        if (labels[sample] == cluster) ++count;
    }
    counts[cluster] = count;
}

__global__ void update_centroids(const float* input, const int* labels,
                                 const int* counts, float* centroids,
                                 int n, int d, int k) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= k * d) return;
    const int cluster = element / d;
    const int feature = element % d;
    const int count = counts[cluster];
    if (count == 0) return;  // Preserve previous centroid bits exactly.

    double sum = 0.0;
    for (int sample = 0; sample < n; ++sample) {
        if (labels[sample] == cluster) {
            sum = __dadd_rn(sum, static_cast<double>(input[sample * d + feature]));
        }
    }
    const double mean = __ddiv_rn(sum, static_cast<double>(count));
    centroids[element] = __double2float_rn(mean);
}

// One block owns one (cluster, feature, sample tile) partial. Thread-local
// samples are visited in ascending order; the shared-memory tree is fixed.
__global__ void centroid_tile_partials(const float* input, const int* labels,
                                       double* partials, int n, int d,
                                       int tile_count) {
    const int tile = blockIdx.x;
    const int element = blockIdx.y;
    const int cluster = element / d;
    const int feature = element % d;
    const int first = tile * reduction_tile_samples;
    const int limit = first + reduction_tile_samples < n
                          ? first + reduction_tile_samples : n;
    double local_sum = 0.0;
    for (int sample = first + threadIdx.x; sample < limit;
         sample += block_size) {
        if (labels[sample] == cluster) {
            local_sum = __dadd_rn(
                local_sum, static_cast<double>(input[sample * d + feature]));
        }
    }
    __shared__ double tree[block_size];
    tree[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = block_size / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            tree[threadIdx.x] =
                __dadd_rn(tree[threadIdx.x], tree[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[element * tile_count + tile] = tree[0];
    }
}

__global__ void finalize_centroids(const double* partials, const int* counts,
                                   float* centroids, int d, int k,
                                   int tile_count) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= k * d) return;
    const int cluster = element / d;
    const int count = counts[cluster];
    if (count == 0) return;  // Preserve previous centroid bits exactly.
    double sum = 0.0;
    for (int tile = 0; tile < tile_count; ++tile) {
        sum = __dadd_rn(sum, partials[element * tile_count + tile]);
    }
    centroids[element] =
        __double2float_rn(__ddiv_rn(sum, static_cast<double>(count)));
}

void check_launch(const char* operation) {
    check(cudaGetLastError(), operation);
}

Result run(const std::vector<float>& input, std::size_t n,
           std::size_t d, std::size_t k, std::size_t max_updates,
           CudaStageTimings* timings, bool tiled_update = false) {
    const auto wall_start = Clock::now();
    validate(input, n, d, k, max_updates);

    const int ni = static_cast<int>(n);
    const int di = static_cast<int>(d);
    const int ki = static_cast<int>(k);
    const int sample_blocks = (ni + block_size - 1) / block_size;
    const int centroid_blocks = (ki * di + block_size - 1) / block_size;
    const int cluster_blocks = (ki + block_size - 1) / block_size;
    const int tile_count = (ni + reduction_tile_samples - 1) /
                           reduction_tile_samples;
    Result result;

    {
        const auto setup_start = Clock::now();
        EventPair events(timings != nullptr);
        DeviceBuffer<float> device_input(n * d);
        DeviceBuffer<float> device_centroids(k * d);
        DeviceBuffer<int> device_labels_a(n);
        DeviceBuffer<int> device_labels_b(n);
        DeviceBuffer<int> device_counts(k);
        DeviceBuffer<int> device_changed(1);
        std::unique_ptr<DeviceBuffer<double>> partials;
        if (tiled_update) {
            partials = std::make_unique<DeviceBuffer<double>>(
                static_cast<std::size_t>(ki) * di * tile_count);
        }
        if (timings != nullptr) {
            timings->setup_ms = elapsed_ms(setup_start, Clock::now());
        }

        stage(events, timings, &CudaStageTimings::h2d_ms, [&] {
            check(cudaMemcpy(device_input.get(), input.data(), n * d * sizeof(float),
                             cudaMemcpyHostToDevice), "input H2D copy");
        });
        stage(events, timings, &CudaStageTimings::initialization_ms, [&] {
            initialize_centroids<<<centroid_blocks, block_size>>>(
                device_input.get(), device_centroids.get(), ni, di, ki);
            check_launch("centroid initialization launch");
        });

        int* current_labels = device_labels_a.get();
        int* next_labels = device_labels_b.get();
        stage(events, timings, &CudaStageTimings::initial_assignment_ms, [&] {
            assign_samples<<<sample_blocks, block_size>>>(
                device_input.get(), device_centroids.get(), current_labels,
                nullptr, nullptr, ni, di, ki);
            check_launch("initial assignment launch");
        });

        double convergence_flag_d2h_ms = 0.0;
        std::size_t completed_updates = 0;
        bool converged = false;
        for (std::size_t pass = 1; pass <= max_updates; ++pass) {
            stage(events, timings, &CudaStageTimings::count_ms, [&] {
                count_clusters<<<cluster_blocks, block_size>>>(
                    current_labels, device_counts.get(), ni, ki);
                check_launch("cluster count launch");
            });
            stage(events, timings, &CudaStageTimings::centroid_update_ms, [&] {
                if (tiled_update) {
                    centroid_tile_partials<<<dim3(tile_count, ki * di),
                                             block_size>>>(
                        device_input.get(), current_labels, partials->get(),
                        ni, di, tile_count);
                    check_launch("centroid tile partials launch");
                    finalize_centroids<<<centroid_blocks, block_size>>>(
                        partials->get(), device_counts.get(),
                        device_centroids.get(), di, ki, tile_count);
                } else {
                    update_centroids<<<centroid_blocks, block_size>>>(
                        device_input.get(), current_labels, device_counts.get(),
                        device_centroids.get(), ni, di, ki);
                }
                check_launch("centroid update launch");
            });

            stage(events, timings, &CudaStageTimings::convergence_ms, [&] {
                check(cudaMemset(device_changed.get(), 0, sizeof(int)),
                      "convergence flag reset");
            });
            stage(events, timings, &CudaStageTimings::reassignment_ms, [&] {
                assign_samples<<<sample_blocks, block_size>>>(
                    device_input.get(), device_centroids.get(), next_labels,
                    current_labels, device_changed.get(), ni, di, ki);
                check_launch("reassignment launch");
            });
            int changed = 0;
            const double convergence_before_copy =
                timings != nullptr ? timings->convergence_ms : 0.0;
            stage(events, timings, &CudaStageTimings::convergence_ms, [&] {
                check(cudaMemcpy(&changed, device_changed.get(), sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      "convergence flag D2H copy");
            });
            if (timings != nullptr) {
                convergence_flag_d2h_ms +=
                    timings->convergence_ms - convergence_before_copy;
            }

            // After the swap, current_labels owns the most recent labels,
            // including an unchanged pass and the final capped pass.
            std::swap(current_labels, next_labels);
            completed_updates = pass;
            if (changed == 0) {
                converged = true;
                break;
            }
        }

        result.labels.resize(n);
        result.centroids.resize(k * d);
        result.update_count = completed_updates;
        result.converged = converged;
        stage(events, timings, &CudaStageTimings::final_d2h_ms, [&] {
            check(cudaMemcpy(result.labels.data(), current_labels,
                             n * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                  "final labels D2H copy");
            check(cudaMemcpy(result.centroids.data(), device_centroids.get(),
                             k * d * sizeof(float), cudaMemcpyDeviceToHost),
                  "final centroids D2H copy");
        });

        if (timings != nullptr) {
            timings->device_algorithm_ms =
                timings->initialization_ms + timings->initial_assignment_ms +
                timings->count_ms + timings->centroid_update_ms +
                timings->reassignment_ms + timings->convergence_ms -
                convergence_flag_d2h_ms;
            timings->one_shot_gpu_ms = timings->h2d_ms +
                                       timings->device_algorithm_ms +
                                       convergence_flag_d2h_ms +
                                       timings->final_d2h_ms;
        }
    }  // Free local device buffers/events before complete wall timestamp.
    if (timings != nullptr) {
        timings->native_wall_ms = elapsed_ms(wall_start, Clock::now());
    }
    return result;
}

}  // namespace

Result kmeans_cuda(const std::vector<float>& input, std::size_t n,
                   std::size_t d, std::size_t k) {
    return run(input, n, d, k, 100, nullptr);
}

Result kmeans_cuda_with_update_cap(const std::vector<float>& input,
                                   std::size_t n, std::size_t d,
                                   std::size_t k, std::size_t max_updates) {
    return run(input, n, d, k, max_updates, nullptr);
}

CudaRun kmeans_cuda_diagnostic(const std::vector<float>& input, std::size_t n,
                               std::size_t d, std::size_t k,
                               std::size_t max_updates) {
    CudaRun output;
    output.result = run(input, n, d, k, max_updates, &output.timings);
    return output;
}

Result kmeans_cuda_tiled(const std::vector<float>& input, std::size_t n,
                         std::size_t d, std::size_t k) {
    return run(input, n, d, k, 100, nullptr, true);
}

Result kmeans_cuda_tiled_with_update_cap(const std::vector<float>& input,
                                         std::size_t n, std::size_t d,
                                         std::size_t k, std::size_t max_updates) {
    return run(input, n, d, k, max_updates, nullptr, true);
}

CudaRun kmeans_cuda_tiled_diagnostic(const std::vector<float>& input,
                                     std::size_t n, std::size_t d, std::size_t k,
                                     std::size_t max_updates) {
    CudaRun output;
    output.result = run(input, n, d, k, max_updates, &output.timings, true);
    return output;
}

CudaUpdatePairTimings benchmark_cuda_update_pair(
    const std::vector<float>& input, std::size_t n, std::size_t d,
    std::size_t k, std::size_t rounds) {
    validate(input, n, d, k, 100);
    if (rounds < 7 || rounds > 50) {
        throw std::invalid_argument("direct update rounds must be 7..50");
    }
    const int ni = static_cast<int>(n);
    const int di = static_cast<int>(d);
    const int ki = static_cast<int>(k);
    const int tile_count = (ni + reduction_tile_samples - 1) /
                           reduction_tile_samples;
    const int sample_blocks = (ni + block_size - 1) / block_size;
    const int centroid_blocks = (ki * di + block_size - 1) / block_size;
    const int cluster_blocks = (ki + block_size - 1) / block_size;

    DeviceBuffer<float> device_input(n * d);
    DeviceBuffer<float> control_centroids(k * d);
    DeviceBuffer<float> tiled_centroids(k * d);
    DeviceBuffer<int> labels(n);
    DeviceBuffer<int> counts(k);
    DeviceBuffer<double> partials(static_cast<std::size_t>(ki) * di * tile_count);
    EventTriplet events;
    check(cudaMemcpy(device_input.get(), input.data(), n * d * sizeof(float),
                     cudaMemcpyHostToDevice), "direct benchmark input H2D");
    initialize_centroids<<<centroid_blocks, block_size>>>(
        device_input.get(), control_centroids.get(), ni, di, ki);
    check_launch("direct benchmark initialization launch");
    check(cudaMemcpy(tiled_centroids.get(), control_centroids.get(),
                     k * d * sizeof(float), cudaMemcpyDeviceToDevice),
          "direct benchmark centroid copy");
    assign_samples<<<sample_blocks, block_size>>>(
        device_input.get(), control_centroids.get(), labels.get(),
        nullptr, nullptr, ni, di, ki);
    check_launch("direct benchmark initial assignment launch");
    count_clusters<<<cluster_blocks, block_size>>>(
        labels.get(), counts.get(), ni, ki);
    check_launch("direct benchmark count launch");
    check(cudaDeviceSynchronize(), "direct benchmark initialization sync");

    const auto launch_control = [&] {
        update_centroids<<<centroid_blocks, block_size>>>(
            device_input.get(), labels.get(), counts.get(),
            control_centroids.get(), ni, di, ki);
        check_launch("direct benchmark control launch");
    };
    const auto launch_tiled = [&] {
        centroid_tile_partials<<<dim3(tile_count, ki * di), block_size>>>(
            device_input.get(), labels.get(), partials.get(),
            ni, di, tile_count);
        check_launch("direct benchmark partial launch");
        finalize_centroids<<<centroid_blocks, block_size>>>(
            partials.get(), counts.get(), tiled_centroids.get(),
            di, ki, tile_count);
        check_launch("direct benchmark final launch");
    };
    for (int warmup = 0; warmup < 2; ++warmup) {
        launch_control();
        launch_tiled();
    }
    check(cudaDeviceSynchronize(), "direct benchmark warm-up sync");

    CudaUpdatePairTimings output;
    for (std::size_t round = 0; round < rounds; ++round) {
        const auto time_control = [&] {
            check(cudaEventRecord(events.first()), "control event start");
            launch_control();
            check(cudaEventRecord(events.last()), "control event stop");
            check(cudaEventSynchronize(events.last()), "control event sync");
            output.control_ms.push_back(event_ms(events.first(), events.last()));
        };
        const auto time_tiled = [&] {
            check(cudaEventRecord(events.first()), "tiled event start");
            centroid_tile_partials<<<dim3(tile_count, ki * di), block_size>>>(
                device_input.get(), labels.get(), partials.get(),
                ni, di, tile_count);
            check_launch("timed partial launch");
            check(cudaEventRecord(events.middle()), "partial event stop");
            finalize_centroids<<<centroid_blocks, block_size>>>(
                partials.get(), counts.get(), tiled_centroids.get(),
                di, ki, tile_count);
            check_launch("timed final launch");
            check(cudaEventRecord(events.last()), "tiled event stop");
            check(cudaEventSynchronize(events.last()), "tiled event sync");
            output.tiled_ms.push_back(event_ms(events.first(), events.last()));
            output.partial_ms.push_back(event_ms(events.first(), events.middle()));
            output.finalize_ms.push_back(event_ms(events.middle(), events.last()));
        };
        if (round % 2 == 0) {
            time_control();
            time_tiled();
        } else {
            time_tiled();
            time_control();
        }
    }
    return output;
}

}  // namespace kmeans
