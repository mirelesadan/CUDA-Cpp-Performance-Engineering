#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "fixed_median.hpp"

namespace phase_a {

struct CudaDeviceInfo {
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessor_count = 0;
    std::size_t global_memory_bytes = 0;
};

struct CudaTimingMilliseconds {
    double host_to_device = 0.0;
    double kernel = 0.0;
    double device_to_host = 0.0;
    double total_gpu_path = 0.0;
};

struct CudaFilterResult {
    std::vector<double> output;
    CudaTimingMilliseconds timing;
};

CudaDeviceInfo cuda_device_info();

CudaFilterResult fixed_median_3x3_cuda_baseline(
    const std::vector<double>& input,
    const Dimensions4D& dimensions);

} // namespace phase_a
