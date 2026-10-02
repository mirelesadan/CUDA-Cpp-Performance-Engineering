#include "kmeans_cuda.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

static_assert(!std::is_copy_constructible<kmeans::CudaKMeansBuffer>::value,
              "CUDA ownership must not be copied");
static_assert(!std::is_copy_assignable<kmeans::CudaKMeansBuffer>::value,
              "CUDA ownership must not be copied");
static_assert(!std::is_move_constructible<kmeans::CudaKMeansBuffer>::value,
              "This baseline deliberately has no move semantics");

void require(bool condition, const std::string& context) {
    if (!condition) throw std::runtime_error(context);
}

template <typename Exception, typename Function>
void require_rejected(Function operation, const std::string& context) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("operation unexpectedly accepted: " + context);
}

bool same_bits(const std::vector<float>& first,
               const std::vector<float>& second) {
    return first.size() == second.size() &&
           std::memcmp(first.data(), second.data(),
                       first.size() * sizeof(float)) == 0;
}

void require_equal(const kmeans::Result& actual, const kmeans::Result& expected,
                   const std::string& context) {
    require(actual.labels == expected.labels, context + ": exact labels");
    require(same_bits(actual.centroids, expected.centroids),
            context + ": centroid bits");
    require(actual.update_count == expected.update_count,
            context + ": update count");
    require(actual.converged == expected.converged,
            context + ": convergence flag");
}

std::size_t expected_bytes(std::size_t n, std::size_t d, std::size_t k) {
    // Input, centroids, two label arrays, counts, flag, FP64 centroid
    // partials, and integer count partials. No resident host input copy.
    return sizeof(float) * (n * d + k * d) +
           sizeof(std::int32_t) * (2 * n + k + 1) +
           sizeof(double) * k * d * ((n + 4095) / 4096) +
           sizeof(std::int32_t) * k * ((n + 1023) / 1024);
}

void test_invalid_dimensions() {
    using Invalid = std::invalid_argument;
    require_rejected<Invalid>([] { kmeans::CudaKMeansBuffer x(0, 1, 2); }, "N=0");
    require_rejected<Invalid>([] { kmeans::CudaKMeansBuffer x(1, 1, 2); }, "N=1");
    require_rejected<Invalid>([] { kmeans::CudaKMeansBuffer x(16, 0, 2); }, "D=0");
    require_rejected<Invalid>([] { kmeans::CudaKMeansBuffer x(16, 33, 2); }, "D>32");
    require_rejected<Invalid>([] { kmeans::CudaKMeansBuffer x(16, 1, 1); }, "K<2");
    require_rejected<Invalid>([] { kmeans::CudaKMeansBuffer x(16, 1, 17); }, "K>N");
    require_rejected<Invalid>([] { kmeans::CudaKMeansBuffer x(64, 1, 33); }, "K>32");
    require_rejected<Invalid>([] {
        kmeans::CudaKMeansBuffer x((1u << 20) + 1u, 1, 2);
    }, "N>2^20");
    require_rejected<Invalid>([] {
        kmeans::CudaKMeansBuffer x(std::numeric_limits<std::size_t>::max(), 32, 2);
    }, "oversized N rejected before product/allocation");
}

void test_state_and_invalid_uploads() {
    const std::vector<float> good = {0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 10, 11, 12, 30, 31, 32};
    const auto expected = kmeans::kmeans_cuda_parallel_count(good, 16, 1, 2);
    kmeans::CudaKMeansBuffer owner(16, 1, 2);
    require_rejected<std::logic_error>([&] { owner.fit(); }, "fit before upload");
    require_rejected<std::logic_error>([&] { owner.download(); }, "download before upload");
    require_rejected<std::invalid_argument>([&] { owner.upload({}); }, "empty input");
    require_rejected<std::logic_error>([&] { owner.fit(); }, "rejected upload does not initialize");
    owner.upload(good);
    require_rejected<std::logic_error>([&] { owner.download(); }, "download before fit");
    owner.fit();
    require_equal(owner.download(), expected, "first completed fit");

    // Validation fails before any state or device data is changed, so a
    // previously completed result remains downloadable and repeatable.
    for (const std::size_t size : {15u, 17u}) {
        require_rejected<std::invalid_argument>([&] {
            owner.upload(std::vector<float>(size, 0.0f));
        }, "wrong input size");
        require_equal(owner.download(), expected, "size rejection preserves output");
    }
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            1024.25f, -1024.25f}) {
        auto invalid = good;
        invalid.back() = bad;
        const auto before = invalid;
        require_rejected<std::invalid_argument>([&] { owner.upload(invalid); },
                                                "nonfinite/out-of-range input");
        require(same_bits(invalid, before), "rejected upload changed input");
        require_equal(owner.download(), expected, "value rejection preserves output");
    }
    owner.fit();
    require_equal(owner.download(), expected, "fit after rejected uploads");
    owner.upload(good);
    require_rejected<std::logic_error>([&] { owner.download(); },
                                       "successful reupload invalidates old output");
    owner.fit();
    require_equal(owner.download(), expected, "fit after successful reupload");

    for (const std::size_t cap : {0u, 101u}) {
        require_rejected<std::invalid_argument>([&] { owner.fit_with_update_cap(cap); },
                                                "invalid validation-only update cap");
        require_equal(owner.download(), expected, "invalid cap preserves output");
    }
    const auto capped = kmeans::kmeans_cuda_parallel_count_with_update_cap(good, 16, 1, 2, 1);
    for (std::size_t run = 0; run < 20; ++run) {
        const auto metadata = owner.fit_with_update_cap(1);
        require(metadata.update_count == 1 && !metadata.converged,
                "validation-only cap preserves nonconvergence");
        require_equal(owner.download(), capped, "cap-one no hidden update");
        owner.fit();
        require_equal(owner.download(), expected, "full fit after capped fit reseeds independently");
    }
}

void exercise(const std::vector<float>& values, std::size_t d, std::size_t k,
              const std::string& name) {
    const std::size_t n = values.size() / d;
    const auto reference = kmeans::kmeans_cuda_parallel_count(values, n, d, k);
    kmeans::CudaKMeansBuffer owner(n, d, k);
    require(owner.device_bytes() == expected_bytes(n, d, k), name + ": memory footprint");

    auto input = values;
    owner.upload(input);
    require(same_bits(input, values), name + ": upload changed input");
    // The owner must not retain, access, or revalidate this host vector.
    // Deliberately invalidating its values makes that error observable.
    std::fill(input.begin(), input.end(), std::numeric_limits<float>::quiet_NaN());
    for (std::size_t run = 0; run < 20; ++run) {
        const auto metadata = owner.fit();
        require(metadata.update_count == reference.update_count &&
                    metadata.converged == reference.converged,
                name + ": fit-only metadata");
        auto first = owner.download();
        const auto second = owner.download();
        require_equal(first, reference, name + ": fit " + std::to_string(run));
        require_equal(second, reference, name + ": repeated download");
        require(first.labels.data() != second.labels.data() &&
                    first.centroids.data() != second.centroids.data(),
                name + ": downloads alias host storage");
        first.labels[0] = -1;
        first.centroids[0] = 1024.0f;
        require_equal(second, reference, name + ": independent host result");
        require_equal(owner.download(), reference, name + ": download leaves resident output intact");
    }

    // Same-shape X2 replaces X1. Each fit starts from X2's frozen seed rows;
    // it must not continue from X1's final centroids or stale label roles.
    std::vector<float> replacement(values.size());
    for (std::size_t i = 0; i < replacement.size(); ++i) {
        replacement[i] = -values[i] * 0.5f + 0.25f;
    }
    const auto before = replacement;
    const auto replacement_reference =
        kmeans::kmeans_cuda_parallel_count(replacement, n, d, k);
    owner.upload(replacement);
    require(same_bits(replacement, before), name + ": replacement changed input");
    require_rejected<std::logic_error>([&] { owner.download(); },
                                       name + ": replacement invalidates output");
    for (std::size_t run = 0; run < 20; ++run) {
        const auto metadata = owner.fit();
        require(metadata.update_count == replacement_reference.update_count &&
                    metadata.converged == replacement_reference.converged,
                name + ": replacement metadata");
        require_equal(owner.download(), replacement_reference, name + ": replacement fit");
    }
    std::cout << "Resident CUDA contract passed: " << name
              << "; 20 X1 + 20 X2 fits, exact one-shot parity\n";
}

void test_edges_and_repeats() {
    exercise({-1024.0f, 1024.0f}, 1, 2, "N=K=2, inclusive magnitude bound");
    exercise({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 30, 31, 32},
             1, 2, "three-update reseeding");
    exercise(std::vector<float>(1025 * 3, -0.0f), 3, 32,
             "signed zero, empty clusters, partial count tile");
    const float small = 0.000244140625f;
    std::vector<float> ordered;
    for (int i = 0; i < 6; ++i) ordered.insert(ordered.end(), {1.0f, small, small});
    ordered.insert(ordered.end(), {2.0f, 2 * small, 2 * small, 0.0f, 0.0f, 0.0f});
    for (int i = 0; i < 8; ++i) ordered.insert(ordered.end(), {1.0f, 0.0f, 0.0f});
    exercise(ordered, 3, 2, "D=3 ordered FP32 additions");
    for (const std::size_t k : {31u, 32u}) {
        constexpr std::size_t n = 4097;
        constexpr std::size_t d = 32;
        std::vector<float> input(n * d);
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t column = 0; column < d; ++column) {
                input[row * d + column] = static_cast<float>(row * k / n) * 8.0f +
                                         static_cast<float>(column) * 0.03125f;
            }
        }
        exercise(input, d, k, "D=32 K=" + std::to_string(k) + " partial centroid tile");
    }
}

void test_representative_footprints() {
    const kmeans::CudaKMeansBuffer gpu(262144, 16, 16);
    require(gpu.device_bytes() == 19022916u, "GPU-oriented allocation total");
    const kmeans::CudaKMeansBuffer iterative(16384, 8, 8);
    require(iterative.device_bytes() == 658212u, "iterative allocation total");
    std::cout << "Resident CUDA allocation totals exact: 19022916 / 658212 bytes\n";
}

void test_device_affinity() {
    int count = 0;
    require(cudaGetDeviceCount(&count) == cudaSuccess, "query CUDA device count");
    if (count < 2) {
        std::cout << "Device-switch rejection not exercised: one CUDA device\n";
        return;
    }
    int original = 0;
    require(cudaGetDevice(&original) == cudaSuccess, "query current CUDA device");
    struct RestoreDevice {
        int device;
        ~RestoreDevice() { cudaSetDevice(device); }
    } restore{original};
    kmeans::CudaKMeansBuffer owner(2, 1, 2);
    owner.upload({0.0f, 1.0f});
    owner.fit();
    const auto expected = owner.download();
    require(cudaSetDevice((original + 1) % count) == cudaSuccess, "change CUDA device");
    require_rejected<std::logic_error>([&] { owner.fit(); }, "fit on wrong device");
    require_rejected<std::logic_error>([&] { owner.download(); }, "download on wrong device");
    require_rejected<std::logic_error>([&] { owner.upload({0.0f, 1.0f}); },
                                       "upload on wrong device");
    require(cudaSetDevice(original) == cudaSuccess, "restore CUDA device");
    require_equal(owner.download(), expected, "wrong-device rejection preserves output");
}

}  // namespace

int main() {
    try {
        test_invalid_dimensions();
        test_state_and_invalid_uploads();
        test_edges_and_repeats();
        test_representative_footprints();
        test_device_affinity();
        std::cout << "Project 2 supported resident CUDA owner contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 resident CUDA owner contract failed: " << error.what() << '\n';
        return 1;
    }
}
