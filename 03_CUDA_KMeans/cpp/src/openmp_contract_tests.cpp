#include "kmeans_serial.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Fixture {
    const char* name;
    std::vector<float> input;
    std::size_t d;
    std::size_t k;
    std::size_t update_cap;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_rejected(Function operation, const std::string& name) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("invalid input accepted: " + name);
}

void require_equal(const kmeans::Result& actual, const kmeans::Result& expected,
                   const std::string& context) {
    require(actual.labels == expected.labels, context + ": labels");
    require(actual.centroids.size() == expected.centroids.size(),
            context + ": centroid size");
    require(std::memcmp(actual.centroids.data(), expected.centroids.data(),
                        actual.centroids.size() * sizeof(float)) == 0,
            context + ": centroid bits");
    require(actual.update_count == expected.update_count,
            context + ": update count");
    require(actual.converged == expected.converged,
            context + ": convergence flag");
}

std::vector<Fixture> public_fixtures() {
    const std::vector<float> moving = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 30, 31, 32};
    std::vector<float> fp32;
    for (int i = 0; i < 7; ++i) {
        fp32.insert(fp32.end(), {0.0f, 0.0001220703125f});
    }
    fp32.insert(fp32.end(), {0.5f, 0.0f});
    for (int i = 0; i < 8; ++i) {
        fp32.insert(fp32.end(), {1.0f, 0.0f});
    }
    return {
        {"separated_signed_and_limit", {
            -1024, 0, -1023, -1, -1022, 0, -1021, 1,
            -1020, 0, -1019, -1, -1018, 0, -1017, 1,
            1017, 0, 1018, -1, 1019, 0, 1020, 1,
            1021, 0, 1022, -1, 1023, 0, 1024, 1}, 2, 2, 100},
        {"exact_tie_duplicate_seeds_empty_cluster", {
            0, 3, 3, 3, 3, 3, 6, 3, 3, 9, 9, 9, 9, 9, 9, 9}, 1, 3, 100},
        {"singleton_cluster_zero_inertia", {
            0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 100, 10, 10},
            1, 3, 100},
        {"three_updates", moving, 1, 2, 100},
        {"update_cap_without_hidden_update", moving, 1, 2, 1},
        {"fp32_distance_rounding", fp32, 2, 2, 100},
    };
}

void test_fixtures(int thread_count) {
    for (const auto& fixture : public_fixtures()) {
        const auto before = fixture.input;
        const auto n = fixture.input.size() / fixture.d;
        const auto serial = kmeans::kmeans_serial_addressed_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap);
        auto parallel = kmeans::kmeans_openmp_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap,
            thread_count);
        const auto repeated = kmeans::kmeans_openmp_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap,
            thread_count);
        const std::string context = std::string(fixture.name) + " at " +
                                    std::to_string(thread_count) + " threads";
        require_equal(parallel, serial, context);
        require_equal(repeated, parallel, context + " repeatability");
        require(std::memcmp(fixture.input.data(), before.data(),
                            fixture.input.size() * sizeof(float)) == 0,
                context + ": input changed");
        require(parallel.centroids.data() != fixture.input.data(),
                context + ": output aliases input");
        parallel.labels[0] = -1;
        parallel.centroids[0] = 99.0f;
        require(repeated.labels[0] == serial.labels[0] &&
                std::memcmp(repeated.centroids.data(), serial.centroids.data(),
                            repeated.centroids.size() * sizeof(float)) == 0,
                context + ": outputs not independently owned");
    }
}

void test_three_feature_order(int thread_count) {
    // Seed rows 4 and 12 become [1,2^-12,2^-12] and [1,0,0].
    // For row 7, ordered FP32 adds round each 2^-24 term away and choose
    // cluster 0; combining the small terms first would choose cluster 1.
    const float small = 0.000244140625f;
    std::vector<float> input;
    for (int i = 0; i < 6; ++i) {
        input.insert(input.end(), {1.0f, small, small});
    }
    input.insert(input.end(), {2.0f, 2 * small, 2 * small});
    input.insert(input.end(), {0.0f, 0.0f, 0.0f});
    for (int i = 0; i < 8; ++i) {
        input.insert(input.end(), {1.0f, 0.0f, 0.0f});
    }
    const auto serial = kmeans::kmeans_serial_addressed(input, 16, 3, 2);
    const auto parallel = kmeans::kmeans_openmp(input, 16, 3, 2, thread_count);
    require(serial.labels[7] == 0, "D=3 feature-order discriminator");
    require_equal(parallel, serial, "D=3 feature-order parity");
}

void test_invalid_inputs() {
    const std::vector<float> good(32, 0.0f);
    const int beyond_processors = omp_get_num_procs() + 1;
    require_rejected([&] { kmeans::kmeans_openmp(good, 16, 2, 2, 0); },
                     "zero threads");
    require_rejected([&] { kmeans::kmeans_openmp(good, 16, 2, 2, -1); },
                     "negative threads");
    require_rejected([&] {
        kmeans::kmeans_openmp(good, 16, 2, 2, beyond_processors);
    }, "threads exceed available processors");
    require_rejected([&] { kmeans::kmeans_openmp(good, 16, 2, 1, 1); }, "K=1");
    require_rejected([&] { kmeans::kmeans_openmp(good, 16, 2, 17, 1); }, "K>N");
    require_rejected([&] { kmeans::kmeans_openmp(good, 16, 0, 2, 1); }, "D=0");
    require_rejected([&] { kmeans::kmeans_openmp(good, 16, 33, 2, 1); }, "D>32");
    require_rejected([&] { kmeans::kmeans_openmp(good, 16, 1, 2, 1); }, "input size");
    require_rejected([&] {
        kmeans::kmeans_openmp_with_update_cap(good, 16, 2, 2, 0, 1);
    }, "zero update cap");
    require_rejected([&] {
        kmeans::kmeans_openmp_with_update_cap(good, 16, 2, 2, 101, 1);
    }, "update cap over 100");
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            1024.25f, -1024.25f}) {
        auto input = good;
        input[0] = bad;
        require_rejected([&] { kmeans::kmeans_openmp(input, 16, 2, 2, 1); },
                         "invalid value");
    }
}

}  // namespace

int main() {
    try {
        test_invalid_inputs();
        for (const int count : {1, 2, 4, 8, 16, 20}) {
            if (count > omp_get_num_procs()) {
                continue;
            }
            test_fixtures(count);
            test_three_feature_order(count);
            std::cout << "Project 2 OpenMP contracts passed at " << count
                      << " threads\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 OpenMP contract test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
