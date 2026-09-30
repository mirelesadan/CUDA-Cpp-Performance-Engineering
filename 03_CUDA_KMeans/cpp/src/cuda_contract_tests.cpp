#include "kmeans_cuda.hpp"
#include "kmeans_serial.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    std::vector<std::int32_t> expected_labels;
    std::vector<float> expected_centroids;
    std::size_t expected_updates;
    bool expected_converged;
    double expected_inertia;
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

double validation_inertia(const Fixture& fixture, const kmeans::Result& result) {
    // The validation metric uses double-precision distances; this is separate
    // from the frozen ordered-FP32 arithmetic used to assign labels.
    double inertia = 0.0;
    for (std::size_t sample = 0; sample < result.labels.size(); ++sample) {
        const std::size_t centroid =
            static_cast<std::size_t>(result.labels[sample]);
        require(centroid < fixture.k, std::string(fixture.name) + ": label range");
        for (std::size_t feature = 0; feature < fixture.d; ++feature) {
            const double difference =
                static_cast<double>(fixture.input[sample * fixture.d + feature]) -
                static_cast<double>(result.centroids[centroid * fixture.d + feature]);
            inertia += difference * difference;
        }
    }
    return inertia;
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
            1021, 0, 1022, -1, 1023, 0, 1024, 1},
            2, 2, 100, {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
            {-1020.5f, 0.0f, 1020.5f, 0.0f}, 1, true, 92.0},
        {"exact_tie_duplicate_seeds_empty_cluster", {
            0, 3, 3, 3, 3, 3, 6, 3, 3, 9, 9, 9, 9, 9, 9, 9},
            1, 3, 100, {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2},
            {3.0f, 3.0f, 9.0f}, 1, true, 18.0},
        {"singleton_cluster_zero_inertia", {
            0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 100, 10, 10},
            1, 3, 100, {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1},
            {0.0f, 10.0f, 100.0f}, 1, true, 0.0},
        {"three_updates", moving,
            1, 2, 100, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
            {6.0f, 31.0f}, 3, true, 184.0},
        {"update_cap_without_hidden_update", moving,
            1, 2, 1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
            {4.0f, 19.285715103149414f}, 1, false, 636.7550564980047},
        {"fp32_distance_rounding", fp32,
            2, 2, 100, {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
            {0.0625f, 0.0001068115234375f, 1.0f, 0.0f},
            1, true, 0.21875001303851604},
    };
}

void test_public_fixtures() {
    std::size_t compared_labels = 0;
    for (const auto& fixture : public_fixtures()) {
        const auto before = fixture.input;
        const auto n = fixture.input.size() / fixture.d;
        const auto serial = kmeans::kmeans_serial_addressed_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap);
        auto cuda = kmeans::kmeans_cuda_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap);
        const auto repeat = kmeans::kmeans_cuda_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap);
        const std::string context = fixture.name;
        require_equal(cuda, serial, context + ": addressed serial parity");
        require_equal(repeat, cuda, context + ": repeatability");
        require(cuda.labels == fixture.expected_labels,
                context + ": authoritative labels");
        require(cuda.centroids.size() == fixture.expected_centroids.size() &&
                std::memcmp(cuda.centroids.data(), fixture.expected_centroids.data(),
                            cuda.centroids.size() * sizeof(float)) == 0,
                context + ": authoritative centroid bits");
        require(cuda.update_count == fixture.expected_updates &&
                cuda.converged == fixture.expected_converged,
                context + ": authoritative termination");
        const double inertia = validation_inertia(fixture, cuda);
        require(std::abs(inertia - fixture.expected_inertia) <=
                    2e-5 * std::max(1.0, std::abs(fixture.expected_inertia)),
                context + ": validation inertia");
        if (fixture.expected_inertia == 0.0) {
            require(inertia == 0.0, context + ": exact zero inertia");
        }
        require(std::memcmp(fixture.input.data(), before.data(),
                            before.size() * sizeof(float)) == 0,
                context + ": input changed");
        require(cuda.centroids.data() != fixture.input.data(),
                context + ": output aliases input");
        cuda.labels[0] = -1;
        cuda.centroids[0] = 99.0f;
        require_equal(repeat, serial, context + ": independent result ownership");
        compared_labels += n;
    }
    std::cout << "Project 2 CUDA public fixtures passed: six cases, "
              << compared_labels << " exact labels, zero centroid-bit mismatches\n";
}

void test_feature_and_cluster_edges() {
    // D=1 is covered by public fixtures. D=3 also distinguishes ordered
    // FP32 feature additions from reassociation of the two small terms.
    const float small = 0.000244140625f;
    std::vector<float> odd;
    for (int i = 0; i < 6; ++i) {
        odd.insert(odd.end(), {1.0f, small, small});
    }
    odd.insert(odd.end(), {2.0f, 2 * small, 2 * small});
    odd.insert(odd.end(), {0.0f, 0.0f, 0.0f});
    for (int i = 0; i < 8; ++i) {
        odd.insert(odd.end(), {1.0f, 0.0f, 0.0f});
    }
    const auto odd_serial = kmeans::kmeans_serial_addressed(odd, 16, 3, 2);
    const auto odd_cuda = kmeans::kmeans_cuda(odd, 16, 3, 2);
    require(odd_serial.labels[7] == 0, "D=3 feature-order discriminator");
    require_equal(odd_cuda, odd_serial, "D=3 odd-feature order");

    std::vector<float> even;
    for (int sample = 0; sample < 64; ++sample) {
        for (int feature = 0; feature < 4; ++feature) {
            even.push_back(static_cast<float>(
                (sample * 17 + feature * 11) % 47 - 23));
        }
    }
    const auto before = even;
    for (const std::size_t k : {31u, 32u}) {
        const auto serial = kmeans::kmeans_serial_addressed(even, 64, 4, k);
        const auto cuda = kmeans::kmeans_cuda(even, 64, 4, k);
        require_equal(cuda, serial, "D=4 K=" + std::to_string(k));
    }
    require(std::memcmp(even.data(), before.data(),
                        even.size() * sizeof(float)) == 0,
            "D=4 edge input changed");
    std::cout << "Project 2 CUDA edge cases passed: D=1/3/4, K=2/31/32\n";
}

void test_invalid_inputs() {
    const std::vector<float> good(32, 0.0f);
    require_rejected([&] { kmeans::kmeans_cuda(good, 16, 2, 1); }, "K=1");
    require_rejected([&] { kmeans::kmeans_cuda(good, 16, 2, 17); }, "K>N");
    require_rejected([&] { kmeans::kmeans_cuda(good, 16, 0, 2); }, "D=0");
    require_rejected([&] { kmeans::kmeans_cuda(good, 16, 33, 2); }, "D>32");
    require_rejected([&] {
        kmeans::kmeans_cuda(good, (1u << 20) + 1u, 2, 2);
    }, "N>2^20");
    require_rejected([&] { kmeans::kmeans_cuda(good, 16, 1, 2); }, "input size");
    require_rejected([&] {
        kmeans::kmeans_cuda_with_update_cap(good, 16, 2, 2, 0);
    }, "zero update cap");
    require_rejected([&] {
        kmeans::kmeans_cuda_with_update_cap(good, 16, 2, 2, 101);
    }, "update cap over 100");
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            1024.25f, -1024.25f}) {
        auto input = good;
        input[0] = bad;
        require_rejected([&] { kmeans::kmeans_cuda(input, 16, 2, 2); },
                         "nonfinite or out-of-range value");
    }
}

}  // namespace

int main() {
    try {
        test_invalid_inputs();
        test_public_fixtures();
        test_feature_and_cluster_edges();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 CUDA contract test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
