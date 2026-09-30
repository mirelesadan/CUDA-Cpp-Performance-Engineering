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

double inertia_for(const std::vector<float>& input, std::size_t d,
                   std::size_t k, const kmeans::Result& result,
                   const std::string& context) {
    double inertia = 0.0;
    for (std::size_t sample = 0; sample < result.labels.size(); ++sample) {
        const std::size_t centroid = static_cast<std::size_t>(result.labels[sample]);
        require(centroid < k, context + ": label range");
        for (std::size_t feature = 0; feature < d; ++feature) {
            const double difference =
                static_cast<double>(input[sample * d + feature]) -
                static_cast<double>(result.centroids[centroid * d + feature]);
            inertia += difference * difference;
        }
    }
    return inertia;
}

void require_frozen_contract(const kmeans::Result& actual,
                             const kmeans::Result& reference,
                             const std::vector<float>& input, std::size_t d,
                             std::size_t k, const std::string& context) {
    require(actual.labels == reference.labels, context + ": exact labels");
    require(actual.update_count == reference.update_count,
            context + ": update count");
    require(actual.converged == reference.converged,
            context + ": convergence flag");
    require(actual.centroids.size() == k * d, context + ": centroid shape");
    for (std::size_t feature = 0; feature < d; ++feature) {
        double scale = 1.0;
        for (std::size_t sample = 0; sample < actual.labels.size(); ++sample) {
            scale = std::max(scale,
                std::abs(static_cast<double>(input[sample * d + feature])));
        }
        for (std::size_t cluster = 0; cluster < k; ++cluster) {
            const auto index = cluster * d + feature;
            require(std::isfinite(actual.centroids[index]),
                    context + ": nonfinite centroid");
            const double difference =
                std::abs(static_cast<double>(actual.centroids[index]) -
                         static_cast<double>(reference.centroids[index]));
            require(difference <= 5e-6 * scale,
                    context + ": centroid tolerance");
        }
    }
    const double reference_inertia =
        inertia_for(input, d, k, reference, context);
    const double actual_inertia =
        inertia_for(input, d, k, actual, context);
    require(std::abs(actual_inertia - reference_inertia) <=
                2e-5 * std::max(1.0, std::abs(reference_inertia)),
            context + ": inertia tolerance");
    if (reference_inertia == 0.0) {
        require(actual_inertia == 0.0, context + ": exact zero inertia");
    }
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
        auto tiled = kmeans::kmeans_cuda_tiled_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap);
        const auto tiled_repeat = kmeans::kmeans_cuda_tiled_with_update_cap(
            fixture.input, n, fixture.d, fixture.k, fixture.update_cap);
        const std::string context = fixture.name;
        require_equal(cuda, serial, context + ": addressed serial parity");
        require_equal(repeat, cuda, context + ": repeatability");
        require_frozen_contract(tiled, serial, fixture.input, fixture.d,
                                fixture.k, context + ": tiled");
        require_equal(tiled_repeat, tiled, context + ": tiled repeatability");
        require(tiled.labels == fixture.expected_labels,
                context + ": tiled authoritative labels");
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
        require(tiled.centroids.data() != fixture.input.data(),
                context + ": tiled output aliases input");
        tiled.labels[0] = -1;
        tiled.centroids[0] = 99.0f;
        require_frozen_contract(tiled_repeat, serial, fixture.input,
                                fixture.d, fixture.k,
                                context + ": tiled independent output");
        cuda.labels[0] = -1;
        cuda.centroids[0] = 99.0f;
        require_equal(repeat, serial, context + ": independent result ownership");
        compared_labels += n;
    }
    std::cout << "Project 2 CUDA public fixtures passed: six cases, "
              << compared_labels << " exact labels; control centroid bits exact, "
              << "tiled centroids within frozen tolerance and repeatable\n";
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
    const auto odd_tiled = kmeans::kmeans_cuda_tiled(odd, 16, 3, 2);
    require(odd_serial.labels[7] == 0, "D=3 feature-order discriminator");
    require_equal(odd_cuda, odd_serial, "D=3 odd-feature order");
    require_frozen_contract(odd_tiled, odd_serial, odd, 3, 2,
                            "D=3 tiled odd-feature order");

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
        const auto tiled = kmeans::kmeans_cuda_tiled(even, 64, 4, k);
        require_equal(cuda, serial, "D=4 K=" + std::to_string(k));
        require_frozen_contract(tiled, serial, even, 4, k,
                                "D=4 tiled K=" + std::to_string(k));
    }
    require(std::memcmp(even.data(), before.data(),
                        even.size() * sizeof(float)) == 0,
            "D=4 edge input changed");
    std::cout << "Project 2 CUDA edge cases passed: D=1/3/4, K=2/31/32\n";
}

void test_tiled_boundaries() {
    for (const std::size_t n : {4095u, 4096u, 4097u, 8193u}) {
        const std::size_t d = n == 4097 ? 32u : 3u;
        const std::size_t k = n == 4097 ? 32u : 2u;
        std::vector<float> input(n * d);
        for (std::size_t sample = 0; sample < n; ++sample) {
            const auto cluster = sample * k / n;
            for (std::size_t feature = 0; feature < d; ++feature) {
                input[sample * d + feature] =
                    static_cast<float>(cluster * 8) +
                    static_cast<float>(feature) * 0.03125f;
            }
        }
        const auto before = input;
        const auto reference =
            kmeans::kmeans_serial_addressed(input, n, d, k);
        const auto tiled = kmeans::kmeans_cuda_tiled(input, n, d, k);
        const auto repeat = kmeans::kmeans_cuda_tiled(input, n, d, k);
        const auto context = "tile boundary N=" + std::to_string(n);
        require_frozen_contract(tiled, reference, input, d, k, context);
        require_equal(repeat, tiled, context + ": deterministic repeat");
        require(std::memcmp(input.data(), before.data(),
                            input.size() * sizeof(float)) == 0,
                context + ": input changed");
    }
    std::cout << "Project 2 CUDA tiled boundaries passed: N=4095/4096/4097/8193, D=3/32, K=2/32\n";
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
        test_tiled_boundaries();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 CUDA contract test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
