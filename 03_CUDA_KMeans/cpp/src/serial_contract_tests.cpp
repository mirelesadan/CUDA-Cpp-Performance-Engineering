#include "kmeans_serial.hpp"

#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

void test_seed_rows() {
    require(kmeans::initial_row_indices(16, 2) ==
                std::vector<std::size_t>({4, 12}), "K=2 seed rows");
    require(kmeans::initial_row_indices(16, 3) ==
                std::vector<std::size_t>({2, 8, 13}), "K=3 seed rows");
    const auto large = kmeans::initial_row_indices(1u << 20, 32);
    require(large.front() == 16384 && large.back() == 1032192,
            "maximum-N seed rows");
}

void test_ownership_and_repeatability() {
    const std::vector<float> input = {0, 0, 0, 0, 0, 0, 0, 0,
                                      10, 10, 10, 10, 10, 10, 10, 10};
    const auto before = input;
    auto first = kmeans::kmeans_serial(input, 16, 1, 2);
    const auto second = kmeans::kmeans_serial(input, 16, 1, 2);
    require(std::memcmp(input.data(), before.data(), input.size() * sizeof(float)) == 0,
            "input changed");
    require(first.labels.size() == 16 && first.centroids.size() == 2,
            "output sizes");
    require(first.labels == second.labels && first.centroids == second.centroids &&
                first.update_count == second.update_count &&
                first.converged == second.converged, "repeatability");
    require(first.update_count == 1 && first.converged &&
                first.centroids == std::vector<float>({0, 10}),
            "one-pass convergence");
    require(first.centroids.data() != input.data(), "centroid ownership");
    first.centroids[0] = 99;
    first.labels[0] = 1;
    require(input[0] == 0 && second.centroids[0] == 0 && second.labels[0] == 0,
            "outputs are not independent");
}

void test_ties_empty_and_cap() {
    const std::vector<float> tied = {0, 3, 3, 3, 3, 3, 6, 3,
                                      3, 9, 9, 9, 9, 9, 9, 9};
    const auto tie_result = kmeans::kmeans_serial(tied, 16, 1, 3);
    require(tie_result.labels[6] == 0 && tie_result.centroids[1] == 3,
            "lowest-index tie or empty-centroid retention");
    for (const auto label : tie_result.labels) {
        require(label != 1, "duplicate seed cluster must remain empty");
    }
    const std::vector<float> moving = {0, 1, 2, 3, 4, 5, 6, 7,
                                        8, 9, 10, 11, 12, 30, 31, 32};
    const auto capped = kmeans::kmeans_serial_with_update_cap(moving, 16, 1, 2, 1);
    const auto complete = kmeans::kmeans_serial(moving, 16, 1, 2);
    require(capped.update_count == 1 && !capped.converged &&
                capped.labels[12] == 1 && capped.centroids[0] == 4,
            "reduced-cap final reassignment");
    require(complete.update_count == 3 && complete.converged,
            "multi-update convergence");
}

void test_fp32_assignment() {
    const float small = 0.0001220703125f;
    std::vector<float> input;
    for (int i = 0; i < 7; ++i) {
        input.insert(input.end(), {0.0f, small});
    }
    input.insert(input.end(), {0.5f, 0.0f});
    for (int i = 0; i < 8; ++i) {
        input.insert(input.end(), {1.0f, 0.0f});
    }
    const auto result = kmeans::kmeans_serial(input, 16, 2, 2);
    require(result.labels[7] == 0 && result.update_count == 1 && result.converged,
            "FP32-sensitive assignment");
}

void test_invalid_inputs() {
    const std::vector<float> good(32, 0.0f);
    require_rejected([&] { kmeans::kmeans_serial(good, 16, 2, 1); }, "K=1");
    require_rejected([&] { kmeans::kmeans_serial(good, 16, 2, 17); }, "K>N");
    require_rejected([&] { kmeans::kmeans_serial(good, 16, 0, 2); }, "D=0");
    require_rejected([&] { kmeans::kmeans_serial(good, 16, 33, 2); }, "D>32");
    require_rejected([&] { kmeans::kmeans_serial(good, (1u << 20) + 1u, 2, 2); },
                     "N>2^20");
    require_rejected([&] { kmeans::kmeans_serial(good, 16, 1, 2); }, "input size");
    require_rejected([&] { kmeans::kmeans_serial_with_update_cap(good, 16, 2, 2, 0); },
                     "zero cap");
    require_rejected([&] { kmeans::kmeans_serial_with_update_cap(good, 16, 2, 2, 101); },
                     "cap over 100");
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(), 1024.25f, -1024.25f}) {
        auto input = good;
        input[0] = bad;
        require_rejected([&] { kmeans::kmeans_serial(input, 16, 2, 2); },
                         "nonfinite or out-of-range value");
    }
}

}  // namespace

int main() {
    try {
        test_seed_rows();
        test_ownership_and_repeatability();
        test_ties_empty_and_cap();
        test_fp32_assignment();
        test_invalid_inputs();
        std::cout << "Project 2 native serial contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 native serial contract test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
