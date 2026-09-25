#include "kmeans_serial.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::size_t parse_size(const char* text, const std::string& name) {
    const std::string value(text);
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(name + " must be a nonnegative integer");
    }
    std::size_t parsed = 0;
    const auto number = std::stoull(value, &parsed);
    if (parsed != value.size() || number > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(name + " must be a valid size");
    }
    return static_cast<std::size_t>(number);
}

std::vector<float> read_input(const std::string& path, std::size_t n, std::size_t d) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open input: " + path);
    }
    const auto byte_count = stream.tellg();
    if (byte_count < 0 || static_cast<std::uintmax_t>(byte_count) !=
                              n * d * sizeof(float)) {
        throw std::runtime_error("input binary size does not match N*D float32 values");
    }
    stream.seekg(0);
    std::vector<float> input(n * d);
    stream.read(reinterpret_cast<char*>(input.data()), byte_count);
    if (!stream) {
        throw std::runtime_error("failed to read complete input binary");
    }
    return input;
}

template <typename Value>
void write_binary(const std::string& path, const std::vector<Value>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open output: " + path);
    }
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(Value)));
    if (!stream) {
        throw std::runtime_error("failed to write complete output binary: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(float) == 4 && sizeof(std::int32_t) == 4,
                  "binary validation bridge requires 32-bit float and int");
    try {
        if (argc < 7 || (argc - 7) % 2 != 0) {
            throw std::invalid_argument(
                "usage: phase2_kmeans_serial input.f32 N D K labels.i32 centroids.f32 "
                "[--test-update-cap 1..100] [--timed-runs 7..100]");
        }
        const std::size_t n = parse_size(argv[2], "N");
        const std::size_t d = parse_size(argv[3], "D");
        const std::size_t k = parse_size(argv[4], "K");
        if (n < 2 || n > (1u << 20) || d < 1 || d > 32 ||
            k < 2 || k > 32 || k > n) {
            throw std::invalid_argument("invalid N, D, or K for Project 2 contract");
        }
        std::size_t update_cap = 100;
        bool reduced_cap_requested = false;
        std::size_t timed_runs = 0;
        for (int argument = 7; argument < argc; argument += 2) {
            const std::string option(argv[argument]);
            if (option == "--test-update-cap") {
                update_cap = parse_size(argv[argument + 1], "test update cap");
                reduced_cap_requested = true;
                if (update_cap < 1 || update_cap > 100) {
                    throw std::invalid_argument("test update cap must be 1..100");
                }
            } else if (option == "--timed-runs") {
                timed_runs = parse_size(argv[argument + 1], "timed runs");
                if (timed_runs < 7 || timed_runs > 100) {
                    throw std::invalid_argument("timed runs must be 7..100");
                }
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }

        // Raw little-endian float32 exchange is a test/benchmark bridge, not
        // a native data format or Python binding. File I/O precedes timing.
        const auto input = read_input(argv[1], n, d);
        const auto fit = [&]() {
            return reduced_cap_requested
                ? kmeans::kmeans_serial_with_update_cap(input, n, d, k, update_cap)
                : kmeans::kmeans_serial(input, n, d, k);
        };

        kmeans::Result result;
        std::vector<double> timings_ms;
#ifdef KMEANS_PHASE_TIMING
        std::vector<kmeans::PhaseTimings> phase_runs;
#endif
        if (timed_runs == 0) {
            result = fit();
#ifdef KMEANS_PHASE_TIMING
            phase_runs.push_back(kmeans::last_phase_timings());
#endif
        } else {
            result = fit();  // One untimed warm-up; allocations remain inside fit.
            timings_ms.reserve(timed_runs);
#ifdef KMEANS_PHASE_TIMING
            phase_runs.reserve(timed_runs);
#endif
            for (std::size_t run = 0; run < timed_runs; ++run) {
                const auto start = std::chrono::steady_clock::now();
                auto timed_result = fit();
                const auto end = std::chrono::steady_clock::now();
                timings_ms.push_back(
                    std::chrono::duration<double, std::milli>(end - start).count());
#ifdef KMEANS_PHASE_TIMING
                phase_runs.push_back(kmeans::last_phase_timings());
#endif
                result = std::move(timed_result);
            }
        }

        write_binary(argv[5], result.labels);
        write_binary(argv[6], result.centroids);
        std::cout << "{\"update_count\":" << result.update_count
                  << ",\"converged\":" << (result.converged ? "true" : "false")
                  << ",\"warmup\":" << (timed_runs ? 1 : 0)
                  << ",\"timings_ms\":[";
        for (std::size_t index = 0; index < timings_ms.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << std::fixed << std::setprecision(6) << timings_ms[index];
        }
        std::cout << ']';
#ifdef KMEANS_PHASE_TIMING
        std::cout << ",\"phase_timings\":[";
        for (std::size_t index = 0; index < phase_runs.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            const auto& phase = phase_runs[index];
            std::cout << "{\"validation_ms\":" << phase.validation_ms
                      << ",\"initialization_ms\":" << phase.initialization_ms
                      << ",\"initial_assignment_ms\":" << phase.initial_assignment_ms
                      << ",\"centroid_update_ms\":" << phase.centroid_update_ms
                      << ",\"reassignment_ms\":" << phase.reassignment_ms
                      << ",\"convergence_check_ms\":" << phase.convergence_check_ms
                      << ",\"assignment_passes\":" << phase.assignment_passes
                      << ",\"centroid_updates\":" << phase.centroid_updates << '}';
        }
        std::cout << ']';
#endif
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 serial driver error: " << error.what() << '\n';
        return 1;
    }
}
