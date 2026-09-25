#include "kmeans_serial.hpp"

#include <omp.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::size_t parse_size(const char* text, const char* name) {
    const std::string value(text);
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be nonnegative");
    }
    std::size_t used = 0;
    const auto parsed = std::stoull(value, &used);
    if (used != value.size() || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return static_cast<std::size_t>(parsed);
}

std::vector<int> parse_counts(const std::string& text) {
    std::vector<int> counts;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        const auto part = text.substr(begin, end == std::string::npos
                                                ? end : end - begin);
        const auto count = parse_size(part.c_str(), "thread count");
        if (count == 0 || count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("thread counts must be positive int values");
        }
        counts.push_back(static_cast<int>(count));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (counts.empty() || text.back() == ',') {
        throw std::invalid_argument("thread counts cannot be empty");
    }
    return counts;
}

std::vector<float> read_input(const char* path, std::size_t n, std::size_t d) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open input");
    }
    const auto bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::uintmax_t>(bytes) != n * d * sizeof(float)) {
        throw std::runtime_error("input file size does not equal N*D float32 values");
    }
    stream.seekg(0);
    std::vector<float> input(n * d);
    stream.read(reinterpret_cast<char*>(input.data()), bytes);
    if (!stream) {
        throw std::runtime_error("could not read complete input");
    }
    return input;
}

template <typename T>
void write_binary(const char* path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open output");
    }
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) {
        throw std::runtime_error("could not write complete output");
    }
}

bool same_bits(const kmeans::Result& left, const kmeans::Result& right) {
    return left.labels == right.labels &&
           left.centroids.size() == right.centroids.size() &&
           std::memcmp(left.centroids.data(), right.centroids.data(),
                       left.centroids.size() * sizeof(float)) == 0 &&
           left.update_count == right.update_count &&
           left.converged == right.converged;
}

void print_times(const std::vector<std::vector<double>>& times) {
    std::cout << '[';
    for (std::size_t variant = 0; variant < times.size(); ++variant) {
        if (variant) std::cout << ',';
        std::cout << '[';
        for (std::size_t run = 0; run < times[variant].size(); ++run) {
            if (run) std::cout << ',';
            std::cout << std::fixed << std::setprecision(6) << times[variant][run];
        }
        std::cout << ']';
    }
    std::cout << ']';
}

#ifdef KMEANS_PHASE_TIMING
void print_phases(const std::vector<std::vector<kmeans::PhaseTimings>>& phases) {
    std::cout << '[';
    for (std::size_t variant = 0; variant < phases.size(); ++variant) {
        if (variant) std::cout << ',';
        std::cout << '[';
        for (std::size_t run = 0; run < phases[variant].size(); ++run) {
            if (run) std::cout << ',';
            const auto& phase = phases[variant][run];
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
    }
    std::cout << ']';
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 9 || (argc - 7) % 2 != 0) {
            throw std::invalid_argument(
                "usage: phase2_kmeans_openmp_scaling input.f32 N D K labels.i32 "
                "centroids.f32 --counts 1,2,... [--runs 0..100] "
                "[--test-update-cap 1..100]");
        }
        const auto n = parse_size(argv[2], "N");
        const auto d = parse_size(argv[3], "D");
        const auto k = parse_size(argv[4], "K");
        if (n < 2 || n > (1u << 20) || d < 1 || d > 32 ||
            k < 2 || k > 32 || k > n) {
            throw std::invalid_argument("invalid N, D, or K for Project 2 contract");
        }
        std::vector<int> counts;
        std::size_t runs = 0;
        std::size_t update_cap = 100;
        for (int arg = 7; arg < argc; arg += 2) {
            const std::string option(argv[arg]);
            if (option == "--counts") {
                counts = parse_counts(argv[arg + 1]);
            } else if (option == "--runs") {
                runs = parse_size(argv[arg + 1], "runs");
                if (runs > 100) throw std::invalid_argument("runs must be 0..100");
            } else if (option == "--test-update-cap") {
                update_cap = parse_size(argv[arg + 1], "test update cap");
                if (update_cap < 1 || update_cap > 100) {
                    throw std::invalid_argument("test update cap must be 1..100");
                }
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
        if (counts.empty()) throw std::invalid_argument("--counts is required");
        const int runtime_max_threads = omp_get_max_threads();
        const int runtime_processors = omp_get_num_procs();

        // All input I/O and all output comparisons are outside the fit timer.
        const auto input = read_input(argv[1], n, d);
        const auto fit = [&](std::size_t variant) {
            if (variant == 0) {
                return kmeans::kmeans_serial_addressed_with_update_cap(
                    input, n, d, k, update_cap);
            }
            return kmeans::kmeans_openmp_with_update_cap(
                input, n, d, k, update_cap, counts[variant - 1]);
        };
        auto reference = fit(0);
        for (std::size_t variant = 1; variant <= counts.size(); ++variant) {
            const auto result = fit(variant);  // One untimed warm-up per path.
            if (!same_bits(reference, result)) {
                throw std::runtime_error("serial/OpenMP warm-up outputs differ bitwise");
            }
        }

        std::vector<std::vector<double>> times(counts.size() + 1);
#ifdef KMEANS_PHASE_TIMING
        std::vector<std::vector<kmeans::PhaseTimings>> phases(counts.size() + 1);
#endif
        for (std::size_t round = 0; round < runs; ++round) {
            // Rotate order across rounds so each configuration samples the
            // same bounded session rather than one isolated time window.
            for (std::size_t offset = 0; offset <= counts.size(); ++offset) {
                const auto variant = (round + offset) % (counts.size() + 1);
                const auto start = std::chrono::steady_clock::now();
                auto result = fit(variant);
                const auto end = std::chrono::steady_clock::now();
                times[variant].push_back(
                    std::chrono::duration<double, std::milli>(end - start).count());
#ifdef KMEANS_PHASE_TIMING
                phases[variant].push_back(kmeans::last_phase_timings());
#endif
                if (!same_bits(reference, result)) {
                    throw std::runtime_error("serial/OpenMP timed outputs differ bitwise");
                }
            }
        }
        write_binary(argv[5], reference.labels);
        write_binary(argv[6], reference.centroids);
        std::cout << "{\"n\":" << n << ",\"d\":" << d << ",\"k\":" << k
                  << ",\"update_count\":" << reference.update_count
                  << ",\"converged\":" << (reference.converged ? "true" : "false")
                  << ",\"omp_max_threads\":" << runtime_max_threads
                  << ",\"omp_num_procs\":" << runtime_processors
                  << ",\"warmup_each\":1,\"timed_runs_each\":" << runs
                  << ",\"counts\":[";
        for (std::size_t index = 0; index < counts.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << counts[index];
        }
        std::cout << "],\"times_ms\":";
        print_times(times);
#ifdef KMEANS_PHASE_TIMING
        std::cout << ",\"phase_timings\":";
        print_phases(phases);
#endif
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 OpenMP scaling driver error: " << error.what() << '\n';
        return 1;
    }
}
