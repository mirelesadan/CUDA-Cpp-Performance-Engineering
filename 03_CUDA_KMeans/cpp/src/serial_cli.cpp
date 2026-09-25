#include "kmeans_serial.hpp"

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

bool same_result_bits(const kmeans::Result& baseline,
                      const kmeans::Result& candidate) {
    return baseline.labels == candidate.labels &&
           baseline.centroids.size() == candidate.centroids.size() &&
           std::memcmp(baseline.centroids.data(), candidate.centroids.data(),
                       baseline.centroids.size() * sizeof(float)) == 0 &&
           baseline.update_count == candidate.update_count &&
           baseline.converged == candidate.converged;
}

void print_timing_blocks(const std::vector<std::vector<double>>& blocks) {
    std::cout << '[';
    for (std::size_t block = 0; block < blocks.size(); ++block) {
        if (block != 0) {
            std::cout << ',';
        }
        std::cout << '[';
        for (std::size_t run = 0; run < blocks[block].size(); ++run) {
            if (run != 0) {
                std::cout << ',';
            }
            std::cout << std::fixed << std::setprecision(6) << blocks[block][run];
        }
        std::cout << ']';
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(float) == 4 && sizeof(std::int32_t) == 4,
                  "binary validation bridge requires 32-bit float and int");
    try {
        if (argc < 7 || (argc - 7) % 2 != 0) {
            throw std::invalid_argument(
                "usage: phase2_kmeans_serial input.f32 N D K labels.i32 centroids.f32 "
                "[--test-update-cap 1..100] [--timed-runs 7..100] "
                "[--variant baseline|addressed] "
                "[--paired-runs 5..100] [--paired-blocks 1..10]");
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
        std::size_t paired_runs = 0;
        std::size_t paired_blocks = 1;
        std::string variant = "baseline";
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
            } else if (option == "--variant") {
                variant = argv[argument + 1];
                if (variant != "baseline" && variant != "addressed") {
                    throw std::invalid_argument("variant must be baseline or addressed");
                }
            } else if (option == "--paired-runs") {
                paired_runs = parse_size(argv[argument + 1], "paired runs");
                if (paired_runs < 5 || paired_runs > 100) {
                    throw std::invalid_argument("paired runs must be 5..100");
                }
            } else if (option == "--paired-blocks") {
                paired_blocks = parse_size(argv[argument + 1], "paired blocks");
                if (paired_blocks < 1 || paired_blocks > 10) {
                    throw std::invalid_argument("paired blocks must be 1..10");
                }
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
        if (paired_runs && (timed_runs || variant != "baseline" ||
                            reduced_cap_requested)) {
            throw std::invalid_argument("paired runs cannot combine with other fit options");
        }
        if (!paired_runs && paired_blocks != 1) {
            throw std::invalid_argument("paired blocks require paired runs");
        }
#ifdef KMEANS_PHASE_TIMING
        if (paired_runs || variant != "baseline") {
            throw std::invalid_argument("phase-timing target supports baseline only");
        }
#endif

        // Raw little-endian float32 exchange is a test/benchmark bridge, not
        // a native data format or Python binding. File I/O precedes timing.
        const auto input = read_input(argv[1], n, d);
        if (paired_runs) {
            const auto baseline_fit = [&] { return kmeans::kmeans_serial(input, n, d, k); };
            const auto candidate_fit = [&] {
                return kmeans::kmeans_serial_addressed(input, n, d, k);
            };
            auto baseline_result = baseline_fit();  // Warm each implementation once.
            auto candidate_result = candidate_fit();
            if (!same_result_bits(baseline_result, candidate_result)) {
                throw std::runtime_error("baseline/candidate warm-up outputs differ bitwise");
            }
            std::vector<std::vector<double>> baseline_times(paired_blocks);
            std::vector<std::vector<double>> candidate_times(paired_blocks);
            const auto time_fit = [&](const auto& fit, kmeans::Result& output) {
                const auto start = std::chrono::steady_clock::now();
                auto next = fit();
                const auto end = std::chrono::steady_clock::now();
                output = std::move(next);  // Outside the measured fit call.
                return std::chrono::duration<double, std::milli>(end - start).count();
            };
            for (std::size_t block = 0; block < paired_blocks; ++block) {
                baseline_times[block].reserve(paired_runs);
                candidate_times[block].reserve(paired_runs);
                for (std::size_t run = 0; run < paired_runs; ++run) {
                    const bool candidate_first = (block + run) % 2 != 0;
                    double baseline_ms = 0.0;
                    double candidate_ms = 0.0;
                    if (candidate_first) {
                        candidate_ms = time_fit(candidate_fit, candidate_result);
                        baseline_ms = time_fit(baseline_fit, baseline_result);
                    } else {
                        baseline_ms = time_fit(baseline_fit, baseline_result);
                        candidate_ms = time_fit(candidate_fit, candidate_result);
                    }
                    if (!same_result_bits(baseline_result, candidate_result)) {
                        throw std::runtime_error("baseline/candidate timed outputs differ bitwise");
                    }
                    baseline_times[block].push_back(baseline_ms);
                    candidate_times[block].push_back(candidate_ms);
                }
            }
            write_binary(argv[5], candidate_result.labels);
            write_binary(argv[6], candidate_result.centroids);
            std::cout << "{\"update_count\":" << candidate_result.update_count
                      << ",\"converged\":" << (candidate_result.converged ? "true" : "false")
                      << ",\"warmup_each\":1,\"paired_runs_per_block\":"
                      << paired_runs << ",\"first_variant_per_block\":[";
            for (std::size_t block = 0; block < paired_blocks; ++block) {
                if (block != 0) {
                    std::cout << ',';
                }
                std::cout << (block % 2 == 0 ? "\"baseline\"" : "\"addressed\"");
            }
            std::cout << "],\"baseline_times_ms\":";
            print_timing_blocks(baseline_times);
            std::cout << ",\"addressed_times_ms\":";
            print_timing_blocks(candidate_times);
            std::cout << "}\n";
            return 0;
        }
        const auto fit = [&]() {
            if (variant == "addressed") {
                return reduced_cap_requested
                    ? kmeans::kmeans_serial_addressed_with_update_cap(
                          input, n, d, k, update_cap)
                    : kmeans::kmeans_serial_addressed(input, n, d, k);
            }
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
