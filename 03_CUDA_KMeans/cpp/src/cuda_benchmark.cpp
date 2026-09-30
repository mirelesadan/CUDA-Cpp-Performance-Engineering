#include "kmeans_cuda.hpp"
#include "kmeans_serial.hpp"

#include <omp.h>

#include <chrono>
#include <algorithm>
#include <cmath>
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

std::vector<float> read_input(const char* path, std::size_t n, std::size_t d) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open input");
    const auto bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::uintmax_t>(bytes) != n * d * sizeof(float)) {
        throw std::runtime_error("input size does not equal N*D float32 values");
    }
    stream.seekg(0);
    std::vector<float> input(n * d);
    stream.read(reinterpret_cast<char*>(input.data()), bytes);
    if (!stream) throw std::runtime_error("could not read complete input");
    return input;
}

template <typename T>
void write_binary(const char* path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open output");
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) throw std::runtime_error("could not write complete output");
}

bool same_bits(const kmeans::Result& left, const kmeans::Result& right) {
    return left.labels == right.labels &&
           left.centroids.size() == right.centroids.size() &&
           std::memcmp(left.centroids.data(), right.centroids.data(),
                       left.centroids.size() * sizeof(float)) == 0 &&
           left.update_count == right.update_count &&
           left.converged == right.converged;
}

double inertia(const std::vector<float>& input, const kmeans::Result& result,
               std::size_t n, std::size_t d) {
    double total = 0.0;
    for (std::size_t sample = 0; sample < n; ++sample) {
        const std::size_t cluster =
            static_cast<std::size_t>(result.labels[sample]);
        for (std::size_t feature = 0; feature < d; ++feature) {
            const double difference =
                static_cast<double>(input[sample * d + feature]) -
                static_cast<double>(result.centroids[cluster * d + feature]);
            total += difference * difference;
        }
    }
    return total;
}

bool same_frozen_contract(const kmeans::Result& reference,
                          const kmeans::Result& candidate,
                          const std::vector<float>& input,
                          std::size_t n, std::size_t d, std::size_t k) {
    if (candidate.labels != reference.labels ||
        candidate.update_count != reference.update_count ||
        candidate.converged != reference.converged ||
        candidate.centroids.size() != k * d) return false;
    for (std::size_t feature = 0; feature < d; ++feature) {
        double scale = 1.0;
        for (std::size_t sample = 0; sample < n; ++sample) {
            scale = std::max(scale,
                std::abs(static_cast<double>(input[sample * d + feature])));
        }
        for (std::size_t cluster = 0; cluster < k; ++cluster) {
            const auto index = cluster * d + feature;
            if (!std::isfinite(candidate.centroids[index]) ||
                std::abs(static_cast<double>(candidate.centroids[index]) -
                         static_cast<double>(reference.centroids[index])) >
                    5e-6 * scale) return false;
        }
    }
    const double reference_inertia = inertia(input, reference, n, d);
    const double candidate_inertia = inertia(input, candidate, n, d);
    return std::abs(candidate_inertia - reference_inertia) <=
               2e-5 * std::max(1.0, std::abs(reference_inertia)) &&
           (reference_inertia != 0.0 || candidate_inertia == 0.0);
}

void print_times(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << std::fixed << std::setprecision(6) << values[i];
    }
    std::cout << ']';
}

void print_stages(const kmeans::CudaStageTimings& t) {
    std::cout << "{\"setup_ms\":" << t.setup_ms
              << ",\"h2d_ms\":" << t.h2d_ms
              << ",\"initialization_ms\":" << t.initialization_ms
              << ",\"initial_assignment_ms\":" << t.initial_assignment_ms
              << ",\"count_ms\":" << t.count_ms
              << ",\"centroid_update_ms\":" << t.centroid_update_ms
              << ",\"reassignment_ms\":" << t.reassignment_ms
              << ",\"convergence_ms\":" << t.convergence_ms
              << ",\"final_d2h_ms\":" << t.final_d2h_ms
              << ",\"device_algorithm_ms\":" << t.device_algorithm_ms
              << ",\"one_shot_gpu_ms\":" << t.one_shot_gpu_ms
              << ",\"native_wall_ms\":" << t.native_wall_ms << '}';
}

}  // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(float) == 4 && sizeof(std::int32_t) == 4,
                  "binary bridge requires 32-bit float and int");
    try {
        if (argc < 7 || (argc - 7) % 2 != 0) {
            throw std::invalid_argument(
                "usage: phase2_kmeans_cuda_benchmark input.f32 N D K labels.i32 "
                "centroids.f32 [--runs 0..50] [--diagnostic-runs 0..20] "
                "[--test-update-cap 1..100] [--skip-serial 0|1] "
                "[--compare-tiled 0|1] [--update-pairs 0|7..50]");
        }
        const auto n = parse_size(argv[2], "N");
        const auto d = parse_size(argv[3], "D");
        const auto k = parse_size(argv[4], "K");
        if (n < 2 || n > (1u << 20) || d < 1 || d > 32 ||
            k < 2 || k > 32 || k > n) {
            throw std::invalid_argument("invalid N, D, or K");
        }
        std::size_t runs = 0;
        std::size_t diagnostic_runs = 0;
        std::size_t update_cap = 100;
        bool skip_serial = false;
        bool compare_tiled = false;
        std::size_t update_pairs = 0;
        for (int arg = 7; arg < argc; arg += 2) {
            const std::string option(argv[arg]);
            const auto value = parse_size(argv[arg + 1], option.c_str());
            if (option == "--runs" && value <= 50) {
                runs = value;
            } else if (option == "--diagnostic-runs" && value <= 20) {
                diagnostic_runs = value;
            } else if (option == "--test-update-cap" &&
                       value >= 1 && value <= 100) {
                update_cap = value;
            } else if (option == "--skip-serial" && value <= 1) {
                skip_serial = value != 0;
            } else if (option == "--compare-tiled" && value <= 1) {
                compare_tiled = value != 0;
            } else if (option == "--update-pairs" &&
                       (value == 0 || (value >= 7 && value <= 50))) {
                update_pairs = value;
            } else {
                throw std::invalid_argument("unknown option or invalid value: " + option);
            }
        }
        if (update_pairs != 0 && !compare_tiled) {
            throw std::invalid_argument("--update-pairs requires --compare-tiled 1");
        }

        // Raw-file I/O and correctness comparisons are outside the fit timer.
        // Every normal CUDA fit owns device allocation, transfer and teardown.
        const auto input = read_input(argv[1], n, d);
        const auto input_before = input;
        const auto serial_fit = [&] {
            return update_cap == 100
                ? kmeans::kmeans_serial_addressed(input, n, d, k)
                : kmeans::kmeans_serial_addressed_with_update_cap(
                      input, n, d, k, update_cap);
        };
        const auto openmp_fit = [&] {
            return update_cap == 100
                ? kmeans::kmeans_openmp(input, n, d, k, 8)
                : kmeans::kmeans_openmp_with_update_cap(
                      input, n, d, k, update_cap, 8);
        };
        const auto cuda_fit = [&] {
            return update_cap == 100
                ? kmeans::kmeans_cuda(input, n, d, k)
                : kmeans::kmeans_cuda_with_update_cap(
                      input, n, d, k, update_cap);
        };
        const auto tiled_fit = [&] {
            return update_cap == 100
                ? kmeans::kmeans_cuda_tiled(input, n, d, k)
                : kmeans::kmeans_cuda_tiled_with_update_cap(
                      input, n, d, k, update_cap);
        };

        const auto reference = serial_fit();
        auto serial_result = reference;
        auto openmp_result = openmp_fit();
        auto cuda_result = cuda_fit();
        if (!same_bits(reference, openmp_result) ||
            !same_bits(reference, cuda_result)) {
            throw std::runtime_error("serial/OpenMP-8/CUDA outputs differ bitwise");
        }
        kmeans::Result tiled_reference;
        kmeans::Result tiled_result;
        if (compare_tiled) {
            tiled_reference = tiled_fit();
            tiled_result = tiled_reference;
            if (!same_frozen_contract(reference, tiled_reference,
                                      input, n, d, k)) {
                throw std::runtime_error("tiled CUDA violates frozen contract");
            }
        }
        const std::size_t cuda_warmups = runs ? 3 : 1;
        for (std::size_t i = 1; i < cuda_warmups; ++i) {
            cuda_result = cuda_fit();
            if (!same_bits(reference, cuda_result)) {
                throw std::runtime_error("CUDA warm-up output differs bitwise");
            }
            if (compare_tiled) {
                tiled_result = tiled_fit();
                if (!same_bits(tiled_reference, tiled_result)) {
                    throw std::runtime_error("tiled CUDA warm-up not deterministic");
                }
            }
        }

        std::vector<double> serial_times, openmp_times, cuda_times, tiled_times;
        const auto time_fit = [&](const auto& fit, kmeans::Result& result,
                                  bool tiled) {
            const auto start = std::chrono::steady_clock::now();
            auto next = fit();
            const auto end = std::chrono::steady_clock::now();
            result = std::move(next);
            if (tiled) {
                // The full contract was checked before timing. Keep the
                // post-call check symmetric with the control variants.
                if (!same_bits(tiled_reference, result)) {
                    throw std::runtime_error(
                        "timed tiled output is not deterministic");
                }
            } else if (!same_bits(reference, result)) {
                throw std::runtime_error("timed control differs bitwise from serial");
            }
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        for (std::size_t round = 0; round < runs; ++round) {
            if (compare_tiled) {
                // Pair the two GPU fits adjacently and reverse their order on
                // alternate rounds. CPU controls stay in the same session.
                if (round % 2 == 0) {
                    cuda_times.push_back(time_fit(cuda_fit, cuda_result, false));
                    tiled_times.push_back(time_fit(tiled_fit, tiled_result, true));
                } else {
                    tiled_times.push_back(time_fit(tiled_fit, tiled_result, true));
                    cuda_times.push_back(time_fit(cuda_fit, cuda_result, false));
                }
                if (skip_serial) {
                    openmp_times.push_back(time_fit(openmp_fit, openmp_result, false));
                } else if (round % 2 == 0) {
                    serial_times.push_back(time_fit(serial_fit, serial_result, false));
                    openmp_times.push_back(time_fit(openmp_fit, openmp_result, false));
                } else {
                    openmp_times.push_back(time_fit(openmp_fit, openmp_result, false));
                    serial_times.push_back(time_fit(serial_fit, serial_result, false));
                }
            } else {
                // Preserve the original control benchmark rotation.
                const std::size_t variants = skip_serial ? 2 : 3;
                for (std::size_t offset = 0; offset < variants; ++offset) {
                    const auto variant = (round + offset) % variants;
                    if (skip_serial) {
                        if (variant == 0) {
                            openmp_times.push_back(
                                time_fit(openmp_fit, openmp_result, false));
                        } else {
                            cuda_times.push_back(
                                time_fit(cuda_fit, cuda_result, false));
                        }
                    } else if (variant == 0) {
                        serial_times.push_back(
                            time_fit(serial_fit, serial_result, false));
                    } else if (variant == 1) {
                        openmp_times.push_back(
                            time_fit(openmp_fit, openmp_result, false));
                    } else {
                        cuda_times.push_back(
                            time_fit(cuda_fit, cuda_result, false));
                    }
                }
            }
        }

        // Event-based diagnostic fits are separate from normal wall-clock fits.
        std::vector<kmeans::CudaStageTimings> stage_runs;
        std::vector<kmeans::CudaStageTimings> tiled_stage_runs;
        for (std::size_t i = 0; i < diagnostic_runs; ++i) {
            const auto diagnostic =
                kmeans::kmeans_cuda_diagnostic(input, n, d, k, update_cap);
            if (!same_bits(reference, diagnostic.result)) {
                throw std::runtime_error("CUDA diagnostic output differs bitwise");
            }
            stage_runs.push_back(diagnostic.timings);
            if (compare_tiled) {
                const auto tiled_diagnostic =
                    kmeans::kmeans_cuda_tiled_diagnostic(
                        input, n, d, k, update_cap);
                if (!same_frozen_contract(reference, tiled_diagnostic.result,
                                          input, n, d, k) ||
                    !same_bits(tiled_reference, tiled_diagnostic.result)) {
                    throw std::runtime_error(
                        "tiled diagnostic violates contract or determinism");
                }
                tiled_stage_runs.push_back(tiled_diagnostic.timings);
            }
        }
        kmeans::CudaUpdatePairTimings pair_timings;
        if (update_pairs != 0) {
            pair_timings = kmeans::benchmark_cuda_update_pair(
                input, n, d, k, update_pairs);
        }
        if (input != input_before) throw std::runtime_error("input was modified");
        const auto& output_result = compare_tiled ? tiled_result : cuda_result;
        write_binary(argv[5], output_result.labels);
        write_binary(argv[6], output_result.centroids);
        std::cout << "{\"n\":" << n << ",\"d\":" << d << ",\"k\":" << k
                  << ",\"update_count\":" << output_result.update_count
                  << ",\"converged\":" << (output_result.converged ? "true" : "false")
                  << ",\"omp_max_threads\":" << omp_get_max_threads()
                  << ",\"omp_num_procs\":" << omp_get_num_procs()
                  << ",\"omp_control_threads\":8,\"cuda_warmups\":" << cuda_warmups
                  << ",\"cpu_warmups\":1,\"timed_runs_each\":" << runs
                  << ",\"diagnostic_runs\":" << diagnostic_runs
                  << ",\"compare_tiled\":"
                  << (compare_tiled ? "true" : "false")
                  << ",\"serial_timing_skipped\":"
                  << (skip_serial ? "true" : "false")
                  << ",\"serial_times_ms\":";
        print_times(serial_times);
        std::cout << ",\"openmp8_times_ms\":";
        print_times(openmp_times);
        std::cout << ",\"cuda_wall_times_ms\":";
        print_times(cuda_times);
        std::cout << ",\"tiled_wall_times_ms\":";
        print_times(tiled_times);
        std::cout << ",\"cuda_stage_runs\":[";
        for (std::size_t i = 0; i < stage_runs.size(); ++i) {
            if (i) std::cout << ',';
            print_stages(stage_runs[i]);
        }
        std::cout << "],\"tiled_stage_runs\":[";
        for (std::size_t i = 0; i < tiled_stage_runs.size(); ++i) {
            if (i) std::cout << ',';
            print_stages(tiled_stage_runs[i]);
        }
        std::cout << "],\"update_pair\":{\"control_ms\":";
        print_times(pair_timings.control_ms);
        std::cout << ",\"tiled_ms\":";
        print_times(pair_timings.tiled_ms);
        std::cout << ",\"partial_ms\":";
        print_times(pair_timings.partial_ms);
        std::cout << ",\"finalize_ms\":";
        print_times(pair_timings.finalize_ms);
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 CUDA benchmark driver error: "
                  << error.what() << '\n';
        return 1;
    }
}
