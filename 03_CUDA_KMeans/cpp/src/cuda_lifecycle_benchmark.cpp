#include "kmeans_cuda_lifecycle.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::size_t size_arg(const char* text) {
    const std::string value(text);
    if (value.empty() || value[0] == '-') throw std::invalid_argument("negative size");
    std::size_t used = 0;
    const auto result = std::stoull(value, &used);
    if (used != value.size() || result > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("invalid size");
    return static_cast<std::size_t>(result);
}
template <typename T>
void save(const char* path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size()*sizeof(T)));
    if (!out) throw std::runtime_error("cannot write output");
}
void exact(const kmeans::Result& a, const kmeans::Result& b) {
    if (a.labels != b.labels || a.update_count != b.update_count ||
        a.converged != b.converged || a.centroids.size() != b.centroids.size() ||
        std::memcmp(a.centroids.data(), b.centroids.data(),
                    a.centroids.size()*sizeof(float)))
        throw std::runtime_error("OpenMP/retained CUDA benchmark result changed");
}
void print_stages(const std::array<double, kmeans::benchmark::stage_count>& values) {
    std::cout << '{';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '"' << kmeans::benchmark::stage_names[i] << "\":" << values[i];
    }
    std::cout << '}';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 10)
            throw std::invalid_argument(
                "usage: lifecycle input.f32 N D K labels.i32 centroids.f32 runs sequences cap");
        const auto n = size_arg(argv[2]), d = size_arg(argv[3]), k = size_arg(argv[4]);
        const auto runs = size_arg(argv[7]), sequences = size_arg(argv[8]);
        const auto cap = size_arg(argv[9]);
        if (n < 2 || n > (1u << 20) || d < 1 || d > 32 ||
            k < 2 || k > 32 || k > n)
            throw std::invalid_argument("invalid N/D/K");
        std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
        if (!in || in.tellg() != static_cast<std::streamoff>(n*d*sizeof(float)))
            throw std::runtime_error("input byte size does not equal N*D*4");
        in.seekg(0);
        std::vector<float> input(n*d);
        in.read(reinterpret_cast<char*>(input.data()), n*d*sizeof(float));
        if (!in) throw std::runtime_error("cannot read input");

        const auto report = kmeans::benchmark::characterize_cuda_lifecycle(
            input, n, d, k, cap, runs, sequences);
        // Same-session native CPU context; fresh allocations and validation are
        // included. Oracle checks and raw-file I/O are outside all timers.
        std::vector<double> openmp;
        if (runs) {
            exact(report.result, kmeans::kmeans_openmp_with_update_cap(
                input, n, d, k, cap, 8));
            for (std::size_t i = 0; i < runs; ++i) {
                const auto start = std::chrono::steady_clock::now();
                auto result = kmeans::kmeans_openmp_with_update_cap(
                    input, n, d, k, cap, 8);
                const auto stop = std::chrono::steady_clock::now();
                openmp.push_back(std::chrono::duration<double, std::milli>(
                    stop - start).count());
                exact(report.result, result);
            }
        }
        save(argv[5], report.result.labels);
        save(argv[6], report.result.centroids);
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"n\":" << n << ",\"d\":" << d << ",\"k\":" << k
                  << ",\"update_count\":" << report.result.update_count
                  << ",\"converged\":" << (report.result.converged ? "true" : "false")
                  << ",\"checked_fits\":" << report.checked_fits
                  << ",\"device_bytes\":{";
        for (std::size_t i = 0; i < report.device_bytes.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << kmeans::benchmark::buffer_names[i]
                      << "\":" << report.device_bytes[i];
        }
        std::cout << "},\"diagnostic_event_setup_ms\":"
                  << report.diagnostic_event_setup_ms
                  << ",\"diagnostic_event_destroy_ms\":"
                  << report.diagnostic_event_destroy_ms << ",\"openmp8_ms\":[";
        for (std::size_t i = 0; i < openmp.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << openmp[i];
        }
        std::cout << "],\"samples\":[";
        for (std::size_t i = 0; i < report.samples.size(); ++i) {
            if (i) std::cout << ',';
            const auto& s = report.samples[i];
            std::cout << "{\"mode\":\"" << s.mode
                      << "\",\"instrumentation\":\"" << s.instrumentation
                      << "\",\"fits\":" << s.fits << ",\"wall_ms\":" << s.wall_ms
                      << ",\"fit_wall_ms\":" << s.fit_wall_ms
                      << ",\"event_collection_ms\":" << s.event_collection_ms
                      << ",\"host_ms\":";
            print_stages(s.host_ms);
            std::cout << ",\"event_ms\":";
            print_stages(s.event_ms);
            std::cout << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Lifecycle characterization failed: " << error.what() << '\n';
        return 1;
    }
}
