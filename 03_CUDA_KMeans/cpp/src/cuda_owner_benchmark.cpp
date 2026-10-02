#include "kmeans_cuda.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using FitMetadata = decltype(std::declval<kmeans::CudaKMeansBuffer&>().fit());
struct Sample { std::string mode; std::size_t fits; double wall_ms; };

double elapsed(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}
std::size_t size_arg(const char* text) {
    const std::string value(text);
    if (value.empty() || value[0] == '-') throw std::invalid_argument("negative size");
    std::size_t used = 0;
    const auto result = std::stoull(value, &used);
    if (used != value.size() || result > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("invalid size");
    return static_cast<std::size_t>(result);
}
std::vector<float> read(const char* path, std::size_t size) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(size * sizeof(float)))
        throw std::runtime_error("input byte size does not equal N*D*4");
    in.seekg(0);
    std::vector<float> result(size);
    in.read(reinterpret_cast<char*>(result.data()), size * sizeof(float));
    if (!in) throw std::runtime_error("cannot read input");
    return result;
}
template <typename T>
void save(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) throw std::runtime_error("cannot write output");
}
void exact(const kmeans::Result& expected, const kmeans::Result& actual) {
    if (expected.labels != actual.labels ||
        expected.update_count != actual.update_count ||
        expected.converged != actual.converged ||
        expected.centroids.size() != actual.centroids.size() ||
        std::memcmp(expected.centroids.data(), actual.centroids.data(),
                    expected.centroids.size() * sizeof(float)))
        throw std::runtime_error("result differs from retained one-shot CUDA");
}
void metadata_exact(const kmeans::Result& expected, const FitMetadata& actual) {
    if (expected.update_count != actual.update_count ||
        expected.converged != actual.converged)
        throw std::runtime_error("fit metadata differs from retained one-shot CUDA");
}
void unchanged(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)))
        throw std::runtime_error("input modified");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 11)
            throw std::invalid_argument(
                "usage: owner input.f32 replacement.f32 N D K labels.i32 centroids.f32 runs sequences cap");
        const auto n = size_arg(argv[3]), d = size_arg(argv[4]), k = size_arg(argv[5]);
        const auto runs = size_arg(argv[8]), sequences = size_arg(argv[9]);
        const auto cap = size_arg(argv[10]);
        if (cap < 1 || cap > 100) throw std::invalid_argument("invalid update cap");
        if (cap != 100 && (runs || sequences))
            throw std::invalid_argument("timings require the full 100-update contract");
        if (n < 2 || n > (1u << 20) || d < 1 || d > 32 || k < 2 || k > 32 || k > n)
            throw std::invalid_argument("invalid N/D/K");
        if ((runs && (runs < 7 || runs > 30)) ||
            (sequences && (sequences < 7 || sequences > 30)))
            throw std::invalid_argument("runs/sequences must be zero or 7..30");
        const auto input = read(argv[1], n*d);
        const auto replacement = read(argv[2], n*d);
        const auto before = input, replacement_before = replacement;
        const auto reference = kmeans::kmeans_cuda_parallel_count_with_update_cap(input, n, d, k, cap);
        const auto replacement_reference = kmeans::kmeans_cuda_parallel_count_with_update_cap(replacement, n, d, k, cap);
        auto fit = [cap](kmeans::CudaKMeansBuffer& owner) {
            // Only the public nonconvergence fixture uses the reduced-cap hook.
            return cap == 100 ? owner.fit() : owner.fit_with_update_cap(cap);
        };

        kmeans::CudaKMeansBuffer resident(n, d, k);
        resident.upload(input);
        std::size_t checked_fits = 0;
        // All correctness work precedes timing, including repeated independent
        // fits, independent host ownership, and a different resident snapshot.
        for (std::size_t i = 0; i < 20; ++i) {
            metadata_exact(reference, fit(resident));
            auto first = resident.download();
            auto second = resident.download();
            exact(reference, first);
            exact(reference, second);
            if (first.labels.data() == second.labels.data() ||
                first.centroids.data() == second.centroids.data())
                throw std::runtime_error("downloads do not own independent vectors");
            first.labels[0] = -1;
            first.centroids[0] = std::numeric_limits<float>::quiet_NaN();
            exact(reference, second);
            exact(reference, resident.download());
            ++checked_fits;
        }
        resident.upload(replacement);
        metadata_exact(replacement_reference, fit(resident));
        const auto replacement_result = resident.download();
        exact(replacement_reference, replacement_result);
        resident.upload(input);
        metadata_exact(reference, fit(resident));
        const auto validated_result = resident.download();
        exact(reference, validated_result);
        unchanged(before, input);
        unchanged(replacement_before, replacement);

        std::vector<Sample> samples;
        std::vector<double> construction_ms, destruction_ms, openmp8_ms;
        const char* modes[] = {"A_one_shot", "B_construct_upload_fit_download_destroy",
                              "C_resident_fit_download", "D_resident_fit_only",
                              "E_upload_fit_download"};
        auto one_call = [&](std::size_t mode, bool record) {
            kmeans::Result output;
            FitMetadata metadata{};
            const auto start = Clock::now();
            if (mode == 0) {
                output = kmeans::kmeans_cuda_parallel_count(input, n, d, k);
            } else if (mode == 1) {
                // Device construction AND destruction are inside this boundary.
                kmeans::CudaKMeansBuffer owner(n, d, k);
                owner.upload(input);
                metadata = owner.fit();
                output = owner.download();
            } else {
                if (mode == 4) resident.upload(input);
                metadata = resident.fit();
                if (mode != 3) output = resident.download();
            }
            const auto stop = Clock::now();
            if (mode != 0) metadata_exact(reference, metadata);
            // The fit-only correctness download is outside its timer.
            exact(reference, mode == 3 ? resident.download() : output);
            if (record) samples.push_back({modes[mode], 1, elapsed(start, stop)});
        };
        if (runs) {
            for (std::size_t mode = 0; mode < 5; ++mode) one_call(mode, false);
            exact(reference, kmeans::kmeans_openmp(input, n, d, k, 8));
            for (std::size_t run = 0; run < runs; ++run) {
                // Rotate mode order to reduce a fixed-order cadence advantage.
                for (std::size_t offset = 0; offset < 5; ++offset)
                    one_call((run + offset) % 5, true);
                const auto start = Clock::now();
                auto output = kmeans::kmeans_openmp(input, n, d, k, 8);
                const auto stop = Clock::now();
                openmp8_ms.push_back(elapsed(start, stop));
                exact(reference, output);
                // Constructor allocation cost is separate from every repeated
                // boundary; destruction is measured separately, not hidden.
                const auto create_start = Clock::now();
                auto owner = std::make_unique<kmeans::CudaKMeansBuffer>(n, d, k);
                const auto create_stop = Clock::now();
                owner.reset();
                const auto destroy_stop = Clock::now();
                construction_ms.push_back(elapsed(create_start, create_stop));
                destruction_ms.push_back(elapsed(create_stop, destroy_stop));
            }
        }
        auto repeated = [&](std::size_t fits, bool include_io, bool record) {
            kmeans::Result output;
            FitMetadata metadata{};
            const auto start = Clock::now();
            if (include_io) resident.upload(input);
            for (std::size_t fit = 0; fit < fits; ++fit) metadata = resident.fit();
            if (include_io) output = resident.download();
            const auto stop = Clock::now();
            metadata_exact(reference, metadata);
            exact(reference, include_io ? output : resident.download());
            if (record) samples.push_back({include_io ? "R_upload_fits_download" : "R_fit_only",
                                           fits, elapsed(start, stop)});
        };
        if (sequences) {
            for (const std::size_t fits : {1u, 2u, 5u, 10u, 20u}) {
                repeated(fits, false, false);
                repeated(fits, true, false);
                for (std::size_t run = 0; run < sequences; ++run) {
                    repeated(fits, run % 2 != 0, true);
                    repeated(fits, run % 2 == 0, true);
                }
            }
        }
        unchanged(before, input);
        unchanged(replacement_before, replacement);
        save(argv[6], validated_result.labels);
        save(argv[7], validated_result.centroids);
        save(std::string(argv[6]) + ".replacement", replacement_result.labels);
        save(std::string(argv[7]) + ".replacement", replacement_result.centroids);
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"n\":" << n << ",\"d\":" << d << ",\"k\":" << k
                  << ",\"update_count\":" << reference.update_count
                  << ",\"converged\":" << (reference.converged ? "true" : "false")
                  << ",\"replacement_update_count\":" << replacement_result.update_count
                  << ",\"replacement_converged\":" << (replacement_result.converged ? "true" : "false")
                  << ",\"checked_fits\":" << checked_fits
                  << ",\"replacement_checked\":true,\"independent_downloads\":true"
                  << ",\"device_bytes\":" << resident.device_bytes();
        auto print_times = [](const char* name, const std::vector<double>& values) {
            std::cout << ",\"" << name << "\":[";
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << values[i];
            }
            std::cout << ']';
        };
        print_times("construction_ms", construction_ms);
        print_times("destruction_ms", destruction_ms);
        print_times("openmp8_ms", openmp8_ms);
        std::cout << ",\"samples\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (i) std::cout << ',';
            const auto& s = samples[i];
            std::cout << "{\"mode\":\"" << s.mode << "\",\"fits\":" << s.fits
                      << ",\"wall_ms\":" << s.wall_ms << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Resident owner validation/benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
