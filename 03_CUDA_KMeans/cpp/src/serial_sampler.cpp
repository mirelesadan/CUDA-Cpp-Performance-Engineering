// Windows-only, profiling-only user-mode instruction-pointer sampler.
// It buffers PCs while the unchanged serial fit runs; PDB lookup happens later.
#ifndef _WIN32
#error This profiling helper is Windows-only.
#endif
#ifndef _M_X64
#error This profiling helper currently captures x64 instruction pointers only.
#endif

#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <mmsystem.h>

#include "kmeans_serial.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::size_t parse_size(const char* text, const char* name) {
    const std::string value(text);
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be nonnegative");
    }
    std::size_t used = 0;
    const auto number = std::stoull(value, &used);
    if (used != value.size() || number > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " is not a valid size");
    }
    return static_cast<std::size_t>(number);
}

std::vector<float> read_input(const char* path, std::size_t n, std::size_t d) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open raw float32 input");
    }
    const auto bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::uintmax_t>(bytes) != n * d * sizeof(float)) {
        throw std::runtime_error("input file size differs from N*D float32 values");
    }
    stream.seekg(0);
    std::vector<float> input(n * d);
    stream.read(reinterpret_cast<char*>(input.data()), bytes);
    if (!stream) {
        throw std::runtime_error("could not read complete input");
    }
    return input;
}

std::vector<std::pair<std::string, std::size_t>> sorted_counts(
    const std::unordered_map<std::string, std::size_t>& counts) {
    std::vector<std::pair<std::string, std::size_t>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    return sorted;
}

void resolve_samples(const std::vector<DWORD64>& addresses) {
    const HANDLE process = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    if (!SymInitialize(process, nullptr, TRUE)) {
        throw std::runtime_error("DbgHelp SymInitialize failed");
    }
    std::unordered_map<std::string, std::size_t> by_function;
    std::unordered_map<std::string, std::size_t> by_line;
    std::size_t source_lines = 0;
    alignas(SYMBOL_INFO) char symbol_storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_storage);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    for (const auto address : addresses) {
        DWORD64 symbol_displacement = 0;
        std::string function = "<unresolved>";
        if (SymFromAddr(process, address, &symbol_displacement, symbol)) {
            function = symbol->Name;
        }
        ++by_function[function];
        IMAGEHLP_LINE64 line = {};
        line.SizeOfStruct = sizeof(line);
        DWORD line_displacement = 0;
        if (SymGetLineFromAddr64(process, address, &line_displacement, &line) &&
            line.FileName != nullptr) {
            ++source_lines;
            ++by_line[std::string(line.FileName) + ":" + std::to_string(line.LineNumber)];
        } else {
            ++by_line["<unresolved line>"];
        }
    }
    SymCleanup(process);
    std::cout << "main_thread_samples=" << addresses.size()
              << " source_line_resolved=" << source_lines << '\n';
    std::cout << "TOP FUNCTIONS (sample count, percent of all samples, name)\n";
    const auto functions = sorted_counts(by_function);
    for (std::size_t i = 0; i < functions.size() && i < 30; ++i) {
        std::cout << functions[i].second << ' ' << std::fixed << std::setprecision(2)
                  << (100.0 * functions[i].second / addresses.size()) << "% "
                  << functions[i].first << '\n';
    }
    std::cout << "TOP SOURCE LINES (sample count, percent of all samples, file:line)\n";
    const auto lines = sorted_counts(by_line);
    for (std::size_t i = 0; i < lines.size() && i < 80; ++i) {
        std::cout << lines[i].second << ' ' << std::fixed << std::setprecision(2)
                  << (100.0 * lines[i].second / addresses.size()) << "% "
                  << lines[i].first << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 8) {
            throw std::invalid_argument(
                "usage: phase2_kmeans_serial_sampler input.f32 N D K repeats "
                "[--variant baseline|addressed]");
        }
        const auto n = parse_size(argv[2], "N");
        const auto d = parse_size(argv[3], "D");
        const auto k = parse_size(argv[4], "K");
        const auto repeats = parse_size(argv[5], "repeats");
        if (n < 2 || n > (1u << 20) || d < 1 || d > 32 ||
            k < 2 || k > 32 || k > n || repeats < 1 || repeats > 10000) {
            throw std::invalid_argument("dimensions or repeat count outside profiling limits");
        }
        std::string variant = "baseline";
        if (argc == 8) {
            if (std::string(argv[6]) != "--variant") {
                throw std::invalid_argument("expected --variant before implementation name");
            }
            variant = argv[7];
            if (variant != "baseline" && variant != "addressed") {
                throw std::invalid_argument("variant must be baseline or addressed");
            }
        }
        const auto input = read_input(argv[1], n, d);
        const auto fit = [&] {
            return variant == "addressed"
                ? kmeans::kmeans_serial_addressed(input, n, d, k)
                : kmeans::kmeans_serial(input, n, d, k);
        };
        auto result = fit();  // Untimed warm-up.
        std::vector<DWORD64> addresses;
        addresses.reserve(50000);

        HANDLE worker = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
                             GetCurrentProcess(), &worker, 0, FALSE,
                             DUPLICATE_SAME_ACCESS)) {
            throw std::runtime_error("could not duplicate main worker thread handle");
        }
        const MMRESULT timer_result = timeBeginPeriod(1);
        if (timer_result != TIMERR_NOERROR) {
            CloseHandle(worker);
            throw std::runtime_error("could not request 1 ms Windows timer resolution");
        }
        LARGE_INTEGER counter_frequency = {};
        if (!QueryPerformanceFrequency(&counter_frequency)) {
            timeEndPeriod(1);
            CloseHandle(worker);
            throw std::runtime_error("high-resolution performance counter unavailable");
        }
        std::atomic<bool> active{true};
        std::atomic<DWORD> sampler_error{ERROR_SUCCESS};
        LARGE_INTEGER first_sample = {};
        LARGE_INTEGER last_sample = {};
        std::exception_ptr sampler_exception;
        std::thread sampler;
        const auto cleanup = [&] {
            active.store(false, std::memory_order_release);
            if (sampler.joinable()) {
                sampler.join();
            }
            timeEndPeriod(1);
            CloseHandle(worker);
        };
        try {
            sampler = std::thread([&] {
                try {
                    while (active.load(std::memory_order_acquire)) {
                        Sleep(1);
                        if (!active.load(std::memory_order_acquire)) {
                            break;
                        }
                        if (SuspendThread(worker) == static_cast<DWORD>(-1)) {
                            sampler_error.store(GetLastError());
                            break;
                        }
                        CONTEXT context = {};
                        context.ContextFlags = CONTEXT_CONTROL;
                        const BOOL captured = GetThreadContext(worker, &context);
                        const DWORD capture_error = captured ? ERROR_SUCCESS : GetLastError();
                        if (ResumeThread(worker) == static_cast<DWORD>(-1)) {
                            sampler_error.store(GetLastError());
                            break;
                        }
                        if (!captured) {
                            sampler_error.store(capture_error);
                            break;
                        }
                        addresses.push_back(context.Rip);
                        if (addresses.size() == 1) {
                            QueryPerformanceCounter(&first_sample);
                        } else {
                            QueryPerformanceCounter(&last_sample);
                        }
                    }
                } catch (...) {
                    sampler_exception = std::current_exception();
                }
            });
            for (std::size_t run = 0; run < repeats; ++run) {
                result = fit();
            }
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        if (sampler_exception) {
            std::rethrow_exception(sampler_exception);
        }
        if (sampler_error.load() != ERROR_SUCCESS || addresses.empty()) {
            throw std::runtime_error("sampler could not collect main-thread PCs");
        }
        std::cout << "variant=" << variant << " repeats=" << repeats
                  << " update_count=" << result.update_count
                  << " converged=" << (result.converged ? "true" : "false")
                  << " timer_period_request_ms=1 sleep_request_ms=1";
        if (addresses.size() > 1) {
            const double observed_interval_ms =
                1000.0 * static_cast<double>(last_sample.QuadPart - first_sample.QuadPart) /
                static_cast<double>(counter_frequency.QuadPart) /
                static_cast<double>(addresses.size() - 1);
            std::cout << " observed_mean_interval_ms=" << std::fixed
                      << std::setprecision(3) << observed_interval_ms;
        }
        std::cout << '\n';
        resolve_samples(addresses);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Project 2 sampler error: " << error.what() << '\n';
        return 1;
    }
}
