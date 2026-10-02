// Included only by the opt-in characterization build of kmeans_cuda.cu.
// Kernels and the retained public run() above are deliberately unchanged.
namespace benchmark {
namespace {

enum class Instrumentation { none, host, events };

// Deferred event readout: no new per-stage synchronization. Only diagnostics
// create events, and collection occurs after the existing blocking flag copy.
class Recorder {
public:
    Recorder(Instrumentation mode, std::size_t capacity = 0) : mode_(mode) {
        if (mode_ != Instrumentation::events) return;
        slots_.resize(capacity);
        try {
            for (auto& slot : slots_) {
                check(cudaEventCreate(&slot.first), "lifecycle event create");
                check(cudaEventCreate(&slot.last), "lifecycle event create");
            }
        } catch (...) {
            destroy();
            throw;
        }
    }
    ~Recorder() { destroy(); }
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    template <typename Action>
    void step(LifecycleSample& sample, Stage stage_id, bool gpu, Action action) {
        if (mode_ == Instrumentation::none) {
            action();
            return;
        }
        const bool record = gpu && mode_ == Instrumentation::events;
        if (record) {
            if (used_ == slots_.size()) throw std::logic_error("event capacity");
            slots_[used_].stage = stage_id;
            check(cudaEventRecord(slots_[used_].first), "lifecycle event start");
        }
        const auto start = Clock::now();
        action();
        sample.host_ms[stage_id] += elapsed_ms(start, Clock::now());
        if (record) {
            check(cudaEventRecord(slots_[used_].last), "lifecycle event stop");
            ++used_;
        }
    }

    void collect(LifecycleSample& sample) {
        if (used_ == 0) return;
        const auto start = Clock::now();
        check(cudaEventSynchronize(slots_[used_ - 1].last), "lifecycle event sync");
        for (std::size_t i = 0; i < used_; ++i) {
            sample.event_ms[slots_[i].stage] +=
                event_ms(slots_[i].first, slots_[i].last);
        }
        used_ = 0;
        sample.event_collection_ms += elapsed_ms(start, Clock::now());
    }

private:
    struct Slot { cudaEvent_t first = nullptr, last = nullptr; Stage stage{}; };
    void destroy() noexcept {
        for (auto& slot : slots_) {
            if (slot.last) cudaEventDestroy(slot.last);
            if (slot.first) cudaEventDestroy(slot.first);
        }
    }
    Instrumentation mode_;
    std::vector<Slot> slots_;
    std::size_t used_ = 0;
};

std::array<std::size_t, 8> footprint(std::size_t n, std::size_t d, std::size_t k) {
    return {4*n*d, 4*k*d, 4*n, 4*n, 4*k, 4,
            8*k*d*((n + reduction_tile_samples - 1)/reduction_tile_samples),
            4*k*((n + count_tile_samples - 1)/count_tile_samples)};
}

// Not exposed in any user header. All eight allocations use the same sizes,
// order, and pageable copy APIs as the retained one-shot implementation.
class ResidentWorkspace {
public:
    ResidentWorkspace(std::size_t n, std::size_t d, std::size_t k)
        : n_(static_cast<int>(n)), d_(static_cast<int>(d)), k_(static_cast<int>(k)),
          tiles_((n_ + reduction_tile_samples - 1)/reduction_tile_samples),
          count_tiles_((n_ + count_tile_samples - 1)/count_tile_samples) {
        const auto bytes = footprint(n, d, k);
        try {
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                check(cudaMalloc(&buffers_[i], bytes[i]), "lifecycle allocation");
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~ResidentWorkspace() { cleanup(); }
    ResidentWorkspace(const ResidentWorkspace&) = delete;
    ResidentWorkspace& operator=(const ResidentWorkspace&) = delete;

    void release() {
        for (std::size_t i = buffers_.size(); i != 0; --i) {
            void* ptr = buffers_[i - 1];
            buffers_[i - 1] = nullptr;
            if (ptr) check(cudaFree(ptr), "lifecycle free");
        }
    }
    void upload(const std::vector<float>& input, Recorder& recorder,
                LifecycleSample& sample) {
        recorder.step(sample, h2d, true, [&] {
            check(cudaMemcpy(get<float>(0), input.data(),
                             static_cast<std::size_t>(n_)*d_*sizeof(float),
                             cudaMemcpyHostToDevice), "lifecycle input H2D");
        });
        uploaded_ = true;
    }
    void fit(std::size_t cap, Recorder& recorder, LifecycleSample& sample) {
        if (!uploaded_) throw std::logic_error("resident input not uploaded");
        const auto start = Clock::now();
        const int sample_blocks = (n_ + block_size - 1)/block_size;
        const int centroid_blocks = (k_*d_ + block_size - 1)/block_size;
        // Restart every fit, regardless of previous pass parity/termination.
        current_ = get<int>(2);
        int* next = get<int>(3);
        updates_ = 0;
        converged_ = false;
        recorder.step(sample, initialization, true, [&] {
            initialize_centroids<<<centroid_blocks, block_size>>>(
                get<float>(0), get<float>(1), n_, d_, k_);
            check_launch("resident initialization");
        });
        recorder.step(sample, initial_assignment, true, [&] {
            assign_samples<<<sample_blocks, block_size>>>(
                get<float>(0), get<float>(1), current_, nullptr, nullptr, n_, d_, k_);
            check_launch("resident initial assignment");
        });
        for (std::size_t pass = 1; pass <= cap; ++pass) {
            // Every count/partial element is overwritten; no scratch memset.
            recorder.step(sample, count, true, [&] {
                count_tile_partials<<<count_tiles_, block_size>>>(
                    current_, get<int>(7), n_, k_);
                check_launch("resident count partials");
                finalize_counts<<<1, block_size>>>(
                    get<int>(7), get<int>(4), count_tiles_, k_);
                check_launch("resident count finalization");
            });
            recorder.step(sample, centroid_update, true, [&] {
                centroid_tile_partials<<<dim3(tiles_, k_*d_), block_size>>>(
                    get<float>(0), current_, get<double>(6), n_, d_, tiles_);
                check_launch("resident centroid partials");
                finalize_centroids<<<centroid_blocks, block_size>>>(
                    get<double>(6), get<int>(4), get<float>(1), d_, k_, tiles_);
                check_launch("resident centroid finalization");
            });
            recorder.step(sample, flag_reset, true, [&] {
                check(cudaMemset(get<int>(5), 0, sizeof(int)), "resident flag reset");
            });
            recorder.step(sample, reassignment, true, [&] {
                assign_samples<<<sample_blocks, block_size>>>(
                    get<float>(0), get<float>(1), next, current_, get<int>(5),
                    n_, d_, k_);
                check_launch("resident reassignment");
            });
            int changed = 0;
            recorder.step(sample, flag_d2h, true, [&] {
                check(cudaMemcpy(&changed, get<int>(5), sizeof(int),
                                 cudaMemcpyDeviceToHost), "resident flag D2H");
            });
            std::swap(current_, next);
            updates_ = pass;
            if (changed == 0) {
                converged_ = true;
                break;
            }
        }
        recorder.collect(sample);
        sample.fit_wall_ms += elapsed_ms(start, Clock::now());
    }
    Result download(Recorder& recorder, LifecycleSample& sample) {
        if (updates_ == 0) throw std::logic_error("resident fit not run");
        Result output;
        recorder.step(sample, output_allocation, false, [&] {
            output.labels.resize(n_);
            output.centroids.resize(static_cast<std::size_t>(k_)*d_);
        });
        output.update_count = updates_;
        output.converged = converged_;
        recorder.step(sample, labels_d2h, true, [&] {
            check(cudaMemcpy(output.labels.data(), current_, n_*sizeof(int),
                             cudaMemcpyDeviceToHost), "resident labels D2H");
        });
        recorder.step(sample, centroids_d2h, true, [&] {
            check(cudaMemcpy(output.centroids.data(), get<float>(1),
                             static_cast<std::size_t>(k_)*d_*sizeof(float),
                             cudaMemcpyDeviceToHost), "resident centroids D2H");
        });
        recorder.collect(sample);
        return output;
    }

private:
    template <typename T> T* get(std::size_t i) const {
        return static_cast<T*>(buffers_[i]);
    }
    void cleanup() noexcept {
        for (std::size_t i = buffers_.size(); i != 0; --i)
            if (buffers_[i - 1]) cudaFree(buffers_[i - 1]);
    }
    int n_, d_, k_, tiles_, count_tiles_;
    std::array<void*, 8> buffers_{};
    int* current_ = nullptr;
    std::size_t updates_ = 0;
    bool converged_ = false, uploaded_ = false;
};

void require_same(const Result& expected, const Result& actual) {
    if (expected.labels != actual.labels ||
        expected.centroids.size() != actual.centroids.size() ||
        std::memcmp(expected.centroids.data(), actual.centroids.data(),
                    expected.centroids.size()*sizeof(float)) != 0 ||
        expected.update_count != actual.update_count ||
        expected.converged != actual.converged)
        throw std::runtime_error("resident lifecycle differs from retained CUDA");
}

}  // namespace

LifecycleReport characterize_cuda_lifecycle(
    const std::vector<float>& input, std::size_t n, std::size_t d,
    std::size_t k, std::size_t max_updates, std::size_t runs,
    std::size_t sequences) {
    validate(input, n, d, k, max_updates);
    if (!((runs == 0 && sequences == 0) ||
          (runs >= 7 && runs <= 30 && sequences >= 5 && sequences <= 20)))
        throw std::invalid_argument("require runs>=7/sequences>=5, or both zero");
    const auto before = input;
    LifecycleReport report;
    report.device_bytes = footprint(n, d, k);
    report.result = kmeans_cuda_parallel_count_with_update_cap(
        input, n, d, k, max_updates);
    Recorder none(Instrumentation::none);
    Recorder host(Instrumentation::host);
    LifecycleSample discarded;

    // Before timing, check every output of twenty independent resident fits,
    // then twenty fits without intervening downloads. Public reduced-cap cases
    // exercise nonconvergence and odd/even final label-buffer ownership.
    {
        ResidentWorkspace owner(n, d, k);
        owner.upload(input, none, discarded);
        for (int i = 0; i < 20; ++i) {
            owner.fit(max_updates, none, discarded);
            require_same(report.result, owner.download(none, discarded));
            ++report.checked_fits;
        }
        for (int i = 0; i < 20; ++i) owner.fit(max_updates, none, discarded);
        require_same(report.result, owner.download(none, discarded));
        ++report.checked_fits;
        owner.release();
    }
    if (runs == 0) {
        if (std::memcmp(before.data(), input.data(), input.size()*sizeof(float)))
            throw std::runtime_error("lifecycle modified input");
        return report;
    }
    for (int i = 0; i < 2; ++i)
        require_same(report.result, kmeans_cuda_parallel_count_with_update_cap(
            input, n, d, k, max_updates));

    const auto one_shot = [&](Recorder& recorder, LifecycleSample& sample) {
        const auto start = Clock::now();
        recorder.step(sample, validation, false, [&] {
            validate(input, n, d, k, max_updates);
        });
        std::unique_ptr<ResidentWorkspace> owner;
        recorder.step(sample, allocation, false, [&] {
            owner = std::make_unique<ResidentWorkspace>(n, d, k);
        });
        owner->upload(input, recorder, sample);
        for (std::size_t i = 0; i < sample.fits; ++i)
            owner->fit(max_updates, recorder, sample);
        auto output = owner->download(recorder, sample);
        recorder.step(sample, free, false, [&] { owner->release(); owner.reset(); });
        sample.wall_ms = elapsed_ms(start, Clock::now());
        require_same(report.result, output);  // Outside timed region.
    };
    const auto preallocated = [&](ResidentWorkspace& owner, bool upload,
                                   Recorder& recorder, LifecycleSample& sample) {
        const auto start = Clock::now();
        if (upload) {
            recorder.step(sample, validation, false, [&] {
                validate(input, n, d, k, max_updates);
            });
            owner.upload(input, recorder, sample);
        }
        owner.fit(max_updates, recorder, sample);
        auto output = owner.download(recorder, sample); // New host output each call.
        sample.wall_ms = elapsed_ms(start, Clock::now());
        require_same(report.result, output);
    };
    {
        ResidentWorkspace b(n, d, k), c(n, d, k);
        c.upload(input, none, discarded); // Validated once before allocation.
        for (int warm = 0; warm < 3; ++warm) {
            preallocated(b, true, none, discarded);
            preallocated(c, false, none, discarded);
        }
        for (std::size_t round = 0; round < runs; ++round) {
            // Include an uninstrumented shadow lifecycle to expose overhead
            // from the benchmark-only host orchestration itself.
            for (std::size_t offset = 0; offset < 4; ++offset) {
                const auto mode = (round + offset)%4;
                LifecycleSample sample;
                sample.instrumentation = "none";
                if (mode == 0) {
                    sample.mode = "A_native";
                    const auto start = Clock::now();
                    auto output = kmeans_cuda_parallel_count_with_update_cap(
                        input, n, d, k, max_updates);
                    sample.wall_ms = elapsed_ms(start, Clock::now());
                    require_same(report.result, output);
                } else if (mode == 3) {
                    sample.mode = "A_shadow";
                    one_shot(none, sample);
                } else {
                    sample.mode = mode == 1 ? "B_preallocated" : "C_resident_input";
                    preallocated(mode == 1 ? b : c, mode == 1, none, sample);
                }
                report.samples.push_back(std::move(sample));
            }
        }
        b.release();
        c.release();
    }
    const std::array<std::size_t, 5> repetitions{1, 2, 5, 10, 20};
    for (const auto fits : repetitions) {
        LifecycleSample warm;
        warm.fits = fits;
        one_shot(none, warm);
    }
    for (std::size_t round = 0; round < sequences; ++round)
        for (std::size_t offset = 0; offset < repetitions.size(); ++offset) {
            LifecycleSample sample;
            sample.mode = "D_amortized";
            sample.instrumentation = "none";
            sample.fits = repetitions[(round + offset)%repetitions.size()];
            one_shot(none, sample);
            report.samples.push_back(std::move(sample));
        }
    // Direct host API intervals: GPU launch API times and flag-copy waits
    // overlap device execution and must NOT be added to CUDA-event durations.
    for (std::size_t i = 0; i < runs; ++i) {
        LifecycleSample sample;
        sample.mode = "A_shadow";
        sample.instrumentation = "host";
        one_shot(host, sample);
        report.samples.push_back(std::move(sample));
    }
    const auto events_start = Clock::now();
    auto events = std::make_unique<Recorder>(
        Instrumentation::events, 5*max_updates + 5);
    report.diagnostic_event_setup_ms = elapsed_ms(events_start, Clock::now());
    {
        ResidentWorkspace b(n, d, k), c(n, d, k);
        c.upload(input, none, discarded);
        for (std::size_t round = 0; round < sequences; ++round) {
            for (int mode = 0; mode < 3; ++mode) {
                LifecycleSample sample;
                sample.instrumentation = "events";
                sample.mode = mode == 0 ? "A_shadow" :
                              mode == 1 ? "B_preallocated" : "C_resident_input";
                if (mode == 0) one_shot(*events, sample);
                else preallocated(mode == 1 ? b : c, mode == 1, *events, sample);
                report.samples.push_back(std::move(sample));
            }
            for (const auto fits : repetitions) {
                LifecycleSample sample;
                sample.mode = "D_amortized";
                sample.instrumentation = "events";
                sample.fits = fits;
                one_shot(*events, sample);
                report.samples.push_back(std::move(sample));
            }
        }
        b.release();
        c.release();
    }
    const auto events_end = Clock::now();
    events.reset();
    report.diagnostic_event_destroy_ms = elapsed_ms(events_end, Clock::now());
    if (std::memcmp(before.data(), input.data(), input.size()*sizeof(float)))
        throw std::runtime_error("lifecycle modified input");
    return report;
}

}  // namespace benchmark
