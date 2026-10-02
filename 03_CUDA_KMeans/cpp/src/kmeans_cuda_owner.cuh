// Included inside namespace kmeans by kmeans_cuda.cu so this supported owner
// launches the identical retained kernels, without exposing them publicly.
struct CudaKMeansBuffer::Impl {
    enum class State { empty, uploaded, fitted, failed };

    struct Resources {
        DeviceBuffer<float> input, centroids;
        DeviceBuffer<int> labels_a, labels_b, counts, changed;
        DeviceBuffer<double> partials;
        DeviceBuffer<int> count_partials;
        Resources(std::size_t n, std::size_t d, std::size_t k,
                  std::size_t tiles, std::size_t count_tiles)
            : input(n * d), centroids(k * d), labels_a(n), labels_b(n),
              counts(k), changed(1), partials(k * d * tiles),
              count_partials(k * count_tiles) {}
    };

    const std::size_t n, d, k;
    const int tiles, count_tiles;
    int device = 0;
    State state = State::empty;
    FitMetadata metadata;
    int* current_labels = nullptr;
    std::unique_ptr<Resources> resources;

    Impl(std::size_t samples, std::size_t dimensions, std::size_t clusters)
        : n(samples), d(dimensions), k(clusters),
          tiles(static_cast<int>((n + reduction_tile_samples - 1) /
                                 reduction_tile_samples)),
          count_tiles(static_cast<int>((n + count_tile_samples - 1) /
                                       count_tile_samples)) {
        check(cudaGetDevice(&device), "owner construction device");
        // Member RAII unwinds every successful allocation if a later one fails.
        resources = std::make_unique<Resources>(n, d, k, tiles, count_tiles);
    }

    ~Impl() noexcept {
        // Free on the owning device even if the caller changed its current
        // device. Restore caller state where possible. CUDA cleanup cannot
        // report errors from a noexcept destructor (including lost contexts).
        int previous = device;
        const bool known = cudaGetDevice(&previous) == cudaSuccess;
        if (!known || previous != device) cudaSetDevice(device);
        resources.reset();
        if (known && previous != device) cudaSetDevice(previous);
    }

    void require_usable() {
        if (state == State::failed) {
            throw std::logic_error("CUDA owner failed; reconstruct before reuse");
        }
        int current = -1;
        const auto status = cudaGetDevice(&current);
        if (status != cudaSuccess) {
            state = State::failed;
            check(status, "owner current device");
        }
        if (current != device) {
            throw std::logic_error("CUDA owner requires its construction device");
        }
    }
};

CudaKMeansBuffer::CudaKMeansBuffer(std::size_t n, std::size_t d, std::size_t k) {
    // Check before arithmetic, narrowing or any CUDA allocation.
    if (n < 2 || n > max_n || d < 1 || d > max_d ||
        k < 2 || k > max_k || k > n) {
        throw std::invalid_argument(
            "require 2 <= K <= min(N,32), K <= N <= 2^20, 1 <= D <= 32");
    }
    impl_ = std::make_unique<Impl>(n, d, k);
}

CudaKMeansBuffer::~CudaKMeansBuffer() noexcept = default;

std::size_t CudaKMeansBuffer::device_bytes() const noexcept {
    const auto& p = *impl_;
    return sizeof(float) * (p.n * p.d + p.k * p.d) +
           sizeof(int) * (2 * p.n + p.k + 1 + p.k * p.count_tiles) +
           sizeof(double) * p.k * p.d * p.tiles;
}

void CudaKMeansBuffer::upload(const std::vector<float>& input) {
    auto& p = *impl_;
    p.require_usable();
    // The same complete O(ND) contract as one-shot; no skip-validation flag.
    // Validate before invalidating old output: rejected input has no effect.
    validate(input, p.n, p.d, p.k, 100);
    p.state = Impl::State::failed;
    p.metadata = {};
    p.current_labels = nullptr;
    check(cudaMemcpy(p.resources->input.get(), input.data(),
                     input.size() * sizeof(float), cudaMemcpyHostToDevice),
          "owner input H2D");
    p.state = Impl::State::uploaded;
}

FitMetadata CudaKMeansBuffer::fit() {
    return fit_with_update_cap(100);
}

FitMetadata CudaKMeansBuffer::fit_with_update_cap(std::size_t max_updates) {
    auto& p = *impl_;
    p.require_usable();
    if (max_updates < 1 || max_updates > 100) {
        throw std::invalid_argument("update cap must be between 1 and 100");
    }
    if (p.state == Impl::State::empty) {
        throw std::logic_error("CUDA owner fit requires upload");
    }
    // Any CUDA failure from here poisons the owner, preventing stale output.
    p.state = Impl::State::failed;
    p.metadata = {};
    auto& r = *p.resources;
    const int n = static_cast<int>(p.n), d = static_cast<int>(p.d);
    const int k = static_cast<int>(p.k);
    const int sample_blocks = (n + block_size - 1) / block_size;
    const int centroid_blocks = (k * d + block_size - 1) / block_size;
    p.current_labels = r.labels_a.get();
    int* next_labels = r.labels_b.get();
    initialize_centroids<<<centroid_blocks, block_size>>>(
        r.input.get(), r.centroids.get(), n, d, k);
    check_launch("owner initialization launch");
    assign_samples<<<sample_blocks, block_size>>>(
        r.input.get(), r.centroids.get(), p.current_labels,
        nullptr, nullptr, n, d, k);
    check_launch("owner initial assignment launch");

    for (std::size_t pass = 1; pass <= max_updates; ++pass) {
        // Counts and both partial arrays are completely overwritten before
        // use. Initial assignment and reassignment overwrite all labels.
        // Only the atomic change flag requires a memset on each update.
        count_tile_partials<<<p.count_tiles, block_size>>>(
            p.current_labels, r.count_partials.get(), n, k);
        check_launch("owner count partials launch");
        finalize_counts<<<1, block_size>>>(
            r.count_partials.get(), r.counts.get(), p.count_tiles, k);
        check_launch("owner count finalize launch");
        centroid_tile_partials<<<dim3(p.tiles, k * d), block_size>>>(
            r.input.get(), p.current_labels, r.partials.get(), n, d, p.tiles);
        check_launch("owner centroid partials launch");
        finalize_centroids<<<centroid_blocks, block_size>>>(
            r.partials.get(), r.counts.get(), r.centroids.get(), d, k, p.tiles);
        check_launch("owner centroid finalize launch");
        check(cudaMemset(r.changed.get(), 0, sizeof(int)), "owner flag reset");
        assign_samples<<<sample_blocks, block_size>>>(
            r.input.get(), r.centroids.get(), next_labels,
            p.current_labels, r.changed.get(), n, d, k);
        check_launch("owner reassignment launch");
        int changed = 0;
        // Blocking copy completes all preceding default-stream work and
        // surfaces asynchronous errors; no extra event/synchronization added.
        check(cudaMemcpy(&changed, r.changed.get(), sizeof(int),
                         cudaMemcpyDeviceToHost), "owner convergence D2H");
        std::swap(p.current_labels, next_labels);
        p.metadata.update_count = pass;
        if (changed == 0) {
            p.metadata.converged = true;
            break;
        }
    }
    p.state = Impl::State::fitted;
    return p.metadata;
}

Result CudaKMeansBuffer::download() {
    auto& p = *impl_;
    p.require_usable();
    if (p.state != Impl::State::fitted) {
        throw std::logic_error("CUDA owner download requires a completed fit");
    }
    Result output;
    // Host allocation failure leaves resident results valid for a retry.
    output.labels.resize(p.n);
    output.centroids.resize(p.k * p.d);
    output.update_count = p.metadata.update_count;
    output.converged = p.metadata.converged;
    p.state = Impl::State::failed;
    check(cudaMemcpy(output.labels.data(), p.current_labels,
                     p.n * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
          "owner labels D2H");
    check(cudaMemcpy(output.centroids.data(), p.resources->centroids.get(),
                     p.k * p.d * sizeof(float), cudaMemcpyDeviceToHost),
          "owner centroids D2H");
    p.state = Impl::State::fitted;
    return output;
}
