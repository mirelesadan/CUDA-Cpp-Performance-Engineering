# Project 2 — General-Purpose CUDA K-Means

## Objective

Develop, profile, validate, and optimize a general-purpose K-means implementation across Python/reference, C++, CPU-optimized, and CUDA stages. The authoritative contract, profiled CPU/CUDA engineering, supported persistent native ownership, Python CUDA integration, and controlled scikit-learn comparison are complete on the tested Windows environment. Final project closeout and transition to matrix/tensor multiplication are next; the measured scope and remaining limitations are explicit below.

## Why this project exists

K-means demonstrates that the portfolio's performance-engineering skills generalize beyond microscopy to a broadly applicable machine-learning algorithm. Its assignment and cluster-update phases offer different computation, reduction, synchronization, and memory challenges.

## Frozen baseline contract

`fit_kmeans(X, K)` accepts only a finite, C-contiguous NumPy `float32` array of shape `(N,D)`, with `2 <= K <= min(N,32)`, `N <= 2^20`, `1 <= D <= 32`, and every `|X[i,d]| <= 1024`. It permits at most 100 update/reassignment passes. `X` is never modified. There are no weights, normalization, sparse inputs, restarts, or random seeds inside the fit.

Initial centroid `k` is an exact `float32` copy of input row `((2*k+1)*N)//(2*K)`, calculated with integer arithmetic. Assignment computes squared Euclidean distance with separate `float32` subtraction, squaring, and addition, accumulating features in ascending index order. Lowest-index cluster wins exact computed ties. Centroid updates accumulate assigned samples in `float64`, divide by the integer count in `float64`, then round once to `float32`; an empty cluster retains its previous centroid exactly.

After one initial assignment, each pass updates centroids from current labels and reassigns against them. A pass counts toward `update_count`; matching old/new labels returns `converged=True`. If the cap is reached with changing labels, return the last updated centroids and their new labels, `converged=False`, with no hidden extra update. In that case, returned centers need not be the means of returned labels. Outputs are new C-contiguous `int32[N]` labels and `float32[K,D]` centroids, plus update count and convergence flag.

The [authoritative NumPy reference](python/reference.py) computes assignments in bounded sample chunks and never materializes an `N × K × D` tensor. It uses explicit `float32` ufunc boundaries for distances and `float64` accumulation for updates. Validation inertia is a separate `float64` chunked calculation from the returned labels/centroids, not part of the fit or its FP32 assignment rule.

## Public correctness fixtures

The readable [public cases](fixtures/public_cases.json) freeze inputs, initialization rows, labels, centroids, update counts, convergence flags, and validation inertia:

| Case | Distinct behavior |
| --- | --- |
| `separated_signed_and_limit` | separated groups, signed/zero coordinates, allowed magnitude endpoints, one-update convergence |
| `exact_tie_duplicate_seeds_empty_cluster` | exact lowest-index tie, duplicate points and initial centers, persistent empty-centroid retention |
| `singleton_cluster_zero_inertia` | singleton cluster and an exactly representable zero-inertia result |
| `three_updates` | moving assignments and convergence after three updates |
| `update_cap_without_hidden_update` | same input capped at one pass, nonconverged final reassignment |
| `fp32_distance_rounding` | a point assigned to cluster 0 under FP32 rounding that FP64 distance would send to cluster 1 |

The capped case uses a private reduced-update test hook, not a public fit parameter, rather than searching for a contrived natural 100-pass example. A separate exactly representable constant-point self-test also requires zero inertia.

Run the [self-tests](python/test_reference.py) from the repository root with `python -B 03_CUDA_KMeans/python/test_reference.py`. They check the frozen expected outputs, seed indices, ties, ordered FP32 arithmetic, immutability, deterministic repeatability, independent output ownership, invalid input rejection, and fixture schema/contract version. Verified locally with Python 3.12.7 and NumPy 1.26.4.

## Correctness-first serial C++ baseline

The standalone [C++17 implementation](cpp/src/kmeans_serial.cpp) accepts an unchanged row-major `std::vector<float>` plus `(N,D,K)` and returns independently owned `std::vector<std::int32_t>` labels, `std::vector<float>` centroids, update count, and convergence flag. It follows the reference's visibly separate phases: validate, copy deterministic seed rows, assign, accumulate/update, reassign, and test label stability. The baseline deliberately uses nested scalar loops, with no OpenMP, CUDA, SIMD, blocking, or fused stages. A test-only reduced-cap entry point exercises the nonconverged return state; the normal API always allows 100 passes.

Assignment uses separate `float` subtraction, multiplication, and addition in ascending feature order; strict less-than comparison preserves lowest-index ties. MSVC Release builds use `/O2` and explicitly `/fp:precise` without fast-math or FP contraction. Centroid sums visit samples in order as `double`, divide in `double`, and cast once to `float`; empty centroids are copied unchanged. This has been validated on Windows/MSVC, not Linux.

From the repository root, configure with `cmake -S 03_CUDA_KMeans/cpp -B 03_CUDA_KMeans/cpp/build -G "Visual Studio 17 2022" -A x64`, then build with `cmake --build 03_CUDA_KMeans/cpp/build --config Release`. Run the native contract test with `ctest --test-dir 03_CUDA_KMeans/cpp/build -C Release --output-on-failure`. Run the [fixture driver](python/test_cpp_serial.py) with `python -B 03_CUDA_KMeans/python/test_cpp_serial.py 03_CUDA_KMeans/cpp/build/Release/phase2_kmeans_serial.exe`; add `--benchmark` for the controlled baseline timing. Temporary raw-FP32 files bridge tests to the executable; this is not a Python binding. Project 2's default CMake build has no Project 1, OpenMP, CUDA, or Python dependency; OpenMP is separately opt-in below.

All six public fixtures passed: 96/96 labels exact, 0 centroid bit mismatches, identical update counts and convergence flags, and passing independently recomputed inertia. This includes the FP32-versus-FP64 assignment discriminator and the test-only one-pass nonconvergence case. Native checks cover seed indices, invalid inputs, input immutability, independent result storage, and deterministic repeated runs. The Python self-tests (13, including iterative-generator reproducibility) and native contract test pass; the prior Project 1 regressions remain unchanged.

### Initial whole-fit timing—not a profile

The [version-1 `profiling` generator](python/benchmark_data.py) produced `(N,D,K)=(65,536,8,16)` with PCG64 seed `20260924`. On this Windows/MSVC x64 Release build, the input was generated and loaded before timing. One untimed warm-up preceded seven `std::chrono::steady_clock` complete-fit calls; each includes normal validation and algorithm/output allocations, but excludes the raw-file bridge. Native times (ms): `10.234200, 9.846900, 9.732000, 9.769400, 9.836700, 10.043400, 10.112700`; min/median/max: `9.732000 / 9.846900 / 10.234200`. The fit converged after one update: `6.655` million samples/s and `212.976` million sample-cluster distance evaluations/s, counting the initial assignment and one reassignment. Full-workload labels, count, flag, and centroids matched the Python reference exactly.

For context only, the NumPy semantic reference took `30.920000, 31.565800, 29.956200` ms (one warm-up, three Python-call wall timings; median `30.920000` ms) on the same input. This is a historical, unprofiled baseline, not the fresh measurement below.

### Serial baseline profiling checkpoint

The normal x64 Release implementation remains unchanged (`/O2 /fp:precise`, no phase timers). On the same approved `profiling` workload, a fresh one-warm-up/seven-run whole-fit sequence gave `15.131700, 14.372400, 14.378200, 14.353800, 14.399800, 14.511600, 14.217500` ms; min/median/max `14.217500 / 14.378200 / 15.131700` ms. It still converged after one update (two assignment passes). This session was slower than the historical `9.846900` ms median; the cause is unestablished. Symbol-enabled and phase-timed builds were close to this session's normal Release, so their instrumentation does not explain the historical difference. Do not compare either median as an optimization gain.

The unchanged fit also converged after one update on the approved `gpu` `(262,144,16,16)` workload (seven-run median `86.754500` ms) and optional `stress_optional` `(1,048,576,32,32)` workload (seven-run median `1246.135100` ms; considerable run-to-run variation). To observe normal iterative behavior without changing these approved datasets, the [separate deterministic generator](python/benchmark_data.py) adds `iterative_profile`: `(16,384,8,8)`, PCG64 seed `20260927`, FP64 Gaussian center/noise draws with standard deviation 2, cyclic planted labels shuffled, one FP32 cast, and no seed-row overwrite. Its input byte SHA-256 is `97359ddaf01fe506a28efc48159760324af6db9226d56e5a6624b20d62c81904`. C++ and the Python reference agreed on the complete result and natural convergence after 22 updates (23 assignments); its seven-run native median was `25.216000` ms. This diagnostic workload does not replace the approved performance workloads.

Coarse phase timing uses a separately built, symbol-enabled Release target with timers only around complete phases, never inside point loops. Each figure below is the median of seven fits after one warm-up; individual medians need not sum to the median total. Input generation and raw-file I/O are excluded; validation and fit allocations are included.

| Workload | Complete fit | Validation | Initialization | All assignment | Centroid updates | Convergence check | Passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `profiling` | 14.3592 ms | 0.7597 | 0.0014 | 13.3128 (92.7%) | 0.2863 (2.0%) | 0.0154 | 2 assignments / 1 update |
| `iterative_profile` | 24.2839 ms | 0.1872 | 0.0005 | 22.4407 (92.4%) | 1.4796 (6.1%) | 0.0056 | 23 assignments / 22 updates |

For source-level evidence, an optimized symbol build without phase timers requested a 1 ms Windows timer period and slept 1 ms between main-thread instruction-pointer samples, buffering PCs during repeated fits and resolving PDB source/function locations afterward. A later sampler smoke test observed a roughly 2 ms mean actual interval, so the request is not an exact cadence guarantee. The approved workload yielded 12,568 samples (11,969 line-resolved; 1,500 fits); the iterative workload yielded 9,512 (9,435 line-resolved; 700 fits). Assignment accounted for 91.27% and 92.09% of all samples, respectively; centroid update for 1.92% and 6.79%. On the approved workload, the distance-accumulation source line had 36.23%, the combined input/centroid address-load-subtract line 27.51%, the square line 5.35%, cluster/feature loop lines together 12.06%, and strict minimum/selection lines together 9.29%. On the iterative workload these were approximately 33.78%, 27.32%, 4.26%, 13.46%, and 12.47%. The FP64 update sum line increased from 1.51% to 5.62%; count/division/cast and convergence had no separately significant attribution. Inlined optimized instructions can map ambiguously to source, especially the combined address/load/subtract expression; these are directional sample shares, not independently timed operation costs. Profiler-run wall time is not a benchmark.

Both one-update and 22-update fits are assignment-dominated. The update phase becomes more relevant with iteration count and will matter for future parallel reductions, but it is not the primary serial bottleneck here. The first selected optimization was repeated assignment-row address calculation; its isolated result follows below.

Reproduce the whole-fit checks with `python -B 03_CUDA_KMeans/python/profile_workloads.py 03_CUDA_KMeans/cpp/build/Release/phase2_kmeans_serial.exe profiling` (or `gpu`, `stress_optional`, `iterative_profile`). For phase timing and sampling, configure a separate ignored build with `-DPROJECT2_ENABLE_PHASE_TIMING=ON -DPROJECT2_ENABLE_PROFILING_SYMBOLS=ON`; run the corresponding `phase2_kmeans_serial_phase_timing.exe` through that driver, or `phase2_kmeans_serial_sampler.exe` with `--sample-repeats`. Both options default to `OFF`; normal Release and public API remain uninstrumented. No generated arrays or profiler reports are versioned.

### Isolated serial experiment 1 — assignment row addressing

The retained `kmeans_serial_addressed` path computes a sample-row pointer once per sample and a centroid-row pointer once per cluster, then uses the same ascending feature offsets. The original `kmeans_serial` assignment and fit remain callable and unchanged. Subtraction, square, ordered FP32 addition, strict cluster comparison, FP64 centroid update, and all other fit decisions are identical. The candidate is a separate path rather than a change to the numerical contract.

Both paths passed all six public fixtures (96 exact labels each, zero centroid-bit mismatches), including the FP32-sensitive label-0 case, reduced-cap nonconvergence, inertia checks, invalid inputs, input immutability, independent outputs, and repeatability. Native tests and the Project 1 public Phase A/B regression smoke tests passed. In every timed pair on both larger workloads, baseline and candidate labels, centroid bits, update counts, and convergence flags matched exactly; candidate outputs also matched the Python oracle.

The normal MSVC x64 Release build (`/O2 /fp:precise`, no timers/symbol profiling) loaded input once, warmed both paths, then timed complete fits with `std::chrono::steady_clock`. Each seven-pair primary block alternated baseline/candidate and candidate/baseline order; three blocks reversed their starting order as baseline/addressed/baseline. Validation and fit allocations remained inside each timed call; file I/O, result comparisons, and output serialization were outside. Raw first-sequence times (ms) show every primary call:

| Block | Baseline: seven calls | Addressed: seven calls | Median baseline/addressed |
| --- | --- | --- | ---: |
| 1 | 17.3435, 17.0801, 17.4447, 17.2707, 17.0932, 17.3634, 17.5040 | 16.3291, 16.7475, 16.1374, 17.0416, 16.2643, 16.2157, 16.2935 | 17.3435 / 16.2935 |
| 2 | 17.1903, 17.0361, 17.6090, 17.3000, 17.5366, 17.2602, 17.5111 | 16.0903, 16.4803, 16.7146, 16.6305, 17.9678, 16.3202, 16.2016 | 17.3000 / 16.4803 |
| 3 | 17.8546, 17.3402, 17.5767, 17.4755, 21.2794, 19.2651, 23.7187 | 16.9695, 16.3030, 16.3756, 16.3819, 18.3592, 18.6169, 18.2099 | 17.8546 / 16.9695 |

Across these 21 calls per path, baseline min/median/max was `17.0361 / 17.4447 / 23.7187` ms and addressed was `16.0903 / 16.3819 / 18.6169` ms: `1.064876×` speedup, `6.0924%` less runtime, `4.0005` million samples/s, and `128.0164` million sample-cluster distance evaluations/s for the candidate. Because late calls drifted, a bounded second 21-pair confirmation used the same reversing-start protocol; its complete raw times (ms) were:

| Block | Baseline: seven calls | Addressed: seven calls | Median baseline/addressed |
| --- | --- | --- | ---: |
| 1 | 19.8575, 17.3348, 17.6895, 17.2538, 19.4931, 27.9677, 19.3436 | 16.2662, 16.5022, 16.4005, 16.5677, 16.3212, 16.2888, 16.2241 | 19.3436 / 16.3212 |
| 2 | 17.4443, 17.1590, 23.8452, 17.1959, 18.1253, 22.1492, 17.4077 | 16.0900, 16.8359, 17.1511, 16.3405, 16.4837, 16.8326, 16.1225 | 17.4443 / 16.4837 |
| 3 | 17.4659, 17.5925, 17.4778, 18.5338, 19.1282, 17.2930, 17.1994 | 16.8685, 16.5510, 17.2762, 18.4982, 16.4072, 16.7353, 16.2399 | 17.4778 / 16.7353 |

Second-sequence pooled baseline/candidate medians were `17.5925/16.4837` ms (`1.067266×`, `6.3027%` reduction). Its first block includes baseline outliers, so individual block ratios are not treated as precise gains. The controlled sequences consistently favor the candidate, but their absolute times differ from both historical baselines for unresolved environment reasons.

On the 22-update `iterative_profile` diagnostic, the first five paired baseline times were `30.0779, 32.2918, 32.8941, 34.9107, 40.7526` ms (median `32.8941`); addressed times were `29.4573, 28.9914, 30.0874, 31.0668, 64.5520` ms (median `30.0874`, `1.093285×`). The final addressed call was an outlier, so three more five-pair blocks were checked; their pooled medians were `31.0433/28.1952` ms (`1.101014×`), with block medians `32.9520/28.8420`, `29.9879/28.1496`, and `29.7204/28.1429`. No repeatable multi-update regression was observed.

A fresh symbol-only Windows main-thread sampling comparison on the primary workload collected 18,904 baseline and 15,311 candidate samples (1,500 fits each; requested 1 ms timer, observed mean intervals 2.854/3.868 ms). Assignment remained dominant at 91.11%/89.79%; FP32 distance-addition lines drew 35.81%/34.69%, square lines 3.79%/5.38%, cluster selection about 8.9% in both, and centroid update 2.02%/2.27%. The old combined address/load/subtract line drew 29.61%; candidate samples spread across the difference/load (8.67%), centroid-row setup (10.98%), and cluster-loop line (19.00%, up from 7.16%). The targeted *single line* shrank, but source remapping/inlining prevents claiming that all its former cost disappeared. Sample counts and profiler wall times are not speedup measurements; the paired unprofiled timings are the retention evidence.

The addressed path is retained as a readable first CPU improvement. The remaining ordered FP32 accumulation (~35% source attribution) and cluster-minimum bookkeeping (~9%) motivated the bounded scheduling experiment below. Reproduce the retained addressing comparison with `python -B 03_CUDA_KMeans/python/benchmark_assignment_addressing.py 03_CUDA_KMeans/cpp/build/Release/phase2_kmeans_serial.exe profiling` (or `iterative_profile`); the sampler accepts `--variant addressed` through the profiling driver.

### Isolated serial experiment 2 — two-feature distance pipeline (rejected)

An experimental candidate built on the addressed path prepared adjacent FP32 differences and squares before two **sequential, feature-ordered** additions, with a scalar odd-feature tail. It left sample/cluster order, strict selection, and centroid updates untouched. All six public fixtures passed against Python and the addressed control; native tests also covered D=1, D=2, D=3, and a D=3 ordered-addition discriminator. On the primary and 22-update workloads, every paired output matched the addressed control bit for bit, including labels, centroid bits, update count, and convergence.

The normal MSVC x64 Release (`/O2 /fp:precise`) paired harness warmed both paths, then alternated call order within three seven-pair primary blocks, reversing each block's starting order. Each timed call included the complete fit and allocations, but excluded input I/O, comparisons, and output serialization. Raw primary times (ms):

| Block | Addressed control: seven calls | Pipeline: seven calls | Median control/pipeline |
| --- | --- | --- | ---: |
| 1 | 9.4992, 9.3559, 9.2150, 9.4163, 9.3384, 9.5071, 9.6376 | 10.5082, 10.6022, 10.5334, 10.4678, 10.7023, 10.5889, 10.5979 | 9.4163 / 10.5889 |
| 2 | 9.5800, 9.1939, 9.3373, 10.1949, 9.6801, 9.7573, 9.9032 | 10.5045, 10.6566, 10.5355, 11.2471, 10.5616, 10.4966, 10.5079 | 9.6801 / 10.5355 |
| 3 | 9.4283, 9.4996, 9.8738, 9.4789, 9.3464, 9.3611, 9.3313 | 11.0689, 10.5822, 10.6147, 10.6389, 10.4875, 10.6201, 10.8222 | 9.4283 / 10.6201 |

Pooled min/median/max was `9.1939/9.4789/10.1949` ms for the addressed control and `10.4678/10.5889/11.2471` ms for the pipeline: control/candidate speedup `0.895173×`, or **11.7102% longer runtime**. Candidate throughput was `6.1891` million samples/s and `198.0519` million sample-cluster distance evaluations/s, versus `6.9139` and `221.2442` for the control. On the 22-update diagnostic, three five-pair blocks gave pooled `16.3654/17.7841` ms control/candidate (`0.920226×`, 8.6689% longer); every block favored the control. These are fresh same-session comparisons only. Absolute control time again shifted relative to earlier sessions, so historical medians are not used for a speedup claim.

A small optimized-object check found scalar FP32 subtraction, multiplication, and additions in the required order, a scalar odd tail, strict cluster selection, and no FMA or reassociated reduction. MSVC had already unrolled/prepared **four** features per loop in the addressed control; the candidate emitted a distinct **two**-feature loop. This is consistent with, but does not independently prove, the measured regression. The candidate source/test/harness changes were discarded; no new sampling profile was warranted. Cluster-selection bookkeeping remains a smaller ~9% sampled opportunity, whose likely ceiling did not justify another serial micro-optimization. This selected the portable OpenMP experiment below, with the retained addressed serial path as control.

### Assignment-only portable OpenMP scaling

The opt-in `kmeans_openmp(input, N, D, K, thread_count)` path statically partitions independent sample indices for **each** assignment. Each worker keeps the addressed serial path's sample/centroid row pointers, ascending cluster and feature order, separate FP32 subtract/square/ordered add, and strict `<` tie rule. The path calls the **same existing serial centroid-update function** after every assignment, so FP64 sample-order sums, empty-cluster retention, convergence checks, and update counts are not parallel reductions. Dynamic OpenMP teams are disabled for the call and the actual team size is checked; the prior runtime setting is restored. CMake discovers `OpenMP::OpenMP_CXX` only when `PROJECT2_ENABLE_OPENMP=ON` (default `OFF`); normal Release uses MSVC `/O2 /fp:precise`, and GCC/Clang use `-ffp-contract=off`. Linux compilation/scaling has not yet been validated.

All six authoritative public fixtures passed at 1, 2, 4, 8, 16, and 20 threads: 96 exact labels, zero centroid-bit mismatches, and identical update counts/convergence flags versus the retained addressed serial path. The reduced-cap nonconvergence and FP32-sensitive cases passed; an additional native D=3 ordered-addition discriminator, invalid-input/thread-count checks, input immutability, independent outputs, and repeated calls passed. Both benchmark workloads matched the addressed control bit for bit at every tested count. The default CPU-only build, its native test, the opt-in OpenMP native tests, and Project 1's public Phase A/B regression executables passed.

On the 14-core/20-logical-processor i7-13700H, OpenMP exposed 20 processors and a 20-thread maximum; the runtime supplied each requested team size. Static sample chunks differ by at most one row (at 20 threads: 3,276–3,277 primary rows or 819–820 iterative rows). Python generates the input and the normal x64 Release executable loads it before timing, warms each configuration once, then rotates serial and six OpenMP configurations through seven `steady_clock` whole-fit rounds. Fit validation/output allocations are included; file I/O and exact comparisons are excluded. Throughput uses `N / fit_time` samples/s and `N × K × (1 + update_count) / fit_time` sample-cluster distance evaluations/s. The following is one complete controlled sequence; all times are milliseconds and the serial denominator is fresh in that sequence.

| Primary `(65,536,8,16)`, one update | Seven raw fits (ms) | Min / median / max | Speedup | Efficiency | Msamples/s | Mdistances/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Addressed serial | 19.4941, 26.7399, 31.9212, 32.5697, 29.2342, 28.0908, 24.0578 | 19.4941 / 28.0908 / 32.5697 | 1.000× | — | 2.3330 | 74.6562 |
| OpenMP-1 | 18.3983, 19.0673, 27.7629, 23.6298, 23.3964, 21.2478, 19.9020 | 18.3983 / 21.2478 / 27.7629 | 1.322× | 1.322 | 3.0844 | 98.6997 |
| OpenMP-2 | 9.2508, 9.9236, 13.1254, 12.4319, 12.6980, 10.6050, 9.9887 | 9.2508 / 10.6050 / 13.1254 | 2.649× | 1.324 | 6.1797 | 197.7512 |
| OpenMP-4 | 6.6232, 7.4427, 8.9850, 10.6029, 9.6181, 6.6294, 5.8487 | 5.8487 / 7.4427 / 10.6029 | 3.774× | 0.944 | 8.8054 | 281.7730 |
| OpenMP-8 | 5.6924, 5.5919, 8.5685, 8.8671, 7.1609, 6.5468, 6.7966 | 5.5919 / **6.7966** / 8.8671 | **4.133×** | 0.517 | **9.6425** | **308.5590** |
| OpenMP-16 | 7.8467, 5.4740, 7.6373, 9.9954, 7.5209, 7.4636, 5.9488 | 5.4740 / 7.5209 / 9.9954 | 3.735× | 0.233 | 8.7139 | 278.8432 |
| OpenMP-20 | 6.9675, 6.6148, 7.9501, 13.4985, 7.8722, 11.9601, 15.3213 | 6.6148 / 7.9501 / 15.3213 | 3.533× | 0.177 | 8.2434 | 263.7894 |

The 22-update diagnostic `(16,384,8,8)` produced 23 assignment passes. It is not a replacement for the official primary benchmark:

| Iterative diagnostic | Seven raw fits (ms) | Min / median / max | Speedup | Efficiency | Msamples/s | Mdistances/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Addressed serial | 28.2629, 46.0209, 47.2611, 49.5545, 48.3039, 50.4658, 40.5103 | 28.2629 / 47.2611 / 50.4658 | 1.000× | — | 0.3467 | 63.7873 |
| OpenMP-1 | 23.3444, 28.2831, 38.1574, 36.2447, 35.3213, 34.7804, 32.5662 | 23.3444 / 34.7804 / 38.1574 | 1.359× | 1.359 | 0.4711 | 86.6769 |
| OpenMP-2 | 13.5077, 15.3685, 20.6043, 17.7045, 19.6880, 18.7388, 17.4515 | 13.5077 / 17.7045 / 20.6043 | 2.669× | 1.335 | 0.9254 | 170.2763 |
| OpenMP-4 | 9.0004, 12.4797, 12.8184, 12.2889, 15.1445, 11.8797, 10.6626 | 9.0004 / **12.2889** / 15.1445 | **3.846×** | 0.961 | **1.3332** | **245.3154** |
| OpenMP-8 | 8.9495, 13.4263, 12.0619, 13.3460, 12.8406, 13.8901, 12.6334 | 8.9495 / 12.8406 / 13.8901 | 3.681× | 0.460 | 1.2760 | 234.7753 |
| OpenMP-16 | 15.2107, 13.0860, 13.3786, 12.7381, 15.0289, 14.3010, 12.6073 | 12.6073 / 13.3786 / 15.2107 | 3.533× | 0.221 | 1.2246 | 225.3342 |
| OpenMP-20 | 13.2437, 17.8102, 14.3791, 15.7043, 14.8532, 15.9377, 18.9370 | 13.2437 / 15.7043 / 18.9370 | 3.009× | 0.150 | 1.0433 | 191.9637 |

A bounded second normal-Release sequence confirmed the multicore benefit despite substantial laptop timing drift: primary serial/OpenMP-1/4/8/16/20 medians were `23.7479/19.3620/5.8780/5.8367/6.0609/6.7084` ms; iterative medians were `44.6436/33.6416/11.1317/10.4071/12.6211/13.2583` ms. Eight threads had the lowest primary median in both sequences; four and eight were nearly tied in the primary confirmation. OpenMP-1 ran 24.4% faster than serial in the first primary sequence and 18.5% faster in confirmation, so no one-thread framework penalty is visible. That improvement is **not** credited to parallelism: different compiler loop code generation and host variability have not been separated.

Coarse phase timers were enabled only in a separate Release build, not the speedup build; medians below are directional, and independently computed phase medians do not need to sum to whole-fit medians.

| Workload / path | Whole fit | Assignment | Serial centroid update | Update share |
| --- | ---: | ---: | ---: | ---: |
| Primary serial | 20.6935 ms | 18.9981 ms | 0.5492 ms | 2.7% |
| Primary OpenMP-8 | 5.6322 | 4.3742 | 0.3388 | 6.0% |
| Primary OpenMP-16 | 5.0777 | 3.7641 | 0.5222 | 10.3% |
| Iterative serial | 39.8555 | 36.0765 | 3.4305 | 8.6% |
| Iterative OpenMP-8 | 9.9627 | 7.4784 | 2.0292 | 20.4% |
| Iterative OpenMP-16 | 11.4459 | 7.9310 | 3.0170 | 26.4% |

Assignment still consumes about 69–78% of phase-timed fits at higher thread counts. Repeated serial updates become a meaningful ceiling in the iterative case but are not the sole or dominant scaling limit; 16–20 threads also show worse/noisy whole-fit results. Equal-size static chunks make sample-count imbalance unlikely, while hybrid-core scheduling, cache/memory effects, and repeated parallel-region overhead remain plausible but unmeasured. This readable CPU path is retained; the next portfolio experiment is a correctness-first CUDA K-means baseline, **not** a numerically risky parallel centroid reduction with limited primary-workload upside.

Reproduce with `cmake -S 03_CUDA_KMeans/cpp -B 03_CUDA_KMeans/cpp/build -DPROJECT2_ENABLE_OPENMP=ON` (add the appropriate platform generator/Release selection), then build Release and run CTest plus `python -B 03_CUDA_KMeans/python/benchmark_openmp.py PATH_TO_SCALING_EXE fixtures`, `profiling`, or `iterative_profile`. The driver defaults to this machine's 1/2/4/8/16/20 counts and accepts `--counts` for other hosts. A separate ignored build with `-DPROJECT2_ENABLE_PHASE_TIMING=ON` supplies the optional `--phase-executable` comparison. Generated workloads, binaries, and timing logs are not committed.

## Correctness-first CUDA baseline

The opt-in `kmeans_cuda` path keeps the input, centroids, two label arrays, integer counts, and change flag resident on the GPU throughout each fit. A seed-copy kernel uses the frozen integer row formula. Assignment launches one thread per sample in 256-thread blocks, visits clusters and features in ascending order, and uses explicit round-to-nearest `__fsub_rn`, `__fmul_rn`, and `__fadd_rn` with strict `<` tie handling. CUDA fast math is not enabled; the build also disables FMA contraction. Each count worker scans all labels in sample order; each `(cluster,feature)` update worker scans the same ordered labels and accumulates matching FP32 inputs in FP64 with `__dadd_rn`, divides with `__ddiv_rn`, and converts once with `__double2float_rn`. Empty clusters retain their prior bits. This deliberately redundant update is a correctness-first control, not a reduction optimization.

An integer change flag is copied to the host once per update pass; complete labels and centroids return only after convergence or the update cap. Swapping current/next label buffers returns the final reassignment without an extra update, including the reduced-cap nonconvergence case. The regular `Result` API has no profiling events; a separate diagnostic API records stage times.

The public six-case fixture passed against the frozen NumPy result: 96/96 labels, zero centroid-bit mismatches, exact update counts and flags, and matching independently recomputed inertia. The FP32 rounding, duplicate-seed/empty-cluster, and reduced-cap cases passed. D=1, D=3 ordered arithmetic, D=4, K=2/32, signed-zero, and subnormal edge checks also matched centroid bits. On the primary `(65,536,8,16)`, GPU `(262,144,16,16)`, and 22-update `(16,384,8,8)` workloads, CUDA, addressed serial, OpenMP-8, and NumPy agreed on all labels, centroid bits, counts, and flags; input was unchanged. Serial, OpenMP, and CUDA native contract tests passed, as did a separate default CPU-only Release build/test. Windows/MSVC is verified; Linux is not.

On the AC-powered RTX 4070 Laptop GPU, CUDA 12.9, sm_89, and MSVC Release `/O2`, input generation and raw-file I/O occurred before timing. Each normal whole-fit `steady_clock` call includes validation, allocations, H2D, all device work, D2H, and teardown; it excludes output comparison and serialization. The fresh addressed-serial and OpenMP-8 controls each had one warm-up; CUDA had three. Call order rotated across seven primary/GPU rounds and five iterative rounds. Raw whole-fit times are milliseconds; parentheses give medians:

| Workload | Addressed serial raw (median) | OpenMP-8 raw (median) | CUDA native-wall raw (median) |
| --- | --- | --- | --- |
| Primary | 10.504, 14.589, 11.564, 10.273, 10.407, 10.873, 11.802 (10.873) | 2.904, 4.313, 2.474, 2.484, 2.487, 2.896, 2.726 (2.726) | 7.837, 7.971, 8.103, 7.721, 7.621, 7.859, 7.933 (7.859) |
| GPU | 58.461, 66.186, 58.544, 80.273, 63.984, 58.795, 66.131 (63.984) | 17.797, 19.031, 18.697, 18.047, 19.216, 19.871, 18.211 (18.697) | 50.375, 50.216, 50.845, 54.638, 50.461, 49.037, 60.156 (50.461) |
| Iterative | 16.914, 20.368, 19.272, 21.934, 18.242 (19.272) | 4.667, 6.205, 4.430, 5.172, 5.709 (5.172) | 38.090, 36.240, 36.158, 36.986, 48.044 (36.986) |

Fresh serial/OpenMP-8 speedups versus CUDA native wall were `1.384×/0.347×` primary, `1.268×/0.371×` GPU, and `0.521×/0.140×` iterative; ratios below one mean CUDA is slower. Absolute laptop CPU time has varied across sessions, so only these same-session ratios are used. This deliberately unoptimized CUDA baseline is not yet competitive with OpenMP-8 on these workloads.

Three separate event-instrumented fits per workload provide directional cumulative stage medians (ms), not the normal-call speedup denominator:

| Workload | Setup* | H2D | Seed init | First assign | Count | Centroid update | Reassign | Flag reset/D2H | Final D2H | Device algorithm | One-shot GPU path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary | 0.545 | 0.374 | 0.124 | 0.045 | 0.798 | 4.899 | 0.070 | 0.092 | 0.083 | 6.049 | 6.546 |
| GPU | 0.227 | 2.809 | 0.035 | 0.455 | 3.021 | 45.951 | 0.475 | 0.091 | 0.275 | 49.947 | 53.324 |
| Iterative | 0.163 | 0.113 | 0.107 | 0.012 | 4.688 | 29.703 | 0.390 | 1.249 | 0.048 | 35.321 | 36.385 |

*Setup is host-clock allocation/event creation; component timings are CUDA-event measurements, and device-algorithm/one-shot totals are sums of those components. Device algorithm excludes the tiny convergence-flag D2H; one-shot GPU includes it and H2D/final D2H, but excludes setup and host validation/teardown. Individually computed medians need not sum. The first primary diagnostic setup call was a 196.786 ms outlier, and one initialization event was 1.432 ms; diagnostic runs are not a stable allocation benchmark. Centroid update alone occupied approximately 81%, 92%, and 84% of device-algorithm medians; count plus update occupied approximately 94%, 98%, and 97%. Transfer cost is secondary here. The 22-update cumulative count/update/flag results reveal the iterative reduction cost directly; no CUDA kernel optimization was made.

Reproduce with `cmake -S 03_CUDA_KMeans/cpp -B 03_CUDA_KMeans/cpp/build/cuda -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" -DPROJECT2_ENABLE_CUDA=ON -DPROJECT2_ENABLE_OPENMP=ON -DPROJECT2_CUDA_ARCHITECTURES=89`, then build Release, run CTest, and run `python -B 03_CUDA_KMeans/python/benchmark_cuda.py PATH_TO_EXE all` in the existing Python environment. CUDA and OpenMP are off by default; CUDA can build without OpenMP, while the combined benchmark target requires both. The driver creates temporary raw arrays only; no benchmark data or generated binaries are committed.

### Focused count/update profile — reduction decision checkpoint

Nsight Compute CLI 2025.2.1 profiled one warmed, normal (non-diagnostic) launch of each unchanged kernel on the generated `gpu` workload `(N,D,K)=(262,144,16,16)`. A separate MSVC Release / CUDA 12.9 / `sm_89` build added only `-DCMAKE_CUDA_FLAGS=-lineinfo`; the normal build remains unchanged. For each kernel, `ncu --kernel-name regex:count_clusters` (or `regex:update_centroids`) used `--launch-skip 3 --launch-count 1` on the existing CUDA benchmark with `--runs 1 --diagnostic-runs 0`, after input loading and three correctness-checked CUDA warm-ups. The counter pass selected LaunchStats, Occupancy, SpeedOfLight, SchedulerStats, WarpStateStats, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, and InstructionStats; a separate SourceCounters pass imported the CUDA source. Profiler replay durations and whole-fit wall times are **not** benchmark timings; the unprofiled CUDA-event stage medians above (count 3.021 ms, update 45.951 ms) remain the performance reference. Generated input and `.ncu-rep` reports remain ignored.

| Metric | Count | Centroid update |
| --- | ---: | ---: |
| Grid × block; useful workers | 1 × 256; 16 clusters | 1 × 256; 256 cluster-feature pairs |
| Registers/thread; static/dynamic shared memory; local load/store sectors | 40; 0/0; 0/0 | 31; 0/0; 0/0 |
| Achieved occupancy; active warps/active SM | 2.08%; 1 | 16.67%; 8 |
| SM / DRAM / L1-TEX / L2 throughput (peak %) | 0.11 / 0.08 / 4.01 / 0.04 | 2.39 / 0.54 / 10.74 / 0.26 |
| L1 / L2 hit rate | 87.50 / 56.27% | 78.75 / 13.56% |
| Average global-load data bytes per 32-byte sector; branch efficiency | 4.00; 100% | 9.60; 100% |
| Scheduler cycles with no eligible warp | 91.22% | 93.46% |
| Dominant PC-sampled stalls | Long scoreboard 9,734/12,762 (76.3%) | Short scoreboard 47,481/76,405 (62.1%); long 20,463 (26.8%) |

Each grid has only one block for 36 SMs (about 0.028 blocks/SM), regardless of the reported occupancy *within its active SM*. Count has only half a warp of useful cluster workers; update has eight useful warps on one SM. Neither is device-wide DRAM-bandwidth limited. Count waits mainly on label-load dependencies, while update combines repeated label loads with a dependent FP64 sum: Nsight reported FP64 as 85.94% utilized among active compute pipelines, though device-wide SM throughput was only 2.39%. Zero local load/store sectors provide no evidence of spills. The low average useful bytes/sector are driven by broadcast-like label loads, not excess sectors in the conditional feature loads: source counters attributed 262,144 ideal sectors to count labels; update had 2,097,152 ideal sectors for labels and 524,288 ideal sectors for input values. Cache-hit rates and DRAM bytes vary with replay/cache state; these percentages are directional, not independent timing shares.

Source mapping places count's label load/comparison/increment together at its inner-loop line, with most long-scoreboard samples there. For update, the label comparison line has 7,672 long-scoreboard samples; the input-load/FP64-add expression has 12,714 long- and 30,554 short-scoreboard samples, and loop/index instructions have another 16,915 short-scoreboard samples. The compiler unrolled loops and mapped multiple SASS instructions onto those expressions, so these line counts do not isolate precise load, arithmetic, or indexing time. Division/cast/store had negligible sampled stalls. Branch efficiency was 100% for both kernels; control divergence is not the main limit.

On this all-nonempty GPU workload, count logically inspects `N·K = 4,194,304` labels per Lloyd update; update inspects `N·K·D = 67,108,864` labels while accumulating only `N·D = 4,194,304` input feature values. These are algorithmic operation counts, **not** DRAM traffic: they scale as `O(NK)`, `O(NKD)`, and `O(ND)` respectively. Empty clusters skip their update scans, making `N·K_nonempty·D` the general update count. The frozen contract requires exact labels, update count/convergence, centroid error at most `5e-6 × feature_scale`, and inertia error at most `2e-5 × scale` (with an exact-zero-inertia fixture). The baseline's bitwise-identical centroids are stronger than that contract; even a deterministic fixed-order parallel FP64 reduction can change sample-order sum bits. The current CUDA benchmark also applies a stronger exact-bit gate, so a future non-bitwise candidate must deliberately distinguish that control comparison from frozen contractual validation without weakening exact label/termination checks.

Ranked designs: (1) **deterministic two-stage centroid reduction**—parallel fixed sample tiles produce FP64 partial sums, followed by a fixed-order final reduction using the unchanged count kernel; highest expected gain from many blocks and a shorter sum chain, moderate complexity and rounding-order risk. (2) **parallel integer count reduction**, leaving ordered centroid sums untouched; low numerical risk and moderate complexity but addresses only the smaller count stage. (3) **atomic FP64 sum/count accumulation**; ample parallelism but contention, nondeterministic FP64 order, and the highest reproducibility risk. The selected next *single* experiment is design 1, with the current count kernel and assignment unchanged. No kernel optimization was made during this checkpoint.

### Isolated CUDA experiment — deterministic tiled FP64 centroid reduction

The retained control still assigns one worker to each `(cluster,feature)` and scans all `N` labels in sample order. The separate candidate changes only centroid update. Stage A uses fixed 4,096-sample tiles; one 256-thread block owns each `(cluster,feature,tile)`, each thread accumulates its assigned samples in ascending order in FP64, and a fixed shared-memory binary tree writes one FP64 partial. Stage B adds that pair's partials in ascending tile order, divides by the **unchanged** integer count in FP64, and rounds once to FP32. Empty clusters retain previous centroid bits. Assignment, count, layout, and convergence remain unchanged. On `(262,144,16,16)`, Stage A is a `(64,256)` grid (16,384 blocks) and Stage B one 256-thread block. Workspace is `8 × ceil(N/4096) × K × D` bytes: 16,384 partials / 128 KiB on this workload, at most 2 MiB within the frozen dimension limits; Stage A also uses 2 KiB shared memory/block.

The candidate is deterministic but reorganizes FP64 additions; baseline centroid-bit equality is **not** an acceptance rule. The frozen rule remains exact labels, update count and convergence; every centroid coordinate within `5e-6 × max(1,max_i |X[i,j]|)`; validation inertia within `2e-5 × max(1,|I_ref|)`, including the exact-zero fixture. Six public fixtures (96 exact labels), D=1/3/4/32, K=2/31/32, signed-zero/subnormal cases, and N=4095/4096/4097/8193 tile boundaries passed; repeated candidate results were bitwise identical and input unchanged. The primary, GPU, and 22-update iterative workloads matched the authoritative NumPy reference on labels/termination and had zero measured centroid absolute/scaled error, zero inertia error, and zero centroid-bit differences **in these tested cases**. The old baseline's stricter exact-bit tests remain intact; both variants pass all three Release CTest targets.

For the GPU workload, persistent device buffers, copied input, and frozen first-assignment labels/counts preceded direct timing. Two warm-ups per variant preceded seven AB/BA-interleaved CUDA-event pairs; control timing enclosed only the original update kernel, candidate timing enclosed both tiled kernels without an intermediate host sync. Raw direct-update times (ms): control `33.521664, 33.510208, 33.512222, 33.510399, 33.512447, 33.533695, 33.516544` (min/median/max `33.510208/33.512447/33.533695`); tiled `1.205248, 1.186816, 1.202176, 1.185792, 1.204224, 1.184768, 1.202048` (`1.184768/1.202048/1.205248`). The paired median speedup is `27.879×`, a `96.41%` reduction; Stage-A/Stage-B medians were `1.189760/0.012288` ms. Historical 45.951 ms event timing and profiler replay durations are not speedup denominators.

Normal whole-fit Release `steady_clock` comparisons included allocations, validation, H2D, computation, D2H, and teardown, with three warm-ups per CUDA variant. GPU control/candidate fits ran adjacently in alternating AB/BA order; CPU controls ran in the same AC-powered session, and output checks were outside the timer. Fresh control/tiled/OpenMP-8 medians (ms) were primary `7.6621/3.2920/2.9560` (7 runs; CUDA `2.327×`), GPU `45.5468/12.9150/19.1134` (7 runs; `3.527×`), and iterative `32.8411/6.9164/4.8964` (5 runs; `4.748×`). The GPU tiled fit beats fresh OpenMP-8 by `1.480×`; the other two sizes do not. Separate event-instrumented fits put 22-update cumulative tiled centroid work at `0.760640` ms versus `26.269760` ms for the control; these diagnostic medians are not whole-fit speedup denominators. Cross-session GPU-clock and transfer/initialization variability remain, so claims use fresh paired medians, not best runs.

A focused Nsight Compute 2025.2.1 check on a warmed normal Stage-A launch found 39 registers/thread, 2 KiB static shared memory/block, zero local load/store sectors, 98.71% occupancy (47.38 active warps/SM), 86.88% device-wide SM throughput, 6.50% DRAM throughput, and 91.06% cycles with no eligible scheduler warp; L1TEX-queue throttle dominated at about 62.2% of cycles per issued instruction. Stage B had 36 registers/thread and one block, so still underfills the GPU, but its direct-event median was only ~1% of candidate update time. The grid-wide occupancy/SM-throughput change and shorter per-thread sum chains support the speedup; no logical `N·K·D` label scans were eliminated, and the relative contribution of those two improvements was not isolated. The candidate is retained. The now-unchanged count kernel costs about 3.02 ms on the GPU and 4.62 ms cumulatively over 22 iterative updates, exceeding the new centroid-update cost; isolated parallel integer counting is next.

### Isolated CUDA experiment — parallel integer cluster counting

The retained `kmeans_cuda_parallel_count` path changes only counting relative to `kmeans_cuda_tiled`, which remains callable with its original serial-scan count kernel. Assignment, tiled FP64 centroid reduction, layout, convergence, and Lloyd semantics are unchanged. Stage A assigns one 256-thread block to each 1,024-label tile. Threads cooperatively initialize a 32-integer shared histogram, load up to four labels each, and increment block-local counters with exact integer atomics; after a barrier they write `partials[tile*K+cluster]`. There are no global atomics. Stage B uses one 256-thread block, with its first `K` threads adding partials in ascending tile order. Empty clusters remain zero; partial tiles are guarded. Each label is inspected once rather than `K` times. The frozen `N <= 2^20` bound prevents count overflow regardless of atomic order.

On the GPU workload `(262,144,16,16)`, Stage A has 256 blocks and Stage B has 16 useful workers. Count scratch is `4*ceil(N/1024)*K` bytes: 16 KiB here, 4 KiB primary, 512 bytes iterative, and at most 128 KiB under the contract. The unchanged centroid scratch adds 128 KiB on the GPU workload (144 KiB combined). Stage A uses 128 bytes of static shared memory/block; Stage B uses none.

**Correctness:** an untimed validation-only fit compares candidate and control counts on the actual labels consumed by **every** Lloyd update, before reassignment, and checks their sum equals `N`. All six public fixtures (96 labels; eight update passes and 18 cluster-count comparisons) pass, including reduced-cap semantics. Primary/GPU/iterative workloads check 1/1/22 passes and 16/16/176 cluster counts. Labels, centroid bits, update counts, flags, repeated results, and input immutability match the retained tiled path; centroid absolute/scaled error and independently recomputed inertia error versus NumPy are zero on all three workloads. Forty additional known-label cases cover K=2/32, N=1023/1024/1025/4097/2^20, empty clusters, all-in-one clusters, cyclic labels, and extreme imbalance. Complete-fit tile edges, signed zero, subnormals, and existing serial/OpenMP/CUDA contract tests also pass. The frozen centroid/inertia tolerances remain unchanged.

**Timing:** AC-powered RTX 4070 Laptop GPU, driver 610.88, CUDA 12.9, `sm_89`, MSVC Release `/O2`, unchanged `--fmad=false`. Persistent direct-count buffers/events and label upload precede timing; two warm-ups per variant precede eight alternating AB/BA pairs. Candidate timing encloses both kernels with two events and no midpoint event/synchronization. A second eight-pair block confirms the result; pooled medians retain all 16 observations. Component timings use separate launches with an extra midpoint event, so their sums/fractions are only approximate. GPU-workload raw count times (ms):

| Block | Control | Parallel |
| --- | --- | --- |
| First | 3.172128, 3.164160, 3.164992, 3.164992, 3.173184, 3.164160, 3.169024, 3.163136 | 0.025600, 0.012288, 0.026624, 0.030688, 0.022528, 0.012128, 0.013312, 0.011264 |
| Confirmation | 3.169664, 3.172256, 3.175328, 3.164160, 3.189760, 3.164928, 3.187552, 3.164160 | 0.059552, 0.030560, 0.025600, 0.011264, 0.028672, 0.013312, 0.029696, 0.013120 |

| Direct-count workload | Control min / median / max (ms) | Parallel min / median / max (ms) | Median speedup |
| --- | ---: | ---: | ---: |
| Primary | 0.794624 / 0.795648 / 0.808960 | 0.008192 / 0.011808 / 0.034816 | 67.382× |
| GPU | 3.163136 / 3.167008 / 3.189760 | 0.011264 / 0.024064 / 0.059552 | 131.608× |
| Iterative, fixed final labels / one count | 0.202752 / 0.209920 / 0.229824 | 0.011264 / 0.013888 / 0.039936 | 15.115× |

The GPU count reduction is 99.240%; separate block speedups are 176.618× and 116.854×. The very short candidate launches remain variable, but the large gain repeats. Five separate diagnostic fits measure **cumulative** count over all 22 iterative updates at 4.959296 ms control versus 0.702400 ms parallel; the fixed-label direct microbenchmark is not that cumulative total.

Normal whole-fit `steady_clock` calls include validation, allocation, H2D, computation, D2H, and teardown; checks and I/O remain outside. Each of two eight-round blocks uses three CUDA warm-ups and one CPU warm-up, adjacent AB/BA CUDA fits, and fresh OpenMP-8 controls in the same session. Pooled 16-run medians:

| Workload | Control / parallel / OpenMP-8 (ms) | Control / parallel speedup | OpenMP-8 / parallel speedup |
| --- | ---: | ---: | ---: |
| Primary `(65,536,8,16)` | 2.854500 / 2.036050 / 2.789000 | 1.402× | 1.370× |
| GPU `(262,144,16,16)` | 14.240450 / 10.863650 / 17.775350 | 1.311× | 1.636× |
| Iterative `(16,384,8,8)`, 22 updates | 7.212000 / 2.777600 / 4.694850 | 2.596× | 1.690× |

First/confirmation block whole-fit speedups are 1.433×/1.386× primary, 1.336×/1.345× GPU, and 2.639×/2.421× iterative. An initial iterative candidate call took 12.6178 ms; confirmation candidate calls ranged 2.5973–3.4553 ms. No sample was deleted. These results support retention, not universal CPU/GPU crossover claims.

**Focused profile:** Nsight Compute 2025.2.1, separate normal-optimization `-lineinfo` build, one warmed Stage-A/Stage-B launch after four matching launches (untimed audit plus three warm-ups). Focused launch/occupancy/scheduler/stall/memory/source sections were collected, not a full metric set. Default cache-flushing replay and `--cache-control none` were both checked because count normally consumes labels just written by assignment. Cache-none replay is not a perfectly controlled cache state; neither profiler durations nor profiled native wall time are benchmark results.

| Candidate metric | Stage A, cache-none replay | Stage B, cache-none replay |
| --- | ---: | ---: |
| Blocks × threads; registers/thread | 256 × 256; 16 | 1 × 256; 40 |
| Static / dynamic shared bytes; local load/store sectors | 128 / 0; 0 / 0 | 0 / 0; sectors not collected |
| Theoretical / achieved occupancy; active warps/active SM | 100% / 76.38%; 36.66 | 100% / 3.20%; 1.54 |
| SM / DRAM throughput (% peak) | 23.80 / 0.00 | 0.16 / 1.11 |
| Scheduler cycles without eligible warp | 68.76% | 91.10% |

Stage A has 0.96 eligible warps/scheduler and 100% branch efficiency. Long-scoreboard, immediate-constant-cache miss, and barrier stalls account for about 31.3%, 18.2%, and 10.5% of warp cycles per issued instruction; MIO throttle is 3.7%. L1/L2 hit rates are 0.40%/99.03%. With default cache flushing, Stage A instead has 73.92% occupancy, 9.13% SM throughput, 65.86% DRAM throughput, 3.62% L2 hits, and 91.38% no-eligible cycles: cold memory demand is substantial, but does not establish a sustained bandwidth bottleneck after assignment. Source counters show 8,192 warp label-load requests covering 32,768 ideal 32-byte sectors (four sectors/warp, full sector utilization); partial stores add 512 sectors. Actual address-footprint sectors equal ideal sectors, with no excess. These are coalescing/sector-use metrics, not DRAM-miss byte counts.

Stage B still underfills the GPU. Separate direct Stage-A/Stage-B component medians are 0.007168/0.009216 ms, so finalization is roughly half the short instrumented sequence; launch/event jitter prevents precise fractional accounting. The warm profile indicates mixed latency and short-grid/launch effects, not evidence that shared atomics dominate. Increased grid parallelism, eliminating redundant full-N scans, and block-local aggregation plausibly explain the gain; their individual contributions were not isolated.

Fresh five-fit candidate diagnostic medians (ms) are setup 0.242200, H2D 1.505920, initialization 0.035136, first assignment 0.479200, count 0.020480, centroid update 1.247232, reassignment 0.478208, convergence flag reset/D2H 0.035968, final D2H 0.176000, and device algorithm 2.268288. The one-shot GPU-path sum is 4.010240 ms. Device algorithm excludes flag D2H; one-shot GPU includes it and transfers, but not allocation/validation/teardown. Separate medians do not necessarily add and are not the normal whole-fit denominator. Count is now about 0.9% of device-algorithm time; centroid update and assignment dominate device work.

Ranked remaining opportunities: (1) characterize transfers/residency and native validation/setup/teardown/launch overhead; (2) centroid Stage-A load/stall optimization; (3) assignment optimization. **Exactly one next experiment:** transfer/residency and native host-overhead characterization, because 10.864 ms normal whole-fit versus 2.268 ms device-algorithm diagnostics leaves a larger end-to-end question than further count tuning. This comparison motivates measurement; subtracting independent medians would not precisely attribute that gap. Count/sum fusion is low priority at the measured count share. No next optimization was implemented. Linux, broad size scaling, and general crossover remain unvalidated.

Reproduce after the existing Release build with `python -B 03_CUDA_KMeans/python/benchmark_cuda.py PATH_TO_EXE all --compare-count --diagnostic-runs 5`; repeat `primary`, `gpu`, and `iterative` with `--compare-count --diagnostic-runs 0` for confirmation. The driver prints raw runs and correctness evidence. Default and `--compare-tiled` validation remain available; comparison modes are mutually exclusive. Generated inputs, timing outputs, profiler reports, and builds remain ignored.

### CUDA transfer, residency, and native host-overhead characterization

This checkpoint changes **no kernels or normal APIs**. The retained one-shot path is `kmeans_cuda_parallel_count`; plain `kmeans_cuda` remains the original correctness-first control. A separate opt-in benchmark-only owner reuses those exact retained kernels, not a new supported native/Python interface.

**Measured lifecycle:** validate dimensions, size, update cap, finiteness, and magnitude; allocate eight device buffers; pageable input H2D; frozen seed-row centroid initialization; initial assignment; parallel count, tiled centroid update, flag reset, reassignment, and blocking four-byte convergence D2H per pass; allocate fresh host output vectors; labels D2H; centroids D2H; free device buffers. Normal calls create no events and make no explicit `cudaSetDevice`, device query, or standalone `cudaDeviceSynchronize` call. Blocking D2H operations establish completion. Runtime/context first use is warmed before measurement, not claimed as a cold-start result.

The internal owner starts every fit with centroid initialization and initial assignment, restores label-buffer A/B roles, resets host termination state, and retains the original per-update flag reset. Counts and scratch are completely overwritten. Repeated fits are independent deterministic benchmark repetitions, **not chained refinement**.

| Mode | Included in timed wall interval |
| --- | --- |
| A, current one-shot | Validation, allocation, upload, full fit, fresh output allocation/download, free |
| B, preallocated/nonresident input | Validation and upload **every call**, full fit, fresh output allocation/download; device allocation/free outside |
| C, resident input | Full fresh fit and fresh output allocation/download; validation, allocation, and upload once outside |
| D, amortized lifecycle | Validation, allocation, upload once, R independent full fits, final output allocation/download once, free |

**Validation/build:** six public fixtures, six feature/FP32 edge cases, a signed-zero all-identical N=1025/K=32 partial tile, and both performance workloads pass. Each case checks 20 resident outputs individually and another final output after 20 fits without intervening downloads. Every timed final output is also checked outside timing. Labels, centroid bits, update counts, convergence flags, and repeated results match retained CUDA; NumPy centroid/inertia error is zero on these cases, and input remains unchanged. Exact centroid parity here is stronger than, and does not replace, the frozen general tolerance contract. All three serial/OpenMP/CUDA CTest targets pass; a fresh default CPU-only build/test also passes.

**Protocol/environment:** Windows laptop, RTX 4070 Laptop GPU, CUDA 12.9.86, MSVC 19.44 x64 Release `/O2`, `sm_89`, unchanged `--fmad=false`, AC power, driver **617.14** (different from the preceding experiment). Normal API and B/C receive three warm-ups; D receives a warm sequence per repetition count. Two same-session blocks provide seven normal runs per mode and five D sequences per R per block. Order rotates within blocks. All samples are retained; pooled results below use 14 normal calls or ten D sequences, not historical fastest runs. A confirmation-only, adjacent uninstrumented shadow lifecycle cross-check measured 10.160500 vs 9.799800 ms normal API on GPU, and 2.183800 vs 2.011100 ms iterative. It is a diagnostic host orchestration, not a replacement speedup baseline.

#### Wall-time boundaries and CPU context

All values are milliseconds, **min / median / max**. GPU is `(262,144,16,16)`, one update; iterative is `(16,384,8,8)`, 22 updates.

| Mode | GPU | Iterative |
| --- | ---: | ---: |
| A: current normal one-shot | 9.614700 / 10.072150 / 13.882300 | 1.509300 / 1.719000 / 2.239500 |
| B: preallocated, validate/upload every fit | 8.541700 / 8.970800 / 12.695200 | 1.500300 / 1.794200 / 5.241500 |
| C: prevalidated/uploaded input; fit + output | 2.529000 / 2.588950 / 6.447400 | 1.252800 / 1.403150 / 4.794900 |
| Fresh OpenMP-8 complete fit | 16.785800 / 18.200700 / 23.879600 | 4.301500 / 4.706950 / 5.444900 |

OpenMP-8 / CUDA median ratios for A, C, and D(20) are respectively **1.807×, 7.030×, 5.946×** on GPU and **2.738×, 3.355×, 2.946×** iterative. C excludes initial preparation; D includes amortized preparation. Neither is the current public one-shot behavior. B alone gives modest GPU savings; iterative per-block savings are only ~4–5%, and its pooled medians do not show a reliable improvement. Broad CPU/GPU size crossover was not tested.

The GPU C median moved from 5.248000 ms in the first block to 2.570800 ms in confirmation; its full range remains in the table. Iterative A moved from 1.652200 to 2.011100 ms. Short-call drift/outliers limit precise ratios. No clock/power settings were changed and their contribution was not isolated.

#### Direct host and event diagnostics

Separate host-only shadow runs time the actual validation scan, combined eight allocations/frees, launch/reset API calls, and individual copy calls. Separate CUDA-event runs defer readout until fit completion instead of synchronizing every stage. Event resources are created/destroyed outside all measured sequences (~0.4/~0.2–0.3 ms for this diagnostic pool); normal A/B/C/D use none.

| Interval, cumulative per fit | Host API wall, GPU | Host API wall, iterative | CUDA events, GPU | CUDA events, iterative |
| --- | ---: | ---: | ---: | ---: |
| validation | 4.480050 | 0.132700 | — | — |
| allocation | 0.234950 | 0.110400 | — | — |
| output_allocation | 0.207900 | 0.005400 | — | — |
| free | 0.496050 | 0.149950 | — | — |
| h2d | 2.191500 | 0.050000 | 1.695808 | 0.081456 |
| initialization | 0.028650 | 0.019550 | 0.042288 | 0.032576 |
| initial_assignment | 0.004950 | 0.006200 | 0.451584 | 0.008192 |
| count | 0.007550 | 0.399250 | 0.010240 | 0.186608 |
| centroid_update | 0.009900 | 0.208150 | 1.180672 | 0.665568 |
| flag_reset | 0.008350 | 0.075500 | 0.002048 | 0.066560 |
| reassignment | 0.004250 | 0.088000 | 0.451584 | 0.203792 |
| flag_d2h | 2.200500 | 0.858250 | 0.029216 | 0.307232 |
| labels_d2h | 0.186450 | 0.027600 | 0.190528 | 0.026624 |
| centroids_d2h | 0.025100 | 0.008250 | 0.013520 | 0.012880 |

Host-only diagnostic wall medians are 10.365250 / 2.110750 ms, versus normal A 10.072150 / 1.719000 ms. Their directly measured host intervals leave only about 0.0025 / 0.0040 ms unassigned within those diagnostic calls. However, the diagnostic's overhead/cadence prevents treating that as exact accounting for the separate normal-call median. CPU waits overlap GPU execution: notably, the **host** flag-copy interval includes waiting for preceding kernels and is not four-byte DMA cost. Do not add host API durations to device-event intervals.

The one-shot CUDA-event **path** (algorithm plus H2D, flag copies, and final copies) is GPU 4.138544 ms [3.829376, 7.452736], iterative 1.594704 ms [1.516224, 1.808000]. Algorithm-only event sums, excluding all transfers but including device flag reset, are 2.138416 ms [2.132480, 4.183744] and 1.168624 ms [1.115680, 1.351904]. They are the available device-algorithm estimate, **not pure SM busy time**: event intervals may include submission gaps, especially pageable copies and tiny iterative kernels. Event-diagnostic wall medians rise to 10.450800 / 2.491500 ms; added event submission/readout is therefore not evidence of native algorithm overhead. Per-fit event collection alone is 0.0110 / 0.0281 ms. Independently calculated component medians need not add.

Validation (~4.48 ms) is the largest measured individual GPU-workload host interval; allocation/free (~0.235/0.496 ms) is material but much smaller. Pageable H2D has ~2.19 ms host-call and ~1.70 ms event cost. Removing repeated validation/upload through C therefore has more value than allocation reuse alone. For iterative fits validation is ~0.133 ms and input H2D ~0.050 ms host-call time. The fit instead submits **112 explicit kernels** (two initial plus five per update), 22 flag memsets, and 22 blocking flag copies. Host-only launch/reset API time totals a median 0.800 ms, with 0.858 ms cumulative flag-copy waits; these include runtime submission/backpressure and GPU waits, not isolated pure CPU work. Flag-reset/flag-D2H event medians are 0.066560/0.307232 ms. Iterative synchronization/control is material and does not vanish with residency; its precise avoidable fraction remains unmeasured.

#### Full-lifecycle amortization

D includes preparation and teardown on **each sequence**. Times below retain ten sequences per R; ranges are total wall min/median/max, effective values are median total/R.

| R | GPU total min / median / max | GPU effective ms/fit | Iterative total min / median / max | Iterative effective ms/fit |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.736800 / 11.087000 / 15.204100 | 11.087000 | 2.061600 / 2.138550 / 7.866500 | 2.138550 |
| 2 | 11.928800 / 13.345850 / 15.886700 | 6.672925 | 3.258000 / 3.733100 / 9.456600 | 1.866550 |
| 5 | 18.169000 / 22.543600 / 31.275600 | 4.508720 | 7.010000 / 7.804550 / 13.705900 | 1.560910 |
| 10 | 29.518300 / 32.938550 / 39.542900 | 3.293855 | 13.636200 / 17.735650 / 21.903300 | 1.773565 |
| 20 | 54.418100 / 61.221650 / 77.971300 | 3.061083 | 27.792600 / 31.953850 / 44.417600 | 1.597692 |

Separate event diagnostics provide the following per-effective-fit medians; the host column directly measures validation + allocation/free + output allocation, **not all host overhead**. Launch API times and waits are accounted separately above and overlap device work.

| Workload, R | Algorithm event ms/fit | Once-per-sequence H2D + final D2H / R | Flag D2H event ms/fit | Direct non-fit host ms/fit |
| --- | ---: | ---: | ---: | ---: |
| gpu, 1 | 2.141840 | 1.741024 | 0.026672 | 6.913000 |
| gpu, 2 | 2.288672 | 0.986600 | 0.019712 | 3.954325 |
| gpu, 5 | 2.126163 | 0.363552 | 0.032931 | 1.340290 |
| gpu, 10 | 2.511309 | 0.213488 | 0.032237 | 0.772455 |
| gpu, 20 | 2.476190 | 0.102896 | 0.032690 | 0.392798 |
| iterative, 1 | 1.244080 | 0.123472 | 0.353456 | 0.192200 |
| iterative, 2 | 1.271560 | 0.059280 | 0.356144 | 0.092375 |
| iterative, 5 | 1.249859 | 0.026842 | 0.366720 | 0.037830 |
| iterative, 10 | 1.271581 | 0.013517 | 0.666674 | 0.018835 |
| iterative, 20 | 1.385304 | 0.006121 | 0.495585 | 0.009372 |

For GPU, 20-fit wall cost approaches ~3.06 ms/fit, with an uninstrumented fit-method median ~2.54 ms/fit; once-per-sequence transfers shrink to ~0.103 ms/fit. Residency amortizes the dominant validation/upload cost, but finite setup cost and execution variability remain. For iterative input, external transfers are already small; repeated flag exchanges remain per fit. Effective time is not monotonic across R, and 20 fits provide only a modest wall improvement versus A. No asymptotic constant or exact residual cause is inferred by subtracting independently measured medians.

#### Resident device memory

| Buffer | GPU bytes | Iterative bytes |
| --- | ---: | ---: |
| input | 16,777,216 | 524,288 |
| centroids | 1,024 | 256 |
| labels_a | 1,048,576 | 65,536 |
| labels_b | 1,048,576 | 65,536 |
| counts | 64 | 32 |
| change_flag | 4 | 4 |
| centroid_partials | 131,072 | 2,048 |
| count_partials | 16,384 | 512 |
| Total | 19,022,916 (18.141666 MiB) | 658,212 (0.627720 MiB) |

Requested bytes scale as `4ND + 4KD + 8N + 4K + 4 + 8KD*ceil(N/4096) + 4K*ceil(N/1024)`. This excludes CUDA context/allocator bookkeeping, diagnostic events, host output vectors, and on-chip shared memory. No pinned staging, asynchronous copies, new streams, Python binding, or convergence redesign was introduced.

**Decision at this checkpoint:** rank (1) production-quality explicit persistent native ownership with validation at upload, (2) iterative convergence/launch-overhead investigation, (3) centroid Stage-A kernel tuning. The selected supported-owner milestone is completed below; Python exposure follows it. Do not simply remove checks from the normal API. Pinned transfers are lower priority than avoiding repeated host validation/upload, and no further kernel tuning was performed here.

Reproduce with the existing CUDA/OpenMP Release configuration plus `-DPROJECT2_ENABLE_LIFECYCLE_BENCHMARK=ON` in a separate build directory, then run `python -B 03_CUDA_KMeans/python/characterize_cuda_lifecycle.py PATH_TO_phase2_kmeans_cuda_lifecycle.exe fixtures`, `gpu`, and `iterative`. Defaults are seven normal calls and five sequences per R; repeat the two performance commands for confirmation. The driver prints raw JSON and min/median/max summaries. The option is OFF by default and requires CUDA/OpenMP; no new dependency is needed. Generated workloads, raw logs, outputs, and build trees remain ignored. Linux validation and broader crossover remain future work.

## Supported native resident CUDA owner

[`CudaKMeansBuffer`](cpp/include/kmeans_cuda.hpp) is now a supported, noncopyable/nonmovable C++17 RAII API under the existing opt-in CUDA library. Its [implementation](cpp/src/kmeans_cuda_owner.cuh) launches the **unchanged** retained assignment, parallel integer count, tiled FP64 centroid update, and blocking convergence kernels. The normal validated one-shot API remains unchanged and callable. This is ownership engineering, not a new kernel optimization or Python binding.

```cpp
kmeans::CudaKMeansBuffer buffer(n, d, k); // validate fixed bounds, allocate once
buffer.upload(input);                   // full validation + pageable H2D
kmeans::FitMetadata status = buffer.fit(); // independent 100-cap fit; no bulk D2H
kmeans::Result result = buffer.download(); // new independently owned host vectors

buffer.upload(other_input);             // same N*D; invalidates old fit output
for (int run = 0; run < 20; ++run) status = buffer.fit();
result = buffer.download();             // one final bulk download
```

Every fit starts again from the frozen seed rows of the uploaded input, resets label roles/metadata, and overwrites consumed counts/partials. Repeated fits are independent reruns, **not chained Lloyd refinement**. Only the existing per-update change flag needs resetting. `fit_with_update_cap(1..100)` is a diagnostic counterpart for the frozen nonconvergence fixture; normal `fit()` always uses 100.

Construction fixes `2 <= K <= min(N,32)`, `N <= 2^20`, `1 <= D <= 32`. Upload requires exactly `N*D` finite floats with absolute value at most 1024. Its O(ND) scan and H2D occur once per upload; fit keeps no host-input pointer/vector and cannot rescan it. Caller input is unchanged and can be modified/destroyed after upload. No skip-validation flag weakens one-shot behavior.

Lifecycle: empty -> uploaded -> fitted. Fit before upload and download before a completed fit throw `logic_error`. A successful replacement upload invalidates output; an invalid argument throws `invalid_argument` **without** changing prior input/output. Repeated downloads preserve resident output and allocate independent vectors. Host output-allocation failure also preserves the resident result for retry. CUDA allocation/copy/launch/completion failures throw contextual exceptions; a failed operation poisons an existing owner, whose subsequent upload/fit/download require reconstruction. Partial construction unwinds allocations. Destruction frees all eight resources without throwing; cleanup errors (for example a lost context) cannot be reported by the destructor.

Input, centroids, two label arrays, counts, flag, FP64 centroid partials and integer count partials are allocated once. `device_bytes()` reports requested bytes, not allocator/context overhead: **19,022,916 bytes (18.141666 MiB)** GPU and **658,212 bytes (0.627720 MiB)** iterative, exactly the preceding eight-buffer footprint, with no added persistent device storage. Operations require the construction device to be current; destruction selects that device for cleanup and restores the caller's device where possible. The owner is synchronous, not thread-safe, and must not outlive a CUDA context reset. No multi-GPU support is claimed.

### Supported-owner validation and build

All six public fixtures, six existing FP32/feature edge cases, the N=1025/K=32 signed-zero partial tile, and both measured workloads pass. Each tests 20 independent fits with downloads and X1 -> X2 -> X1 replacement. Labels, update count, convergence and centroid bits match retained CUDA; centroid/inertia errors against NumPy are zero on these cases. The frozen tolerance contract remains authoritative beyond them. Native owner tests additionally cover invalid states/inputs, inclusive bounds, capped/full reseeding, partial count/centroid tiles, 20 X1 plus 20 X2 fits, independent downloads, and host mutation to NaNs after upload (resident results stay unchanged). Input is immutable. No actual CUDA fault was injected; alternate-device checks skip on this single-GPU machine.

All four serial/OpenMP/CUDA/owner CTest targets pass. CPU-only remains valid; CUDA-without-OpenMP also builds/tests, and the owner needs neither Python nor the internal lifecycle benchmark option. For the benchmark only, enable both existing `PROJECT2_ENABLE_CUDA=ON` and `PROJECT2_ENABLE_OPENMP=ON` in the documented Release build; `PROJECT2_ENABLE_LIFECYCLE_BENCHMARK` stays OFF. Run:

```text
cmake -S 03_CUDA_KMeans/cpp -B 03_CUDA_KMeans/cpp/build/cuda12_9 -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" -DPROJECT2_ENABLE_CUDA=ON -DPROJECT2_ENABLE_OPENMP=ON -DPROJECT2_ENABLE_LIFECYCLE_BENCHMARK=OFF
cmake --build 03_CUDA_KMeans/cpp/build/cuda12_9 --config Release
ctest --test-dir 03_CUDA_KMeans/cpp/build/cuda12_9 -C Release --output-on-failure
python -B 03_CUDA_KMeans/python/benchmark_cuda_owner.py 03_CUDA_KMeans/cpp/build/cuda12_9/Release/phase2_kmeans_cuda_owner_benchmark.exe fixtures
python -B 03_CUDA_KMeans/python/benchmark_cuda_owner.py 03_CUDA_KMeans/cpp/build/cuda12_9/Release/phase2_kmeans_cuda_owner_benchmark.exe gpu
python -B 03_CUDA_KMeans/python/benchmark_cuda_owner.py 03_CUDA_KMeans/cpp/build/cuda12_9/Release/phase2_kmeans_cuda_owner_benchmark.exe iterative
```

### Fresh native wall timings

Same-session AC-powered Windows laptop, i7-13700H / RTX 4070 Laptop, driver 617.14, CUDA 12.9.86, MSVC 19.44 x64 Release `/O2`, `sm_89`, unchanged `--fmad=false`. Workloads and generators are unchanged: GPU `(262144,16,16)` takes one update, iterative `(16384,8,8)` takes 22. After correctness, one warm-up per boundary precedes seven trials; A–E order rotates, with fresh OpenMP-8 each round. All times are host `steady_clock`, **not kernel/event times**. File I/O, Python/oracle checks and result comparisons are outside timing. B includes owner destruction; C/D/E reuse an existing owner. Fresh host output allocation is included whenever downloading. D still includes launches and blocking per-update flag copies, but no bulk output transfer.

| Boundary | GPU min / median / max (ms) | Iterative min / median / max (ms) |
| --- | ---: | ---: |
| A: normal one-shot, including free | 10.6628 / 12.8986 / 23.3603 | 1.4161 / 1.7202 / 2.4510 |
| B: construct + upload + fit + download + destroy | 9.4114 / 13.6317 / 21.4236 | 1.4971 / 1.6833 / 2.4619 |
| C: resident fit + download | 2.6481 / 2.7401 / 3.0719 | 1.2174 / 1.4412 / 2.1428 |
| D: resident fit only | 2.1274 / 2.1727 / 2.2530 | 1.1369 / 1.2344 / 2.2319 |
| E: upload + fit + download, device allocation excluded | 8.9548 / 11.1210 / 13.1689 | 1.4272 / 1.6278 / 2.4280 |
| Fresh OpenMP-8 | 18.0232 / 19.2699 / 24.7548 | 3.9760 / 4.4512 / 8.2650 |
| Construction alone | 0.2245 / 0.2492 / 0.3695 | 0.0152 / 0.0261 / 0.0335 |
| Destruction alone | 1.0453 / 1.4023 / 13.0145 | 0.0087 / 0.0134 / 0.0176 |

<details>
<summary>All seven native wall runs (ms), in acquisition order per boundary</summary>

| Boundary | GPU | Iterative |
| --- | --- | --- |
| A | 12.7692, 15.1127, 11.3463, 23.3603, 12.8986, 13.4324, 10.6628 | 1.7223, 1.4161, 1.7202, 2.4510, 1.5032, 1.6832, 1.8018 |
| B | 9.6459, 13.2047, 15.2893, 13.6317, 20.2805, 21.4236, 9.4114 | 1.6258, 2.1986, 1.6693, 2.4619, 1.4971, 1.6860, 1.6833 |
| C | 2.7401, 2.7199, 2.7389, 2.8239, 2.6481, 3.0719, 2.8075 | 1.2867, 1.5522, 1.4412, 2.1428, 1.2477, 1.2174, 1.4554 |
| D | 2.1771, 2.1926, 2.1565, 2.2530, 2.1274, 2.1577, 2.1727 | 1.2344, 1.1445, 1.2768, 2.2319, 1.2032, 1.1369, 1.3375 |
| E | 12.7816, 13.1689, 8.9548, 11.1210, 12.4589, 9.4634, 9.4787 | 1.6279, 1.4272, 1.6278, 2.4280, 1.5887, 1.6329, 1.5750 |
| OpenMP-8 | 18.0232, 19.2325, 19.1380, 21.3261, 19.2699, 19.9504, 24.7548 | 5.4250, 8.2650, 4.4512, 4.5813, 4.3992, 3.9760, 4.0638 |
| Construction | 0.2492, 0.3695, 0.2751, 0.3179, 0.2335, 0.2275, 0.2245 | 0.0335, 0.0312, 0.0261, 0.0268, 0.0236, 0.0152, 0.0163 |
| Destruction | 1.0453, 1.0954, 1.1884, 13.0145, 5.2538, 1.4023, 3.2302 | 0.0176, 0.0164, 0.0134, 0.0147, 0.0119, 0.0087, 0.0094 |

</details>

### Repeated independent fits

One warm sequence per R/boundary, seven paired sequences with alternating boundary order. Both exclude construction/destruction. Fit-only starts from uploaded input; upload-inclusive includes its validation/H2D once, R independent fits, and a final host allocation/download once. Correctness download for fit-only occurs after its timer. Median effective **ms/fit**:

| R | GPU fit only | GPU upload + R fits + download | Iterative fit only | Iterative upload + R fits + download |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.124500 | 9.303700 | 1.258200 | 1.590000 |
| 2 | 2.158400 | 5.906350 | 1.216000 | 1.371950 |
| 5 | 2.247260 | 3.630040 | 1.331580 | 1.452220 |
| 10 | 2.118330 | 2.808470 | 1.427880 | 1.444850 |
| 20 | 2.116790 | 2.452430 | 1.406415 | 1.424125 |

At R=20, total upload-inclusive median is 49.0486 ms GPU / 28.4825 ms iterative. Effective min/max are 2.439435/2.485595 ms GPU (CV 0.610%), 1.406830/1.462360 ms iterative (CV 1.163%). Fit-only effective min/max are 2.113960/2.118620 ms GPU (CV 0.074%), 1.382790/1.418440 ms iterative (CV 0.786%). The driver emits every raw sequence plus min/median/max/mean/CV; no outliers are removed.

The measured benefit is avoiding repeated validation/upload/allocation, not a changed algorithm. Fresh OpenMP-8 / CUDA median ratios for A, D and E are GPU **1.494x / 8.869x / 1.733x**, iterative **2.588x / 3.606x / 2.734x**. Initial preparation is excluded from D, so it is not a one-call speedup. GPU A/B CV is 27.8%/29.7%, while D is 1.67%; isolated free had a 13.0145 ms outlier. No claim that the new one-call owner is faster than one-shot is justified. Iterative overhead remains and effective times are nonmonotonic; these laptop samples do not establish universal ratios or a cause for driver/scheduling variability.

**Decision at the native-owner checkpoint:** expose the demonstrated workflow through correctness-first pybind11 one-shot and resident-owner integration, completed below. Linux validation, CUDA-fault injection, and broader crossover remain TBD.

## Python CUDA interface baseline

The opt-in [`kmeans_native` module](cpp/src/python_bindings.cpp) exposes the retained normal CUDA pipeline and exactly one native owner per Python `CudaKMeansBuffer`. Native kernels, one-shot behavior, and owner implementation are unchanged. The interface intentionally keeps the existing vector APIs and their explicit host copies; this is the correctness-first binding baseline.

### Build and import

Reuse the selected Project 1 environment: Python 3.12.7, NumPy 1.26.4, pybind11 3.1.0 in `hanlab`. No new package was installed. Python remains opt-in via `PROJECT2_ENABLE_PYTHON=ON` and requires `PROJECT2_ENABLE_CUDA=ON`; OpenMP is optional for the module, required only for the existing native comparison executable. CMake discovers pybind11 from the selected interpreter as in Project 1, and copies the matching CUDA runtime DLL beside the Windows extension. From the repository root in PowerShell:

```powershell
conda activate hanlab
cmake -S 03_CUDA_KMeans/cpp -B 03_CUDA_KMeans/cpp/build/python_cuda -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" -DPROJECT2_ENABLE_CUDA=ON -DPROJECT2_ENABLE_OPENMP=ON -DPROJECT2_ENABLE_PYTHON=ON -DPython_EXECUTABLE="$env:CONDA_PREFIX/python.exe"
cmake --build 03_CUDA_KMeans/cpp/build/python_cuda --config Release
ctest --test-dir 03_CUDA_KMeans/cpp/build/python_cuda -C Release --output-on-failure
python -B 03_CUDA_KMeans/python/validate_bindings.py --module-dir 03_CUDA_KMeans/cpp/build/python_cuda/Release --workloads
python -B 03_CUDA_KMeans/python/benchmark_bindings.py --module-dir 03_CUDA_KMeans/cpp/build/python_cuda/Release --native-executable 03_CUDA_KMeans/cpp/build/python_cuda/Release/phase2_kmeans_cuda_owner_benchmark.exe --mode all
```

This builds a local extension, not an installed wheel; add its Release directory to Python's import path. No CPU-only/OpenMP-only/CUDA-without-Python build requires pybind11 or Python discovery. Those three configurations still build and pass their native tests. The Python-enabled CUDA/OpenMP configuration passes all five CTest targets, including public Python contracts. Requesting Python without CUDA gives an explicit configure error. Linux compilation/runtime has not been validated. Extension binaries, DLLs, generated arrays, caches and logs are excluded from Git.

### API, state, and copies

```python
import sys
import numpy as np
sys.path.insert(0, "03_CUDA_KMeans/cpp/build/python_cuda/Release")
import kmeans_native as km

X = np.array([[-2., -1.]] * 8 + [[2., 1.]] * 8, dtype=np.float32)
result = km.kmeans_cuda(X, 2)
# result: {"labels": int32[N], "centroids": float32[K,D],
#          "update_count": int, "converged": bool}

buffer = km.CudaKMeansBuffer(X.shape[0], X.shape[1], 2)
buffer.upload(X)
for _ in range(3):
    metadata = buffer.fit()  # only update_count and converged; no bulk download
result = buffer.download()
buffer.upload(-X)            # replaces input and invalidates previous output
metadata = buffer.fit()
replacement = buffer.download()
```

Input must be a two-dimensional NumPy ndarray with exact native `float32` dtype, C-contiguous layout, `2 <= N <= 2^20`, `1 <= D <= 32`, `2 <= K <= min(N,32)`, finite values and `abs(x) <= 1024`. N/D/K arguments are Python integers, excluding bool. Upload additionally requires exactly the construction `(N,D)` shape. Invalid dtype/layout/shape is rejected, with no conversion, reshape or repair. Valid readonly or unaligned C-contiguous arrays work through the explicit host copy. Input stays unchanged. Labels are independently owned C-contiguous `int32 (N,)`; centroids are independently owned C-contiguous `float32 (K,D)`. Both live beyond temporary C++ results and later owner calls.

Every resident fit reseeds from the uploaded snapshot: three fits are three independent reruns, not chained Lloyd refinement. Upload returns None. Fit/download before their required state raise RuntimeError. Metadata/value validation failures raise TypeError/ValueError (integer overflow raises OverflowError) and preserve previous state. Successful upload invalidates output until fit completes. Repeated downloads remain valid and allocate independent arrays. Native CUDA failures raise contextual RuntimeError and require owner reconstruction, as in C++. Private underscore reduced-cap hooks exist solely for the frozen nonconvergence fixture; benchmark/profiler APIs are not exported.

| Boundary | Explicit copy path |
| --- | --- |
| One-shot | NumPy -> temporary `vector<float>` -> H2D -> GPU fit -> D2H -> result vectors -> new NumPy labels/centroids |
| Owner upload | NumPy -> temporary `vector<float>` -> native validation -> H2D; temporary vector then freed |
| Owner fit | Uploaded GPU input -> GPU algorithm; only existing per-update flag copies and final host metadata, no bulk result download |
| Owner download | GPU labels/centroids -> result vectors -> new NumPy arrays |

The binding checks only dtype/layout/shape/bounds before copying. Full finite/magnitude validation occurs **once in the native call**, never a second Python scan. Resident fits retain no host input and do not repeat validation or upload. NumPy metadata/copy/output creation holds the GIL; native construction, upload, fit, download, one-shot computation and cleanup release it. No Python objects are accessed inside released scopes. A nonblocking wrapper guard rejects overlapping operations on the same owner with a busy RuntimeError; the owner remains valid. Operations inherit the native requirement that the construction CUDA device be current, and the object must not outlive a CUDA context reset.

### Python correctness and GIL check

Both APIs pass all six public fixtures (96 labels per API, including the reduced-cap nonconvergence case), six existing FP32/feature edges, extra partial count/centroid tile cases, inclusive bounds, readonly/unaligned inputs, lifecycle errors, invalid input, snapshot replacement and independent output ownership. Primary `(65536,8,16)`, GPU `(262144,16,16)` and iterative `(16384,8,8)` also pass: labels, update counts, convergence and tested centroid bits are exact, with zero centroid/inertia error against NumPy. Frozen centroid/inertia tolerances remain the general contract. Each performance workload checks 20 fits with downloads, three fits without downloads, and X1 -> X2 -> X1 replacement; outputs are deterministic and input remains unchanged.

A bounded concurrency sanity test counted 289,402 Python worker increments while resident fit released the GIL, and immediately rejected the worker's overlapping same-owner download as busy. This check is separate from timing. Actual CUDA faults and alternate-device behavior were not injected/tested here.

### Python/native wall measurements

2026-10-05, AC power, same Windows laptop/RTX 4070 Laptop, driver 617.14, CUDA 12.9.86, MSVC 19.44 x64 Release `/O2`, `sm_89`, unchanged `--fmad=false`. One warm-up per boundary and seven trials; Python API order rotates. Seven paired sequences per R alternate fit-only and upload-inclusive order. All samples are retained. Python uses `perf_counter`; the subsequent same-session native suite uses `steady_clock` on the identical generated arrays. Inputs/oracles/file I/O are outside every timer. Construction excludes destruction; all owner operation/repeated boundaries use an existing allocation. One-shot includes its complete native ownership lifecycle. These are **wall times including host work**, not CUDA-event device algorithm times; prior event estimates are separate evidence.

| Python API boundary | GPU min / median / max (ms) | Iterative min / median / max (ms) |
| --- | ---: | ---: |
| One-shot | 14.0831 / 14.3966 / 15.0920 | 1.5708 / 1.6939 / 1.9101 |
| Construction | 0.1710 / 0.1934 / 0.2084 | 0.0144 / 0.0170 / 0.0238 |
| Upload | 9.8951 / 10.1646 / 10.6664 | 0.1854 / 0.1941 / 0.2383 |
| Resident fit only | 2.2345 / 2.2501 / 2.3405 | 1.1198 / 1.2778 / 1.4190 |
| Download | 0.6036 / 0.6383 / 0.7712 | 0.0323 / 0.0422 / 0.0544 |
| Upload + fit + download | 12.8571 / 12.9676 / 13.2254 | 1.5001 / 1.6634 / 1.7928 |

All seven Python one-shot runs (ms): GPU `14.3966, 14.4638, 14.2708, 14.0831, 14.1223, 15.0920, 14.6462`; iterative `1.6526, 1.5841, 1.6975, 1.7248, 1.5708, 1.6939, 1.9101`.

Fresh native one-shot runs (ms): GPU `18.8238, 9.8880, 13.7253, 13.7852, 10.2078, 22.9875, 10.1368` (min/median/max `9.8880/13.7253/22.9875`); iterative `5.0435, 1.5148, 1.6257, 1.7413, 1.5569, 1.7084, 1.7893` (`1.5148/1.7084/5.0435`). Native resident-fit medians are `2.1427/1.2797` ms; existing-owner upload/fit/download `8.8231/1.6165` ms; construction `0.1978/0.0283` ms. Python/native median ratios for one-shot, resident fit and upload/fit/download are respectively GPU `1.048910/1.050124/1.469733`, iterative `0.991513/0.998515/1.029013`. These separately sampled intervals do not isolate copy cost: native GPU one-shot/upload-inclusive CVs are 32.55%/41.08%, versus Python 2.23%/0.91%. The small below-one iterative ratios are within variation, not negative binding overhead.

Repeated Python effective medians, **ms/fit**, exclude construction/free. Fit-only excludes upload/download; upload-inclusive performs one validated upload, R independent fits and one final download:

| R | GPU fit only | GPU upload + fits + download | Iterative fit only | Iterative upload + fits + download |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.235500 | 13.155700 | 1.411700 | 1.547600 |
| 2 | 2.240450 | 7.779100 | 1.246250 | 1.513750 |
| 5 | 2.131340 | 4.460120 | 1.328200 | 1.498760 |
| 10 | 2.115610 | 3.275800 | 1.386740 | 1.599960 |
| 20 | 2.118530 | 2.704095 | 1.541815 | 1.721435 |

At R=20, total upload-inclusive medians are `54.0819/34.4287` ms. GPU effective min/max are `2.674775/2.782845` ms/fit (CV 1.199%), iterative `1.292300/2.043125` (CV 15.310%). The driver prints every raw stage/sequence, min/median/max/mean/CV, and fresh native records; logs/arrays remain local.

Python preserves the large-workload residency benefit: R20 upload-inclusive time is **5.324x** lower per fit than repeated Python one-shot calls in this session, and isolated resident fit is close to native. Copies/validation/transfer remain visible at upload/download and are documented for future experiments. Iterative Python resident fit is also close to native and about **1.326x** faster than Python one-shot, but the longer iterative sequences show drift/outliers and no reliable monotonic amortization claim. No copy optimization or additional kernel work was performed.

**Decision at the Python-interface checkpoint:** native CUDA engineering and correctness-first Python integration were ready for external-library comparison. The selected fair scikit-learn comparison is completed below. Historical interface timings above remain separate from that fresh comparison; broader crossover and GPU-library feasibility were not measured at this checkpoint.

## Controlled scikit-learn comparison

The [comparison driver](python/compare_sklearn.py) measures the retained implementation without tuning kernels, compiler flags, native APIs, or the frozen numerical contract. This is a bounded external-library comparison, not a claim to outperform scikit-learn generally.

### Configuration and semantic differences

The existing `hanlab` environment provides Python 3.12.7, NumPy 1.26.4, scikit-learn 1.4.2 and threadpoolctl 3.5.0; nothing was installed. The installed Python implementation and official version-tagged Cython sources were inspected, rather than inferring behavior from current-version defaults. Every scikit-learn call is `KMeans(n_clusters=K, init=initial, n_init=1, algorithm="lloyd", max_iter=100, tol=0.0, copy_x=True).fit(X)`. The explicit C-contiguous float32 initialization array is prepared before timing from rows `((2*k+1)*N)//(2*K)`, in the original cluster order. Both libraries receive the same unchanged float32 input, with no weights or restarts; scikit-learn returns float32 centers.

| Concern | Project 2 | Installed scikit-learn 1.4.2 dense Lloyd |
| --- | --- | --- |
| Initialization | Exact fixed input rows, one run | Same rows supplied explicitly, `n_init=1`; copies/centers them internally |
| Assignment | Ordered separate FP32 subtract/square/add | Centers input; computes `||C||² - 2XCᵀ` using FP32 BLAS, omitting the row-constant input norm |
| Ties | Lowest index for an exact computed-distance tie | First index for an exact computed-score tie in this implementation; different rounding can change which scores tie |
| Centroid update | Deterministic FP64 accumulation, division then FP32 rounding | FP32 thread-local sums, lock-protected reduction with no fixed arrival order, reciprocal/multiply averaging |
| Empty clusters | Retain old center | Relocate farthest assigned samples; unresolved empties use the largest cluster's center |
| Stopping | Initial assignment, then label-unchanged update/reassignment passes | Assignment/update iterations; unchanged labels or center shift at/below tolerance, final assignment if not strict convergence |
| Copies and centering | Explicit NumPy/vector copies and H2D/D2H for one-shot | Safe/default `copy_x=True`: input and initialization copied, input mean subtracted, mean restored to final centers |

`tol=0.0` is supported and makes the tolerance exactly zero; it does not disable a zero-center-shift stop or make the stopping/iteration definitions identical. Tie/reduction details above are implementation observations, not additional promises of the scikit-learn public API. Sources: [1.4.2 KMeans API](https://scikit-learn.org/1.4/modules/generated/sklearn.cluster.KMeans.html), [Python fit/stopping source](https://github.com/scikit-learn/scikit-learn/blob/1.4.2/sklearn/cluster/_kmeans.py), [dense Lloyd source](https://github.com/scikit-learn/scikit-learn/blob/1.4.2/sklearn/cluster/_k_means_lloyd.pyx), [empty-center/update helpers](https://github.com/scikit-learn/scikit-learn/blob/1.4.2/sklearn/cluster/_k_means_common.pyx).

### Thread controls and timing boundaries

On 2026-10-05, the same AC-powered Windows/i7-13700H/RTX 4070 Laptop session used the existing MSVC x64 Release `/O2`, CUDA 12.9, `sm_89`, and unchanged `--fmad=false` build. Sampling/phase instrumentation remained off. Before timing, nested `threadpoolctl` contexts set BLAS to one thread **first**, then OpenMP to eight; applying the BLAS setting loads Intel OpenMP before its limit is set. Runtime inspection confirmed MKL 2023.2 Intel BLAS=1, Microsoft `vcomp140` OpenMP=8 and Intel `libiomp5md` OpenMP=8. The effective scikit-learn helper and every fitted estimator reported eight threads. These are configured budgets, not measured core utilization. The native OpenMP path requests and checks an actual eight-thread assignment team; its available logical-processor count is 20. No default-thread sweep or `copy_x=False` shortcut was used. [Supported thread controls](https://scikit-learn.org/1.4/computing/parallelism.html).

The harness imports the CUDA extension before scikit-learn: the reverse order caused a native access violation before any timing in this environment. The successful order passed runtime checks and the complete comparison; the underlying DLL interaction remains unestablished. No package/system/native-code workaround was applied. Joblib's physical-core discovery also warns and falls back to logical cores; the explicitly verified eight-thread budget is unaffected.

One warm-up and seven trials per boundary use `perf_counter`, rotating scikit-learn fit, CUDA one-shot, resident fit, construction, upload and download. Initialization/data generation, oracle/quality calculations, thread-control setup and file I/O are outside all timers. Scikit-learn includes estimator construction and complete normal `.fit(X)` work; CUDA one-shot includes its normal host copies, validation, allocation, transfers, computation and result construction. Those are the primary user-facing comparison. Resident fit is secondary: allocation, validated upload and bulk download are excluded. Construction excludes destruction; upload/download include their normal copies. Native OpenMP uses the unchanged executable's `steady_clock` complete-fit boundary, including native validation/output allocation but excluding the Python/file bridge; its separate adjacent block is context, not an identical Python API boundary. All samples are retained. These are wall times, not CUDA-event device times, and independent stage medians must not be added as exact accounting.

Reproduce after building the Python module above:

```powershell
python -u -B 03_CUDA_KMeans/python/compare_sklearn.py --module-dir 03_CUDA_KMeans/cpp/build/python_cuda/Release --openmp-executable 03_CUDA_KMeans/cpp/build/python_cuda/Release/phase2_kmeans_openmp_scaling.exe --mode all --runs 7
```

### Output quality and iteration context

The unchanged version-1 generators use primary `(65536,8,16)` seed `20260924`, GPU `(262144,16,16)` seed `20260925`, and iterative `(16384,8,8)` seed `20260927`. CUDA results match the authoritative NumPy contract; native serial/OpenMP results match it too. Input and initialization remain unchanged. Scikit-learn is compared directly by the original cluster indices, with **no permutation** and no demand that the six semantic fixtures serve as its contract tests.

All seven scikit-learn trials have 100% exact label agreement: `65536/65536`, `262144/262144`, and `16384/16384`. Every fit reports the same iteration count within its workload. The table reports the largest centroid/inertia differences observed across the seven fits, not a favorable single trial. Feature scale is `S_j=max(1,max_i|X[i,j]|)`; both inertias are independently recomputed by the same bounded FP64 diagnostic, not taken from `sklearn.inertia_`.

| Workload | sklearn iterations / Project 2 updates | Max centroid abs / feature-scaled difference | Max inertia absolute / relative difference |
| --- | ---: | ---: | ---: |
| Primary | 2 / 1 | 2.861023e-6 / 3.175974e-7 | 7.235940e-7 / 2.204219e-11 |
| GPU | 2 / 1 | 5.722046e-6 / 6.297798e-7 | 1.174345e-5 / 4.476742e-11 |
| Iterative | 23 / 22 | 9.536743e-7 / 8.446737e-8 | 7.101335e-9 / 1.419759e-14 |

Recomputed Project 2 / final-trial scikit-learn inertia is respectively `32827.68637102876 / 32827.68637175235`, `262321.4204471333 / 262321.42045883933`, and `500178.7404378790 / 500178.7404378834`. Quality is effectively identical on these inputs, although centroid bits differ and small scikit-learn reduction variation is observed. This does not establish parity on arbitrary ties, empty clusters or sensitive inputs.

Secondary work-count context avoids misleading division by reported iterations: Project 2 performs `2/2/23` assignment passes and `1/1/22` centroid updates; scikit-learn's recorded `2/2/23` Lloyd iterations each perform assignment and update, hence one additional centroid update. Scikit-learn can also perform a final assignment after non-strict stopping; the harness did not instrument its stopping branch, so total assignment-pass parity is not asserted. Whole-fit timings still include different arithmetic, validation, centering and result work; neither reported iteration count is a universally equivalent work unit. No normalized per-iteration speedup is claimed.

### Fresh measurements

All values below are milliseconds, in acquisition order. Complete raw timings, min/median/max and population CV are preserved; no outlier was removed.

| Workload / boundary | Seven raw calls | Min / median / max | CV |
| --- | --- | ---: | ---: |
| Primary sklearn fit | 7.0074, 6.9141, 6.6513, 7.7057, 7.7132, 9.5203, 7.4589 | 6.6513 / 7.4589 / 9.5203 | 11.66% |
| Primary CUDA one-shot | 3.4968, 2.1899, 2.6266, 2.2576, 2.6629, 2.3881, 2.5010 | 2.1899 / 2.5010 / 3.4968 | 15.64% |
| Primary resident fit | 0.3045, 0.2844, 0.5136, 0.2827, 0.3179, 0.3455, 0.3154 | 0.2827 / 0.3154 / 0.5136 | 22.06% |
| Primary construction | 0.1121, 0.0888, 0.1410, 0.0959, 0.1536, 0.0891, 0.1552 | 0.0888 / 0.1121 / 0.1552 | 23.22% |
| Primary upload | 2.2323, 1.2631, 1.2123, 2.0312, 1.2096, 1.9277, 2.0382 | 1.2096 / 1.9277 / 2.2323 | 24.61% |
| Primary download | 0.0828, 0.1221, 0.0868, 0.1185, 0.0918, 0.1739, 0.1297 | 0.0828 / 0.1185 / 0.1739 | 25.69% |
| Primary native OpenMP-8 | 2.8487, 2.7153, 2.9035, 2.7578, 2.8395, 2.6910, 3.0066 | 2.6910 / 2.8395 / 3.0066 | 3.65% |
| GPU sklearn fit | 37.1723, 32.4224, 32.6734, 30.4826, 33.9599, 30.2469, 40.8374 | 30.2469 / 32.6734 / 40.8374 | 10.41% |
| GPU CUDA one-shot | 14.9099, 14.2033, 14.4551, 15.2326, 15.0863, 15.9705, 19.3422 | 14.2033 / 15.0863 / 19.3422 | 10.36% |
| GPU resident fit | 2.2881, 2.2579, 2.3495, 2.2947, 2.3450, 2.2695, 2.2956 | 2.2579 / 2.2947 / 2.3495 | 1.41% |
| GPU construction | 0.2952, 0.1757, 0.2276, 0.1824, 0.3017, 0.1909, 0.1868 | 0.1757 / 0.1909 / 0.3017 | 22.53% |
| GPU upload | 13.0842, 10.0244, 10.9442, 10.1269, 9.7126, 12.5207, 17.3518 | 9.7126 / 10.9442 / 17.3518 | 20.91% |
| GPU download | 0.6075, 0.6577, 0.6029, 0.6158, 0.6031, 0.6661, 0.8085 | 0.6029 / 0.6158 / 0.8085 | 10.51% |
| GPU native OpenMP-8 | 18.2447, 28.2974, 18.5307, 19.2247, 19.2075, 16.7423, 22.6698 | 16.7423 / 19.2075 / 28.2974 | 17.74% |
| Iterative sklearn fit | 7.6235, 8.6097, 8.5941, 8.1235, 8.0971, 9.0652, 6.8694 | 6.8694 / 8.1235 / 9.0652 | 8.26% |
| Iterative CUDA one-shot | 1.7556, 1.9897, 1.8467, 2.6547, 3.6032, 1.8790, 1.9689 | 1.7556 / 1.9689 / 3.6032 | 27.62% |
| Iterative resident fit | 1.3696, 1.4351, 1.5374, 1.8562, 3.0521, 1.3394, 1.7897 | 1.3394 / 1.5374 / 3.0521 | 31.44% |
| Iterative construction | 0.2356, 0.0213, 0.0275, 0.0209, 0.0326, 0.0256, 0.0236 | 0.0209 / 0.0256 / 0.2356 | 133.27% |
| Iterative upload | 0.2402, 0.2299, 0.1922, 0.1923, 0.4243, 0.2180, 0.1945 | 0.1922 / 0.2180 / 0.4243 | 31.74% |
| Iterative download | 0.0889, 0.0414, 0.0326, 0.0591, 0.0469, 0.0800, 0.0439 | 0.0326 / 0.0469 / 0.0889 | 34.73% |
| Iterative native OpenMP-8 | 5.2401, 4.9484, 5.1781, 4.7541, 4.8405, 4.8262, 5.5600 | 4.7541 / 4.9484 / 5.5600 | 5.32% |

| Workload | sklearn complete Python fit | Project 2 native OpenMP-8 | Project 2 Python CUDA one-shot | Project 2 persistent resident fit |
| --- | ---: | ---: | ---: | ---: |
| Primary | 7.4589 ms | 2.8395 | 2.5010 | 0.3154 |
| GPU | 32.6734 ms | 19.2075 | 15.0863 | 2.2947 |
| Iterative | 8.1235 ms | 4.9484 | 1.9689 | 1.5374 |

Using fresh medians, Project 2 CUDA one-shot is **2.982x / 2.166x / 4.126x faster** than controlled scikit-learn on primary/GPU/iterative respectively. Native OpenMP-8 is **2.627x / 1.701x / 1.642x faster**, with the native/Python boundary caveat above. Resident fit is **23.649x / 14.239x / 5.284x faster as a prepared device-resident workflow**, not an equivalent one-shot comparison. The GPU resident median has 1.41% CV; iterative CUDA one-shot/resident CVs are 27.62%/31.44%, so absolute iterative results are less stable. No causally precise overhead breakdown or universal winner is claimed.

The existing residency study remains separate historical evidence; its R20 numbers are not substituted for these fresh timings. This comparison confirms comparable quality and a practical one-shot benefit on the three tested datasets. It does not locate a broad N/D/K crossover, compare default/best scikit-learn threads, or establish general-library superiority.

**Closeout assessment and one next milestone:** native CUDA engineering, Python integration and this controlled CPU-library comparison are complete on tested Windows. Recommend final Project 2 closeout and transition to Project 3 matrix/tensor multiplication, which adds more portfolio breadth than further K-means micro-tuning. Broader scaling is an explicitly deferred optional study, not falsely marked measured. `cuml`/`cuvs` are absent locally; [RAPIDS' Windows route requires WSL2/Linux](https://docs.nvidia.com/datascience/install/#windows-wsl2), so another GPU comparator would entail a separate environment and portability effort. Linux, fault injection and the import-order interaction remain disclosed limitations. No further comparison, optimization or Project 3 implementation was begun here.

## Planned workflow

Next: final Project 2 closeout and transition to Project 3 matrix/tensor multiplication. Broader scaling and GPU-library feasibility are optional deferred studies, not completed claims or prerequisites for that transition.

## Primary learning goals

- Performance structure of a standard iterative machine-learning algorithm.
- Data layout for samples and cluster representatives.
- Parallel assignment, reductions, synchronization, and update strategies.
- Scaling across independent workload dimensions.
- Transfer amortization, CPU/GPU crossover, and convergence-aware validation.

## Planned performance questions

- Which phase dominates at different sample, feature, and cluster counts?
- How does layout affect distance calculation on CPU and GPU?
- What limits assignment throughput and cluster-update throughput?
- How should reductions and synchronization be structured and measured?
- How much does host/device transfer contribute to end-to-end runtime?
- When does the GPU become beneficial, and how does iteration count affect that crossover?
- How do implementation choices affect convergence and reproducibility?

## Validation strategy

The deterministic seed order removes arbitrary label permutations from our own implementations. Every public fixture requires exact labels, update counts, convergence flags, shape/dtype/layout, and input immutability. For feature `j`, centroid absolute error must be at most `5e-6 * max(1,max_i |X[i,j]|)`; report both maximum absolute and maximum feature-scaled error. Independently recomputed inertia must differ by at most `2e-5 * max(1,|reference inertia|)`; the exactly representable zero-inertia case requires exact zero. [Reusable comparison helpers](python/reference.py) enforce these rules. Near-boundary label differences fail rather than being excused by centroid tolerance. External libraries may have different stopping and empty-cluster semantics, which must be disclosed rather than treated as exact equivalents.

## Benchmarking strategy

The initial and fresh profiling-size whole-fit timings are reported above; broad independent `N`, `D`, `K` scaling has not begun. The [version-1 seeded generator](python/benchmark_data.py) defines these approved workloads:

| Name | `(N,D,K)` | PCG64 seed |
| --- | --- | ---: |
| `profiling` | `(65,536,8,16)` | `20260924` |
| `gpu` | `(262,144,16,16)` | `20260925` |
| `stress_optional` | `(1,048,576,32,32)` | `20260926` |

The generator uses `numpy.random.Generator(numpy.random.PCG64(seed))`, shuffles cyclic planted labels, and adds `N(0,0.25)` FP64 noise to cluster-index-bit prototypes with coordinates `-8` or `+8`. It draws noise in ascending chunks of 16,384 rows, casts the prototype-plus-noise result once to `float32`, clips to the input bound, then overwrites each deterministic initialization row with its exact prototype. Planted labels are generation diagnostics, not the fit correctness oracle. No generated arrays are committed; reproduce them from code, version, seed, and NumPy version above. Future timings will distinguish initialization, transfers, per-phase execution, and end-to-end fit cost; benchmark-only fixed-pass comparisons must be identified separately from normal convergence.

## Status

Project 2 implementation and its controlled scikit-learn comparison are complete for the tested Windows scope. The two-feature distance-pipeline experiment was rejected. Final closeout is next; broader independent scaling, GPU-library comparison and Linux validation are explicitly deferred rather than reported complete. This directory retains its original numeric prefix until a separate repository reorganization.

## Open questions / TBD

- **TBD:** Explain the historical-versus-fresh baseline timing difference; broader independent size/stage scaling remains future work.
- **TBD:** Iterative synchronization's avoidable cost, Linux validation, and broader CPU/GPU crossover; later CUDA layout/fusion decisions remain evidence-dependent.
- **TBD:** GPU-library comparison and CUDA failure-injection coverage; host-copy removal is a later option, not part of this binding baseline. The observed Windows import-order failure remains unexplained; the comparison uses the documented successful order.
