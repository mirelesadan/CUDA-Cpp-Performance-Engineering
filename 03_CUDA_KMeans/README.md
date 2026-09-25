# Project 2 — General-Purpose CUDA K-Means

## Objective

Develop, profile, validate, and optimize a general-purpose K-means implementation across Python/reference, C++, CPU-optimized, and CUDA stages. The authoritative Python contract, serial C++ baseline, CPU profile, retained addressing optimization, and assignment-only OpenMP scaling are complete; CUDA is next.

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

## Planned workflow

Next: implement a correctness-first CUDA K-means baseline. Python integration and library comparisons come after the native behavior is trustworthy.

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

Active Project 2. Python reference, deterministic fixtures, correctness-first serial C++ baseline, native profiling, retained serial address-generation optimization, and assignment-only portable OpenMP scaling are complete. The two-feature distance-pipeline experiment was rejected; correctness-first CUDA is next. This directory retains its original numeric prefix until a separate repository reorganization.

## Open questions / TBD

- **TBD:** Explain the historical-versus-fresh baseline timing difference; broader independent size/stage scaling remains future work.
- **TBD:** Profile-driven CUDA mapping, reduction, layout, and fusion decisions.
- **TBD:** Availability and fair configuration of external libraries; binding approach.
