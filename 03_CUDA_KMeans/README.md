# Project 2 — General-Purpose CUDA K-Means

## Objective

Develop, profile, validate, and optimize a general-purpose K-means implementation across Python/reference, C++, CPU-optimized, and CUDA stages. The authoritative Python contract and straightforward serial C++ baseline are complete; optimization has not begun.

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

From the repository root, configure with `cmake -S 03_CUDA_KMeans/cpp -B 03_CUDA_KMeans/cpp/build -G "Visual Studio 17 2022" -A x64`, then build with `cmake --build 03_CUDA_KMeans/cpp/build --config Release`. Run the native contract test with `ctest --test-dir 03_CUDA_KMeans/cpp/build -C Release --output-on-failure`. Run the [fixture driver](python/test_cpp_serial.py) with `python -B 03_CUDA_KMeans/python/test_cpp_serial.py 03_CUDA_KMeans/cpp/build/Release/phase2_kmeans_serial.exe`; add `--benchmark` for the controlled baseline timing. Temporary raw-FP32 files bridge tests to the executable; this is not a Python binding. Project 2's CMake build has no Project 1, OpenMP, CUDA, or Python dependency.

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

Both one-update and 22-update fits are assignment-dominated. The update phase becomes more relevant with iteration count and will matter for future parallel reductions, but it is not the primary serial bottleneck here. Ranked CPU candidates: (1) reduce repeated assignment-loop row/centroid address calculation without changing traversal or FP32 arithmetic (the combined load/address line and loop lines account for substantial samples; low numerical risk); (2) investigate the FP32 distance-accumulation dependency (the largest single source line, but higher exact-semantics risk); (3) simplify cluster-minimum bookkeeping (9–12% selection attribution, smaller expected ceiling). The **next isolated experiment** is candidate 1, with a fresh same-session baseline and exact output comparison; none of these changes is implemented here.

Reproduce the whole-fit checks with `python -B 03_CUDA_KMeans/python/profile_workloads.py 03_CUDA_KMeans/cpp/build/Release/phase2_kmeans_serial.exe profiling` (or `gpu`, `stress_optional`, `iterative_profile`). For phase timing and sampling, configure a separate ignored build with `-DPROJECT2_ENABLE_PHASE_TIMING=ON -DPROJECT2_ENABLE_PROFILING_SYMBOLS=ON`; run the corresponding `phase2_kmeans_serial_phase_timing.exe` through that driver, or `phase2_kmeans_serial_sampler.exe` with `--sample-repeats`. Both options default to `OFF`; normal Release and public API remain uninstrumented. No generated arrays or profiler reports are versioned.

## Planned workflow

Next: test one isolated assignment-addressing change against the profiled straightforward serial baseline, then remeasure before later OpenMP and correctness-first CUDA. Python integration and library comparisons come after the native behavior is trustworthy.

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

Active Project 2. Python reference, deterministic fixtures, correctness-first serial C++ baseline, and first native profiling checkpoint complete; CPU optimization has not begun. This directory retains its original numeric prefix until a separate repository reorganization.

## Open questions / TBD

- **TBD:** Explain the historical-versus-fresh baseline timing difference; broader independent size/stage scaling remains future work.
- **TBD:** Profile-driven CUDA mapping, reduction, layout, and fusion decisions.
- **TBD:** Availability and fair configuration of external libraries; binding approach.
