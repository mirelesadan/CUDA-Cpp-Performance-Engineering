# Project 2 — General-Purpose CUDA K-Means

## Objective

Develop, profile, validate, and optimize a general-purpose K-means implementation across Python/reference, C++, CPU-optimized, and CUDA stages. This milestone freezes the Python semantics and public fixtures only; it does not benchmark or implement native code.

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

## Planned workflow

The reference and correctness cases are complete. Next: implement a clear serial C++ baseline, validate against these cases, then benchmark/profile CPU phases, add OpenMP and correctness-first CUDA, and let profiling drive GPU optimizations. Python integration and library comparisons come after the native behavior is trustworthy.

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

No timings are claimed yet. The [version-1 seeded generator](python/benchmark_data.py) defines these future workloads:

| Name | `(N,D,K)` | PCG64 seed |
| --- | --- | ---: |
| `profiling` | `(65,536,8,16)` | `20260924` |
| `gpu` | `(262,144,16,16)` | `20260925` |
| `stress_optional` | `(1,048,576,32,32)` | `20260926` |

The generator uses `numpy.random.Generator(numpy.random.PCG64(seed))`, shuffles cyclic planted labels, and adds `N(0,0.25)` FP64 noise to cluster-index-bit prototypes with coordinates `-8` or `+8`. It draws noise in ascending chunks of 16,384 rows, casts the prototype-plus-noise result once to `float32`, clips to the input bound, then overwrites each deterministic initialization row with its exact prototype. Planted labels are generation diagnostics, not the fit correctness oracle. No generated arrays are committed; reproduce them from code, version, seed, and NumPy version above. Future timings will distinguish initialization, transfers, per-phase execution, and end-to-end fit cost; benchmark-only fixed-pass comparisons must be identified separately from normal convergence.

## Status

Next active project. Authoritative Python reference and deterministic correctness fixtures complete; native implementation and benchmarking have not begun. This directory retains its original numeric prefix until a separate repository reorganization.

## Open questions / TBD

- **TBD:** Measured reference/native size and stage-scaling results; no benchmark campaign has begun.
- **TBD:** Profile-driven CUDA mapping, reduction, layout, and fusion decisions.
- **TBD:** Availability and fair configuration of external libraries; binding approach.
