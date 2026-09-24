# Project 2 — General-Purpose CUDA K-Means

## Objective

Develop, profile, validate, and optimize a general-purpose K-means implementation across Python/reference, C++, CPU-optimized, and CUDA stages.

## Why this project exists

K-means demonstrates that the portfolio's performance-engineering skills generalize beyond microscopy to a broadly applicable machine-learning algorithm. Its assignment and cluster-update phases offer different computation, reduction, synchronization, and memory challenges.

## Problem definition

The [portfolio roadmap](../ROADMAP.md) now defines the first deterministic Lloyd contract: dense finite row-major `float32` samples, reproducible sample-row initialization, squared-Euclidean assignment with lowest-index ties, mean updates with empty-centroid retention, assignment-stability convergence, a 100-update cap, and separate numerical validation rules. The reference and datasets have not been created yet.

## Planned workflow

Establish and profile a trusted reference; implement clear modern C++; validate convergence and results; benchmark, profile, and optimize CPU assignment and update phases; implement and validate CUDA; optimize GPU execution and data movement; add an appropriate Python interface; and analyze scaling across samples, dimensions, and clusters.

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

The initial fixed-order seed removes arbitrary cluster-label permutations from our own implementations. Public fixtures will require exact labels, update counts, and convergence behavior; centroids and independently recomputed inertia use the preregistered tight tolerances in the roadmap. External libraries may have different stopping and empty-cluster semantics, which must be disclosed rather than treated as exact equivalents.

## Benchmarking strategy

The reference, straightforward C++, optimized CPU, initial CUDA, and optimized CUDA implementations will be compared across multiple values of sample count, dimensionality, cluster count, and iteration behavior. Timings will distinguish initialization, transfers, per-phase execution, and end-to-end cost when relevant.

## Status

Next active project; contract defined, implementation not started. This directory retains its original numeric prefix until a separate repository reorganization.

## Open questions / TBD

- **TBD:** Authoritative NumPy implementation and checked-in tiny public fixtures.
- **TBD:** Public seeded benchmark generator and measured size/scaling results.
- **TBD:** Profile-driven CUDA mapping, reduction, layout, and fusion decisions.
- **TBD:** Availability and fair configuration of external libraries; binding approach.
