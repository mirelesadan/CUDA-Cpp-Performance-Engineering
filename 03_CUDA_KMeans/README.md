# Project 3 — General-Purpose CUDA K-Means

## Objective

Develop, profile, validate, and optimize a general-purpose K-means implementation across Python/reference, C++, CPU-optimized, and CUDA stages.

## Why this project exists

K-means demonstrates that the portfolio's performance-engineering skills generalize beyond microscopy to a broadly applicable machine-learning algorithm. Its assignment and cluster-update phases offer different computation, reduction, synchronization, and memory challenges.

## Problem definition

Given generic samples with a later-defined dimensionality and a chosen number of clusters, iteratively assign samples to clusters and update cluster representatives until a later-defined stopping condition is met. The project will use generic synthetic and/or standard non-microscopy datasets. Dataset choice, initialization, distance metric, precision, empty-cluster behavior, convergence criteria, and reproducibility rules are **TBD**.

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

Correctness and convergence criteria will be defined before optimization. Because equivalent K-means solutions may differ in label ordering or local optimum, validation will use appropriately defined objective, assignment, centroid, and convergence comparisons rather than assuming byte-for-byte identity without justification.

## Benchmarking strategy

The reference, straightforward C++, optimized CPU, initial CUDA, and optimized CUDA implementations will be compared across multiple values of sample count, dimensionality, cluster count, and iteration behavior. Timings will distinguish initialization, transfers, per-phase execution, and end-to-end cost when relevant.

## Status

Planned

## Open questions / TBD

- **TBD:** Reference algorithm and implementation.
- **TBD:** Generic synthetic and/or standard non-microscopy datasets.
- **TBD:** Initialization and random-seed policy.
- **TBD:** Distance metric, precision, convergence rule, and maximum iterations.
- **TBD:** Empty-cluster behavior and deterministic/reproducible modes.
- **TBD:** Target problem-size ranges and correctness metrics.
- **TBD:** Hardware, toolchain, profiling tools, external comparisons, and binding approach.
