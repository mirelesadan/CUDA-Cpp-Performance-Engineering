# Project 4 — CUDA Matrix / Tensor Multiplication

## Objective

Understand and optimize a fundamental AI/HPC computational primitive by progressing from clear CPU and CUDA baselines toward increasingly efficient dense matrix multiplication and/or a closely related tensor multiplication.

## Why this project exists

Dense multiplication underlies many modern AI and deep-learning workloads. This project connects lower-level CUDA decisions to the performance of high-level frameworks and demonstrates an ability to measure progress against mature NVIDIA libraries without treating an unrealistic library victory as the goal.

## Problem definition

The initial operation will be a later-defined dense matrix multiplication and/or closely related tensor multiplication with specified shapes, layouts, transposition rules, data types, accumulation behavior, and numerical requirements. Those choices are **TBD**. The project is about understanding a core computational building block, not inventing a new neural network.

## Planned workflow

Establish trusted library/reference results; implement and validate a clear CPU baseline and an initial CUDA kernel; profile both; apply measured CPU and GPU optimizations in understandable stages; validate numerical behavior throughout; compare scaling with NVIDIA cuBLAS and, where useful, high-level framework calls; then document both improvements and remaining gaps.

## Primary learning goals

- GEMM/tensor shapes, layouts, indexing, computation count, and data reuse.
- Naive versus optimized CPU execution.
- CUDA thread/block mapping and coalesced global-memory access.
- Shared-memory tiling, synchronization, occupancy, registers, and arithmetic intensity.
- Numerical validation and fair comparison with optimized libraries.
- How Python AI frameworks rely on lower-level C++/CUDA libraries and kernels.

## Planned performance questions

- Where are the limits of the naive CPU and CUDA implementations?
- How do matrix dimensions, aspect ratios, layouts, and data types affect performance?
- Which access patterns are coalesced, and how much reuse does tiling create?
- Is each kernel limited by bandwidth, computation, launch overhead, or resources?
- How does custom-kernel performance evolve at each measured optimization stage?
- What fraction of relevant cuBLAS performance is reached under a fair comparison?
- What profiling evidence explains the remaining gap?

## Validation strategy

Correctness and numerical tolerances will be defined before optimization for each supported data type and operation. Results will be compared with trusted references across representative and edge-case dimensions, with accumulation behavior and tolerance documented.

## Benchmarking strategy

Comparisons may include naive CPU, optimized CPU, naive CUDA, progressively optimized CUDA, and cuBLAS, with NumPy, PyTorch, or TensorFlow added only when useful and methodologically comparable. Multiple matrix dimensions will be tested, setup and transfer costs will be disclosed, and kernel-only versus end-to-end timings will be distinguished.

Beating cuBLAS is not the objective. Success means explaining measured improvement from the custom baseline, quantifying the fraction of relevant mature-library performance achieved, and understanding the remaining difference.

## Status

Planned

## Open questions / TBD

- **TBD:** Initial matrix versus tensor operation scope.
- **TBD:** Shapes, aspect ratios, layouts, transposition options, and batching.
- **TBD:** Data types, accumulation types, and numerical tolerances.
- **TBD:** CPU baseline and external reference libraries.
- **TBD:** Sequence of CUDA optimization experiments.
- **TBD:** Fair timing and synchronization methodology for cuBLAS/framework comparisons.
- **TBD:** Hardware, toolchain, profiling tools, and Python integration scope.
