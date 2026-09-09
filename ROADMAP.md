# Portfolio Roadmap

## Portfolio objective

The objective of this portfolio is to build credible, demonstrable competence in modern C++, performance-oriented C++, CPU optimization, profiling, benchmarking, CUDA C++, GPU programming, GPU memory and performance reasoning, Python interoperability, and high-performance scientific and AI computing.

Each project should become an evidence-backed engineering narrative: begin with a correct reference, locate measured bottlenecks, implement clear lower-level versions, optimize deliberately, validate throughout, and explain the resulting performance. The work must remain understandable and modifiable by the author rather than becoming a collection of opaque AI-generated kernels.

## Developer Technology and NVIDIA motivation

The portfolio is designed to strengthen preparation for NVIDIA Developer Technology and AI-oriented engineering roles. It should demonstrate the ability to understand real workloads, reason about CPU and GPU execution, use profiling evidence, communicate tradeoffs, and help connect high-level scientific or AI applications to efficient C++ and CUDA implementations.

The intended interview-level outcome is the ability to explain code structure, memory-management choices, thread/block/grid organization, performance bottlenecks, profiling results, optimization tradeoffs, and why a measured change helped or hurt.

## Starting skill set

The author is an Applied Physics PhD candidate with substantial experience in scientific computing and computational electron microscopy. Current strengths include Python, Jupyter, NumPy, SciPy, scikit-image, OpenCV, multidimensional arrays, large microscopy datasets, algorithm development, analysis, visualization, and research software pipelines.

The main gaps this portfolio addresses are C and modern C++ experience, low-level memory concepts, CPU optimization, CUDA kernel development, and GPU performance profiling. Assembly and generated instructions may be studied when they answer a performance question, but assembly programming is not a primary objective.

## Skills to develop

- Modern C++ language, library, ownership, and build concepts.
- Correct benchmarking and CPU profiling.
- Memory layout, allocation, copying, cache locality, vectorization, and multithreading.
- CUDA execution and memory models, kernel development, and correctness validation.
- GPU profiling, transfer analysis, coalescing, occupancy, divergence, tiling, and scaling.
- Python interfaces to efficient C++ and CUDA implementations with deliberate ownership and copy behavior.
- Clear technical communication through reproducible experiments and repository-quality documentation.

## Engineering philosophy

> Correctness first. Then measurement. Then optimization. Then measurement again.

Optimization hypotheses are useful only when tested. A fast result that is incorrect, irreproducible, measured unfairly, or unexplained is not a successful outcome. The portfolio should preserve simple reference versions because they provide correctness or performance baselines and make the engineering progression visible.

## Repeated performance-engineering workflow

### Stage 1 — Establish a reference implementation

Begin from a correct Python or other trusted reference implementation. Establish known inputs and expected outputs. For research algorithms, first understand the existing implementation without silently changing its scientific behavior.

### Stage 2 — Profile the reference

Measure baseline runtime over relevant problem sizes and determine where time is actually spent. Separate measured bottlenecks from assumptions.

### Stage 3 — Implement straightforward modern C++

Reimplement the selected computation with clarity and correctness as the priorities. Document data representation, ownership, and interfaces without prematurely optimizing.

### Stage 4 — Validate, benchmark, and profile C++

Compare outputs with the reference, measure runtime correctly, and investigate allocation, copies, memory access, cache behavior, and computational structure.

### Stage 5 — Optimize the CPU implementation

Consider compiler optimization, algorithmic improvements, fewer allocations and copies, contiguous access, cache locality, data layout, multithreading, and SIMD/vectorization where evidence supports them. Measure every meaningful change.

### Stage 6 — Implement initial CUDA

Move appropriate measured hotspots to CUDA C++. Begin with a clear, correct GPU implementation and validate it against trusted CPU/reference results.

### Stage 7 — Profile and optimize CUDA

Investigate transfer costs, kernel-launch overhead, thread/block/grid mapping, warps, occupancy, global/shared memory, registers, synchronization, coalescing, divergence, arithmetic intensity, bandwidth, and problem-size scaling using appropriate NVIDIA tools. Optimize in response to evidence.

### Stage 8 — Integrate with Python

Expose the high-performance implementation through a clean Python interface so Python or Jupyter users can access CPU or GPU acceleration without writing CUDA. Binding technology, including whether pybind11 is appropriate, remains TBD until project requirements are understood.

### Stage 9 — Perform final validation and benchmarking

Compare relevant reference Python, straightforward C++, optimized CPU, initial CUDA, optimized CUDA, and mature external-library implementations across multiple problem sizes. Where useful, identify the CPU/GPU crossover point and distinguish kernel-only from end-to-end performance.

### Stage 10 — Document the engineering story

Explain the problem, algorithm, baseline, profiling, implementation progression, validation, CPU and GPU optimizations, benchmark method, results, scaling, limitations, and lessons learned. Claims should follow directly from reproducible measurements.

## Project 1 — 4D-STEM Median Filter Acceleration

### Purpose

Use two related median-filter phases to learn the complete workflow and then apply it to a measured research-software bottleneck. Phase A is deliberately controlled and conceptually approachable; Phase B supplies the more consequential performance-engineering problem.

### Established scientific preparation and performance boundary

The 4Denoise storage order is `(scan_y, scan_x, detector_y, detector_x)`. Scientific preparation center-aligns every diffraction pattern and corrects global detector ellipticity. Alignment and ellipticity correction generate the common preprocessed input but remain outside Project 1's C++/CUDA acceleration scope. Both median phases operate in real space over axes 0 and 1, keeping detector coordinates independent.

The replaceable practical source is a local, non-versioned `ripple_data_reduced.npy`. Its explicit axis contract is `(scan_y, scan_x, detector_y, detector_x)`, but its numerical dimensions and dtype are inspected at runtime rather than treated as project invariants. Scientific preparation saves the current unfiltered result locally as `01_4DSTEM_Median_Filter_Acceleration/benchmark_data/median_filter_input.npy`. Experimental source, benchmark, and historical fixture arrays are excluded from the public repository because redistribution rights are not established; a small deterministic synthetic fixture is committed for runnable Phase A validation.

If only scan extents change, the detector calibration may remain compatible. If detector pixels are cropped, shifted, or resampled, the established ellipse-fit annulus must be validated or scientifically re-established. The NPY shape alone is not enough to infer detector calibration.

### Phase A — fixed `3 × 3` median warm-up

Phase A applies the public 4Denoise `HyperData.denoise(...)` path for a centered `3 × 3` median over axes 0 and 1 with `reflect` boundaries. It is retained to establish the C++ build, multidimensional indexing, validation, benchmarking, basic CPU optimization, and first CUDA-kernel workflow. The established notebook and local `(16, 16, 16, 16)` experimental fixture support the recorded finite-`float64`, bit-for-bit evidence; the public default fixture is deterministic and synthetic.

The first canonical-input x64 Release baseline was recorded on 2026-09-08 with one warm-up and three timed calls per path. The straightforward C++ median was 3756.627 ms at the median, while the complete public 4Denoise wrapper call was 4112.253 ms at the median, a `4Denoise/C++` time ratio of `1.095×`. Loading was excluded, output allocation was included, and no profiling or algorithm optimization was performed.

The initial optimized-Release CPU profile was completed on 2026-09-08 without changing the algorithm. Of 17,996 in-filter instruction-pointer samples, the `std::nth_element` call site accounted for 62.45%, indexing/reflection for 18.76%, neighborhood gathering for 13.59%, and remaining loop/output-store work for 5.20%; allocation and pre-loop validation were below sampling resolution. This evidence makes fixed median-of-nine selection and repeated address-generation work the first CPU optimization targets.

The first isolated serial optimization was completed on 2026-09-08. A fixed 19-comparator median-of-nine network replaced only `std::nth_element` in a separate candidate while preserving the baseline. The candidate had zero bitwise mismatches across both the 65,536-element fixture and all 47,228,125 canonical outputs. Fresh median times were 3515.558 ms for the baseline and 3096.183 ms for the candidate, a `1.135449×` speedup and 11.93% runtime reduction.

The second isolated serial optimization was completed on 2026-09-08 with the median-of-nine version as its fresh baseline. Reusable C-order strides, detector offsets, and scan-plane bases replaced checked `flat_index(...)` calls only inside a separate candidate; reflection, loop order, neighborhood semantics, median selection, and allocation were unchanged. The candidate again had zero bitwise mismatches across the fixture and canonical comparison. Fresh median times were 3098.906 ms for the current baseline and 2803.897 ms for the candidate, a `1.105214×` speedup and 9.52% runtime reduction.

The current optimized serial implementation was reprofiled on 2026-09-08 with low-overhead user-mode instruction-pointer sampling and optimized MSVC symbols. Of 13,308 samples attributed to the filter source or median helper, fixed median-of-nine selection accounted for 70.00%, reflection/boundary handling for 5.85%, fused neighborhood-load/direct-address work for 16.29%, and remaining loop/local/output work for 7.86%. These shares are directional and use the unprofiled 2.803897 s median only for rough equivalent contributions. The roadmap decision is to move next to a bounded CPU-parallelism experiment, then CUDA, rather than pursue another scalar micro-optimization.

Portable OpenMP CPU parallelism was completed on 2026-09-08 without changing the optimized scalar kernel. Static decomposition over independent contiguous `scan_y` slabs matched the frozen Python fixture and all 47,228,125 optimized-serial canonical outputs bit for bit. On the 14-core/20-logical-processor Windows laptop, the best median was 319.583 ms at 20 threads versus a fresh 2789.567 ms serial median: `8.729×` speedup and 147.781 Moutput/s. Scaling flattened beyond eight threads and the 16-to-20-thread gain was modest. The portable CPU path is retained; Linux/Lambda GCC-or-Clang validation remains future work, and the next Phase A implementation step is the first CUDA kernel.

The correctness-first CUDA baseline was completed on 2026-09-09 as a separate, unoptimized one-output-per-thread kernel with 256-thread one-dimensional blocks. It preserves the C-order layout, reflect boundaries, `float64` data, and fixed median-of-nine semantics. The public 840-element fixture and all 47,228,125 canonical outputs matched bit for bit. On the AC-powered RTX 4070 Laptop GPU system, CUDA-event medians were 115.121 ms H2D, 38.191 ms kernel, 107.876 ms D2H, and 221.693 ms for the median per-run transfer-inclusive total; independently computed component medians do not sum to the total median. Fresh CPU medians were 2696.036 ms serial and 332.295 ms at 20 OpenMP threads, making the measured CUDA total `12.161×` faster than serial and `1.499×` faster than OpenMP. Transfers accounted for 76–86% of each total run.

The unchanged CUDA baseline was profiled on 2026-09-09 with Nsight Compute 2025.2.1 using focused replayed hardware-counter sections and source line information. Achieved occupancy was 95.57% with 40 registers/thread and no local-memory traffic; branch efficiency was 100%. SM throughput reached 83.87% while DRAM throughput reached only 40.57%, L2 hit rate was 89.24%, and loads used 29.54 of 32 bytes per sector. Of 522,280 PC samples, L1TEX-throttle and short-scoreboard stalls accounted for 56.01% and 26.19%. The kernel is therefore mixed instruction/latency limited—FP64 comparison-pipeline pressure plus nine-load queue/dependency stalls—not primarily bandwidth, occupancy/register, or divergence limited. Replay timing is not a benchmark; the unprofiled 38.191 ms median remains the reference. The ranked candidates are dimension-aware grid/thread mapping, mapping-aware shared-memory scan tiling, then reflection specialization/precomputation. The next controlled experiment is dimension-aware mapping because it removes repeated flat-coordinate decode and prepares a clean layout for any later tiling experiment. CUDA optimization and Python bindings remain planned.

### Phase B — adaptive median performance target

Phase B reproduces the current 4Denoise adaptive algorithm with `s=3` and `sMax=7`. The source performs nested Python pixel loops, strict two-stage min/median/max decisions, and conditional window growth with per-plane global-minimum constant padding. Initial profiling on bounded prepared subsets established repeated per-pixel neighborhood statistics as the dominant Python cost. Exact subset shapes and timings are runtime results recorded in the executed notebook, not permanent project claims.

The adaptive reference includes a deterministic `(7, 7, 1, 4)` branch test and a local, already-preprocessed `(16, 16, 4, 4)` real-data fixture. Finite `float64` outputs must match bit for bit. The experimental fixture is not distributed. Phase B becomes the main performance-engineering narrative after the Phase A warm-up.

### Intended contribution

Demonstrate a credible progression from a controlled fixed-window exercise to a measured adaptive bottleneck, then through modern C++, CPU optimization, CUDA, GPU profiling, validation, Python integration, and clear performance reporting.

## Project 2 — 4D-STEM 3D Median Filter

### Purpose

Extend the learned workflow to a more computationally and memory-intensive high-dimensional neighborhood operation.

### Currently known problem

The broad idea is to reorganize, unfold, or reshape 4D-STEM information into a long three-dimensional representation and perform a local 3D median/neighborhood filter. The exact transformation and scientific algorithm are TBD and must come from the intended research method rather than being inferred here.

### Intended contribution

Explore how dimensionality, neighborhood size, memory traffic, layout, cache effects, GPU memory behavior, thread organization, and volume scaling change the performance problem. It should build on Project 1 without merely copying it.

## Project 3 — General-Purpose CUDA K-Means

### Purpose

Demonstrate that the acquired performance-engineering skills generalize beyond microscopy to a standard machine-learning algorithm.

### Currently known problem

Develop a general-purpose K-means progression using generic synthetic and/or standard non-microscopy datasets. Study the different computational characteristics of assignment and update phases, convergence/correctness, transfer costs, and scaling with samples, dimensions, and clusters. Final datasets, initialization rules, convergence criteria, precision, and interface are TBD.

### Intended contribution

Provide a broadly applicable machine-learning example with meaningful CPU/GPU crossover analysis at sufficiently large, later-selected problem sizes. Microscopy is not the application or motivation for this project.

## Project 4 — CUDA Matrix / Tensor Multiplication

### Purpose

Connect the portfolio explicitly to AI and deep-learning workloads by studying a fundamental dense computational primitive rather than inventing a neural network.

### Currently known problem

Progress from clear CPU and CUDA baselines toward increasingly optimized dense matrix multiplication and/or a closely related tensor multiplication. Potential study areas include coalesced access, shared-memory tiling, thread/block design, arithmetic intensity, numerical validation, and dimension scaling. The precise initial scope, shapes, data types, and optimization sequence are TBD.

### Intended contribution

Explain why mature AI libraries are fast. Comparison with NVIDIA cuBLAS should eventually provide a relevant performance reference; beating cuBLAS is not the objective. A successful result will quantify improvement over the custom baseline, report the fraction of relevant mature-library performance reached, and use profiling to explain the remaining gap. NumPy, PyTorch, or TensorFlow comparisons may be added later when methodologically appropriate.

## How the projects complement one another

1. **Project 1:** Apply the complete workflow to an authentic scientific bottleneck.
2. **Project 2:** Address a harder, higher-dimensional, memory-intensive scientific workload.
3. **Project 3:** Transfer the same reasoning to a general-purpose machine-learning algorithm.
4. **Project 4:** Study a core primitive underlying AI workloads and compare with a highly optimized NVIDIA library.

Together, the projects retain the credibility of real scientific problems while demonstrating that the resulting C++/CUDA skills are not confined to microscopy.

## Expected learning progression

Project 1 establishes foundational C++, multidimensional memory representation, measurement, introductory CUDA, and integration. Project 2 deepens memory-system and scaling analysis. Project 3 adds a more general algorithm with distinct computational phases and workload parameters. Project 4 culminates in increasingly sophisticated GPU optimization and comparison with a mature performance ceiling.

Later projects should reuse principles and measurement discipline, but each should introduce a new performance question rather than repeat an earlier implementation mechanically.

## Benchmarking philosophy

- Benchmark multiple representative problem sizes and report distributions or stable summary statistics rather than a single convenient run.
- Define the timed region, warm-up behavior, synchronization, repetition method, and included setup costs.
- Record hardware, operating environment, compiler and flags, CUDA/toolkit versions, library versions, and input characteristics for serious results.
- Report kernel-only and end-to-end time separately when that distinction matters.
- Include allocation and host/device transfer costs where they occur in the real workflow.
- Compare implementations under equivalent correctness and workload conditions.
- Report CPU/GPU crossover behavior instead of assuming GPU execution always wins.
- Avoid unsupported speedup claims and disclose limitations in comparisons with mature libraries.

## Profiling philosophy

Profiling precedes optimization and should answer a specific question. Use the least intrusive suitable tool, check whether observations are stable, and connect profiler evidence to source-level or algorithmic behavior. Re-profile after changes because bottlenecks move. Generated assembly or machine instructions are supporting evidence when needed, not ends in themselves.

## Validation and correctness philosophy

Correctness criteria, edge cases, data types, and tolerances must be defined before optimization. Preserve trusted reference inputs and outputs, test across representative shapes, and distinguish exact equivalence from tolerance-based numerical agreement. Optimization must not silently alter scientific meaning or convergence behavior.

## Python-integration goal

Each mature project should offer a clean Python-facing interface appropriate for scientific or AI workflows. Integration should document array layout, supported data types, ownership and lifetime, error behavior, device selection where relevant, and whether calls copy or share memory. The binding approach remains a reversible decision until requirements are known.

## Repository-quality goal

All projects currently live in this public parent portfolio, and future work should arrive through incremental commits that preserve the performance-engineering progression. Each project may eventually become a standalone repository. A finished project should be reproducible, navigable, appropriately tested, honest about limitations, and understandable to a technical reviewer.

## Project Progress

### Project 1 — 4D-STEM Median Filter Acceleration

#### Phase A — Fixed `3 × 3` Median

- [x] Establish development environment/toolchain
- [x] Configure, build, and run the Phase A C++17 smoke test
- [x] Establish the fixed Python reference and real-data preparation
- [x] Establish the fixed correctness fixture and bit-for-bit criterion
- [x] Record an initial fixed Python timing for comparison with Phase B
- [x] Generalize both Python references to runtime-sized compatible 4D input
- [x] Save and verify the canonical unfiltered benchmark input with provenance
- [x] Load and validate the Phase A NPY fixture in C++
- [x] Establish verified 4D C-order indexing
- [x] Implement straightforward fixed `3 × 3` median in C++
- [x] Validate baseline C++ against the frozen Python fixture
- [x] Benchmark straightforward C++ and compare with the public 4Denoise wrapper
- [x] Profile C++
- [x] Complete isolated serial experiment 1: fixed median-of-nine selection
- [x] Complete isolated serial experiment 2: reduce flat-address generation
- [x] Reprofile the current optimized serial implementation
- [x] Implement and measure portable OpenMP CPU parallelism
- [x] Implement and exactly validate the first correctness-first CUDA kernel
- [x] Record separate CUDA transfer, kernel, and total-path baseline timings
- [x] Profile the CUDA baseline
- [ ] Perform basic CUDA optimization
- [ ] Add a Python binding after the CUDA path matures

#### Phase B — Adaptive Median

- [x] Locate the current public/private implementation and derive its actual algorithm
- [x] Establish the finite-`float64`, `s=3`, `sMax=7`, real-space contract
- [x] Create deterministic branch coverage and a small four-dimensional real-data fixture
- [x] Profile the unmodified Python implementation on representative prepared data
- [x] Compare adaptive and fixed filters fairly on the same prepared subset
- [ ] Define the formal benchmark hardware, repetition policy, and problem-size matrix
- [ ] Implement straightforward C++
- [ ] Validate C++ output
- [ ] Benchmark and profile C++
- [ ] Optimize and re-measure the CPU implementation
- [ ] Implement initial CUDA
- [ ] Validate and profile CUDA
- [ ] Optimize CUDA in response to measured behavior
- [ ] Add the Python binding
- [ ] Benchmark all relevant implementations
- [ ] Document the complete measured engineering story
- [x] Final pre-native repository/data organization

### Project 2 — 4D-STEM 3D Median Filter

- [ ] Define the intended scientific transformation and neighborhood operation
- [ ] Establish representative reference data and correctness criteria
- [ ] Profile the reference implementation
- [ ] Implement and validate straightforward C++
- [ ] Benchmark, profile, and optimize the CPU implementation
- [ ] Implement and validate initial CUDA
- [ ] Profile and optimize CUDA
- [ ] Add an appropriate Python interface
- [ ] Benchmark implementations across representative volume sizes
- [ ] Document results, limitations, and lessons
- [ ] Final repository cleanup

### Project 3 — General-Purpose CUDA K-Means

- [ ] Define general-purpose scope, datasets, and convergence criteria
- [ ] Establish and profile a trusted reference
- [ ] Implement and validate straightforward C++
- [ ] Benchmark, profile, and optimize CPU phases
- [ ] Implement and validate initial CUDA
- [ ] Profile and optimize assignment and update phases
- [ ] Analyze transfer costs and CPU/GPU crossover behavior
- [ ] Add an appropriate Python interface
- [ ] Benchmark scaling across samples, dimensions, and clusters
- [ ] Document results, limitations, and lessons
- [ ] Final repository cleanup

### Project 4 — CUDA Matrix / Tensor Multiplication

- [ ] Define operation scope, shapes, data types, and correctness criteria
- [ ] Establish trusted reference implementations
- [ ] Implement and validate straightforward CPU and CUDA baselines
- [ ] Benchmark and profile the CPU implementation
- [ ] Optimize and re-measure the CPU implementation
- [ ] Profile and iteratively optimize CUDA kernels
- [ ] Validate numerical behavior across representative dimensions
- [ ] Compare fairly with cuBLAS and other relevant mature libraries
- [ ] Add an appropriate Python interface if useful
- [ ] Document performance progression and the remaining library gap
- [ ] Final repository cleanup

## Rules for Future Development

- Do not optimize before establishing correctness.
- Profile before deciding what to optimize.
- Change one meaningful variable at a time when practical.
- Benchmark before and after optimizations.
- Preserve reference implementations for comparison.
- Record problem size, hardware, compiler, optimization flags, CUDA version, and relevant environment information in serious benchmark results.
- Separate kernel execution time from end-to-end time when relevant.
- Account for host/device transfer overhead.
- Avoid misleading speedup claims.
- Do not assume the GPU is faster for every problem size.
- Do not sacrifice numerical correctness merely to claim higher speed.
- Explain why optimizations work and why unsuccessful experiments did not.
- Prefer understanding over blindly accepting AI-generated code.
- Mark unknown scientific or implementation details as **TBD** rather than inventing them.
