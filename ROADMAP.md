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

Expose the high-performance implementation through a clean Python interface so Python or Jupyter users can access CPU or GPU acceleration without writing CUDA. Project 1 uses pybind11 with explicit NumPy, host-copy, and persistent-device ownership contracts; future projects should choose bindings from their own requirements.

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

The unchanged CUDA baseline was profiled on 2026-09-09 with Nsight Compute 2025.2.1 using focused replayed hardware-counter sections and source line information. Achieved occupancy was 95.57% with 40 registers/thread and no local-memory traffic; branch efficiency was 100%. SM throughput reached 83.87% while DRAM throughput reached only 40.57%, L2 hit rate was 89.24%, and loads used 29.54 of 32 bytes per sector. Of 522,280 PC samples, L1TEX-throttle and short-scoreboard stalls accounted for 56.01% and 26.19%. The kernel is therefore mixed instruction/latency limited—FP64 comparison-pipeline pressure plus nine-load queue/dependency stalls—not primarily bandwidth, occupancy/register, or divergence limited. The first selected experiment, dimension-aware grid/thread mapping, matched all 47,228,125 outputs bit for bit but produced no measurable gain (`1.000073×`) and was discarded.

A CUDA timing audit then clarified the apparent 38.191-versus-6.54 ms baseline conflict. Both earlier paths timed only the same unchanged 256-thread `sm_89` Release kernel; allocation, copies, initialization, validation, and event creation were outside the kernel event interval. The historical harness used fresh per-call resources and sparse CPU-interleaved invocations, while the mapping experiment issued GPU calls continuously. The new canonical protocol reuses device buffers and events, performs five warm-ups, and records 20 individual launches. It measured `6.501376 / 6.888448 / 6.969344` ms min/median/max with 1.878692% coefficient of variation and exact output. A three-second persistent-buffer idle caused no slowdown, but the exact historical harness reproduced a 45.636513 ms outlier among two approximately 6.9 ms launches. The 38.191 ms value is preserved as historical sparse-invocation evidence; transient scheduling/contention is supported, although the exact driver/WDDM mechanism and the condition that made all three historical samples high remain unresolved.

The Phase A CUDA kernel-optimization checkpoint is complete. Dimension-aware mapping was neutral (`1.000073×`), scan-space shared-memory tiling regressed approximately 1.3%, and an interior reflection fast path improved the median by only 0.635% within 2.34–2.45% run-to-run CV. Every candidate remained bitwise exact and was discarded. The simple profiled baseline is retained rather than adding unsupported complexity.

CUDA transfer/residency characterization was completed on 2026-09-09 with persistent device allocations and events. Across 20 canonical pageable one-call trials, H2D/kernel/D2H/total medians were `44.956097 / 6.557568 / 43.983521 / 99.865630` ms; pageable transfers were variable and occupied a median 93.456% of total time. Fresh medians were 2622.180 ms serial and 336.586 ms at 20 OpenMP threads, so the pageable CUDA total was `26.257×` and `3.370×` faster, respectively. One-copy/one-final-copy residency sequences at 1, 2, 5, and 10 operations reduced effective time per filter from 88.788 to 17.083 ms, although transfers still occupied 61.962% at ten operations. A synchronous pinned-memory diagnostic reduced the transfer pair from approximately 88.9 to 59.2 ms and stabilized both directions below 1% CV. A bounded synthetic size sweep placed the observed pageable-CUDA/OpenMP crossover between 4.1 and 10 million outputs on this laptop; it is directional because pageable transfers were noisy and shapes differed.

The correctness-first Phase A Python interface was completed on 2026-09-09 with opt-in pybind11 bindings for optimized serial, explicit-thread-count OpenMP, and optional one-shot CUDA execution. The module enforces the exact finite, four-dimensional, C-contiguous `float64` contract and returns a new NumPy array without mutating input. The public 840-element fixture and all 47,228,125 local canonical outputs matched bit for bit across available paths. Baseline calls explicitly copied NumPy into a vector and the returned vector into NumPy; CUDA additionally retained its one-shot allocation and transfer path. Canonical Python-call medians were 8.901303 s serial, 2.392543 s OpenMP-20, and 1.118562 s CUDA in a session with unusually slow CPU-native timings, so they establish interface behavior without replacing prior native results. The subsequent isolated experiments removed the CPU copies and added explicit persistent CUDA ownership.

Direct NumPy-buffer CPU bindings were completed on 2026-09-10. New caller-owned pointer entry points share one factored computational core with the retained vector APIs, and pybind11 keeps both NumPy owners alive while the GIL-free native call reads input and writes a new output directly. The fixture and all 47,228,125 canonical outputs remained bitwise exact. Against opt-in copied comparison paths, direct serial improved from 5.922454 to 5.399885 s (`1.096774×`) and direct OpenMP-20 from 1.289512 to 0.777705 s (`1.658099×`), removing approximately 0.52 s from either path. Finite validation retained a 0.158454 s median and is meaningful for the parallel interface but remains part of the contract. CPU direct buffers are retained; the one-shot CUDA function still uses its copied path, while the separate persistent object documented next avoids repeating that ownership cycle.

Persistent CUDA Python ownership was completed on 2026-09-10 without changing the kernel or removing the one-shot API. The non-copyable `CudaMedianBuffer` RAII owner allocates and uploads once, keeps input/output device storage across calls, applies every filter to the same original resident input, and downloads into independent NumPy-owned output. Fixture lifetime, ownership, repeated-call, invalid-input, and bitwise checks passed; the canonical resident result matched all 47,228,125 established outputs. In five interleaved Python workflow trials, the copied one-shot median was 0.835481 s. Persistent effective medians were 0.352179, 0.166032, 0.070753, and 0.043582 s/filter for 1, 2, 5, and 10 filters, respectively. Pageable host stages remained variable, while the ten-filter kernel sequence retained approximately 6.59 ms/launch. The architecture is retained.

The Phase A closeout audit regenerated the public Python fixture exactly, ran every retained native and Python implementation against it, confirmed input/output ownership contracts, reconciled historical versus current performance claims, and verified default CPU-only plus opt-in CUDA/Python build boundaries. Phase A is complete. Linux/Lambda validation remains a later portability task; it does not block the Phase B correctness-first native baseline.

### Phase B — adaptive median performance target

Phase B reproduces the current 4Denoise adaptive algorithm with `s=3` and `sMax=7`. The source performs nested Python pixel loops, strict two-stage min/median/max decisions, and conditional window growth with per-plane global-minimum constant padding. Initial profiling on bounded prepared subsets established repeated per-pixel neighborhood statistics as the dominant Python cost. Exact subset shapes and timings are runtime results recorded in the executed notebook, not permanent project claims.

The adaptive reference includes a deterministic `(7, 7, 1, 4)` branch test and a local, already-preprocessed `(16, 16, 4, 4)` real-data fixture. Finite `float64` outputs must match bit for bit. The experimental fixture is not distributed. Phase B becomes the main performance-engineering narrative after the Phase A warm-up.

The correctness-first native C++17 baseline was completed on 2026-09-10. It follows the Python control flow directly: separate detector-coordinate planes, global-minimum constant padding, explicit `3 × 3`/`5 × 5`/`7 × 7` windows, odd-count `std::nth_element` medians, strict Stage A/Stage B comparisons, and original-center fallback after the maximum window. The new committed public fixture covers every major branch and matched all 196 values bit for bit; the ignored local fixture matched all 4,096 values. A small local timing check established a 0.5261 ms native median versus 66.4967 ms for the public Python path, but formal benchmarking and profiling intentionally remain next.

The native baseline benchmark/profile was completed on 2026-09-10 using the exact historical centered `(64, 35, 8, 8)` subset (143,360 outputs) derived from the ignored canonical input. One warm-up plus seven Release `/O2` filter-only runs produced a 15.3165 ms median, 15.2896–15.4852 ms range, 0.430% CV, and 9.3598 Moutput/s. Five matching public 4Denoise calls produced a 17.7459 s median. Separate counters found 99.8214% of outputs finished at `3 × 3`; only 0.1786% expanded, all to maximum-window fallback, for 143,872 median computations.

Low-overhead main-thread sampling with optimized symbols attributed 10,482 of 10,517 samples to filter source: median selection 44.16%, per-window temporary-vector allocation/teardown 37.24%, gathering plus min/max 7.89%, plane preparation/indexing 5.65%, and remaining work 5.07%. Shares are directional because sampling perturbs wall time and optimized unwind data groups heap frames at their source call/scope-end sites. The ranked next candidates were fixed stack storage without changing selection, a specialized nine-value selector for the 99.82% common path, then direct-address `3 × 3` gathering.

The first isolated serial experiment was completed on 2026-09-10. Fixed 49-value stack storage removed per-window heap allocation while retaining the heap baseline, `std::nth_element`, gathering, padding, and every numerical decision. Public, local, and 143,360-output benchmark comparisons were bitwise exact, with identical branch and Stage B counters. In an AC-powered, alternating-order, logical-processor-0-controlled sequence, the fresh heap/stack medians were 15.6822/11.1307 ms: `1.408914×` speedup, 29.0234% runtime reduction, and 12.8797 Moutput/s. Reprofiling attributed 6,603 samples to filter source: median selection 69.27%, gathering/min-max 11.83%, plane/index work 11.02%, control/output 4.68%, and validation/output allocation 3.20%; per-window heap work disappeared as a measurable category. This evidence selected nine-value specialization for the 99.8214% common path while preserving general `25`/`49`-value selection.

The second isolated serial experiment was completed on 2026-09-10. The Phase A 19-comparator network replaced only the `3 × 3` `std::nth_element` call; larger adaptive windows retained general selection. It passed all 362,880 distinct-rank permutations, ten duplicate patterns, public/local fixtures, all 143,360 benchmark outputs, every branch counter, input immutability, and Phase A regression exactly. Under the established AC-powered, alternating, logical-processor-0 protocol, fresh fixed-stack/specialized medians were 11.3971/8.6934 ms: `1.311006×` speedup, 23.7227% runtime reduction, and 16.4907 Moutput/s. Of 5,328 attributed optimized samples, selection was 61.77%, gathering/min-max 14.21%, plane/index work 13.63%, control/output 6.72%, and validation/output allocation 3.68%. Selection remains largest but is already fixed on 99.8214% of outputs; direct padded-row `3 × 3` gathering is the next higher-value isolated experiment.

The third isolated serial experiment was completed on 2026-09-11. The retained specialized path remains callable, while an internal candidate replaces only the common `3 × 3` nested index loops with three padded-row bases and nine ordered loads; larger-window gathering and every numerical decision remain unchanged. Public, local, and 143,360-output comparisons were bitwise exact, all counters matched, input remained unchanged, and Phase A validation passed. The primary controlled sequence measured 9.1518/8.5972 ms baseline/candidate medians: `1.064509×`, a 6.0600% reduction, and 16.6752 Moutput/s. Two repeated sequences measured 8.2231% and 6.6462% reductions. Of 6,600 attributed candidate samples, selection was 60.17%, gathering/min-max 9.64%, plane/index work 18.82%, control/output 6.73%, and validation/output allocation 4.65%. The targeted gather share fell, while unchanged plane/index work became proportionally larger; the next experiment is portable OpenMP scaling over independent adaptive work rather than another small scalar micro-optimization.

Portable adaptive OpenMP scaling was completed on 2026-09-11. A separate explicit-thread-count API statically partitions the 15,875 independent detector-coordinate planes and merges thread-local diagnostics after the parallel region. Public, local, representative-subset, and all 47,228,125 canonical outputs matched the optimized serial path bit for bit; every diagnostic counter matched and Phase A remained exact. On the canonical workload, serial/OpenMP-1 medians were 3318.3376/3264.2994 ms, so no measurable one-thread penalty was established. Speedups at 2/4/8/16/20 threads were `1.934×/3.524×/5.616×/7.779×/7.379×`; 16 threads was best at 426.5593 ms and 110.7188 Moutput/s. Static plane counts differed by at most one, and only 0.1472% of outputs expanded beyond `3 × 3`, so adaptive load imbalance is unlikely to explain the late flattening. Cache, memory-system, and hybrid-core effects are plausible but unprofiled. CPU scaling is sufficient to move next to a correctness-first adaptive CUDA baseline.

The correctness-first adaptive CUDA baseline was completed on 2026-09-11 without changing any CPU implementation. A 256-thread one-dimensional detector-plane kernel computes the global padding minima, followed by a 256-thread one-output-per-thread adaptive kernel that preserves constant-border, `3 × 3`/`5 × 5`/`7 × 7`, strict Stage A/B, selector, and fallback semantics. Public 196-value, ignored local 4,096-value, representative 143,360-output, and full 47,228,125-output comparisons were bitwise exact; all ten CPU/CUDA diagnostic counters matched, input remained unchanged, and CPU-only plus Phase A regressions passed. Five warm-ups and seven persistent-resource event timings gave a 19.757919 ms steady adaptive-kernel median (0.274% CV, 2,390.339 Moutput/s), `167.950×` versus established serial and `21.589×` versus OpenMP-16 when excluding plane-minimum and transfers. A separate fresh-resource one-shot sequence had a 264.060638 ms median H2D-through-D2H path, `1.615×` versus OpenMP-16, but pageable transfers were noisy amid observed background GPU activity. The baseline is retained; focused Nsight Compute profiling of both kernels is the next experiment.

Focused Nsight Compute 2025.2.1 profiling was completed on 2026-09-11 using one warmed canonical launch per kernel, CUDA 12.9 `sm_89` Release code, temporary line information, and 36 replay passes of targeted sections. The 30-register plane-minimum kernel reached 97.24% DRAM throughput at 248.71 GB/s with only 28.51% achieved occupancy because its 63-block grid covers 0.29 waves/SM; it is a short bandwidth/long-scoreboard pass and only about 8% of the two-kernel device time. The adaptive kernel reached 64.87% SM, 15.64% DRAM, 92.97% L2, 65.75% achieved occupancy, and 99.14% branch efficiency with 46 registers/thread. Its explicit 49-double window created a 392-byte thread stack and 16.86M/71.41M local load/store instructions; schedulers had no eligible warp 84.91% of cycles, led by long-scoreboard, local/global-throttle, and short-scoreboard waits. With 97.81% L2 hits and only 0.1472% larger-window expansion, the kernel is mixed local-memory/cache-latency and FP64-instruction limited—not DRAM, divergence, or occupancy limited. Ranked candidates are common-path storage separation, rare larger-window selection, then modest global-gather coalescing. The next isolated experiment is a right-sized `3 × 3` first kernel with unchanged-semantics fallback processing for flagged expansions.

The common-path split experiment was completed and retained on 2026-09-11. A nine-value `3 × 3` kernel writes one dense fallback byte per output; a second full-grid kernel immediately skips common outputs and preserves the original `5 × 5`/`7 × 7` fallback. Public, local, representative, and canonical outputs were bitwise exact against CPU and monolithic CUDA, and all ten counters matched. Of 47,228,125 canonical outputs, 47,158,590 (99.852768%) completed in the first kernel, 69,535 entered fallback, and 66,348 reached `7 × 7`. In the final-build five-warm-up/seven-run interleaved comparison, monolithic and split medians were 19.767296 and 15.406080 ms: `1.283084×`, 22.062783% less runtime, and 3,065.551 Moutput/s. Two earlier complete sequences measured `1.295295×` and `1.283172×`, confirming a repeatable 22–23% reduction. The common kernel uses 40 registers with no stack/local allocation versus 46 registers and a 392-byte stack for the monolith; focused Nsight measured 100% theoretical and 96.83% achieved occupancy. The retained kernel was selected for the full focused profile reported below.

The retained common kernel was profiled on 2026-09-23 with Nsight Compute 2025.2.1, temporary source line information, and one warmed canonical performance launch; the normal Release build was restored afterward. Its 40-register kernel achieved 96.85% occupancy, 83.65% SM throughput, 20.15% DRAM throughput, 91.89% L2 hit rate, and 99.90% branch efficiency. The compiled kernel had a zero-byte stack frame and no local-memory instructions or sectors; Nsight's 1,024-byte launch-stack entry is the configured device stack limit, not a per-thread frame. Nevertheless, schedulers had no eligible warp in 85.89% of cycles: L1TEX-queue throttle (58.56 cycles/issued instruction) and short scoreboard (19.02) dominated, while long scoreboard (0.18) and local/global throttle were negligible. Source samples cluster at the neighborhood min/max updates and inlined median compare/swaps, but source attribution overlaps after inlining and the sampled PC does not by itself identify the producer of a stall. Relative to the monolithic profile, stack/local traffic and long-scoreboard stalls disappeared while scheduler eligibility remained poor. This is a mixed instruction/dependency and L1TEX-queue bottleneck, not DRAM, occupancy, or branch divergence. Ranked candidates were (1) a balanced nine-value min/max reduction to test the largest sampled dependency chain, (2) a lower-dependency median-of-nine network, and (3) gather/address/coalescing changes, whose measured memory-sector excess is modest. Candidate 1 was selected for the isolated experiment below; no optimization was made during profiling. The 14.055424 ms common-kernel and 15.406080 ms combined medians remain the unprofiled performance references; profiler replay time is not a benchmark.

The balanced nine-value min/max experiment was completed and retained on 2026-09-23. A separate common kernel keeps the nine loads, median network, and all adaptive decisions unchanged while replacing serial min/max updates with a tie-preserving reduction tree. Public, local, 143,360-output subset, and all 47,228,125 canonical outputs matched optimized CPU and split CUDA bit for bit; all ten counters matched and inputs remained unchanged. In the final normal-Release seven-run paired sequence, common-kernel control/candidate medians were `17.236992/15.799296` ms: `1.090997×`, an 8.340756% reduction, and `2,989.255 Moutput/s` for the candidate. Five additional sequences showed 6.63–8.70% gains, although one had severe clock/load drift. The candidate uses 46 rather than 40 registers/thread, reducing achieved occupancy from 95.07% to 80.22%, but retains zero compiled stack/local traffic. Focused Nsight comparison found L1TEX-throttle cycles per issued instruction falling 56.82→39.00, short scoreboard rising 19.02→20.51, and no-eligible-warp cycles remaining near 85%; attribution remains directional. A lower-dependency median-of-nine network was selected as the next isolated experiment.

That median-network experiment was completed and rejected on 2026-09-24. A 20-comparator, depth-seven network replaced only the balanced common kernel's retained 19-comparator, depth-nine selector; mixed signed-zero windows used the retained selector to preserve exact bit patterns. All 362,880 distinct permutations, 262,144 signed-zero/tie windows, public/local fixtures, 143,360-output subset, and 47,228,125-output canonical comparison passed exactly, including all ten counters and input immutability. Three seven-pair interleaved Release sequences gave control/candidate common medians of `12.953600/13.629440`, `12.954624/13.633536`, and `12.957632/13.634560` ms. The pooled medians were `12.954624/13.632512` ms: a repeatable 5.23% regression despite 46→40 registers/thread and no compiled stack/local storage. No combined-path timing or follow-up Nsight campaign was justified; experimental source was removed and the balanced implementation remains the retained path. Next: characterize adaptive CUDA transfer and data-residency costs before considering further kernel changes.

Retained-path adaptive CUDA transfer/residency characterization completed on 2026-09-24 without kernel changes. Public/local/subset/canonical pageable, pinned-staged, and repeated-resident outputs were bitwise exact against optimized CPU and balanced CUDA; diagnostic counters and host inputs remained unchanged. In the second controlled AC-powered CUDA 12.9 `sm_89` Release sequence, seven alternating one-shot calls gave pageable/pinned native-wall medians of `234.543/277.895` ms and event H2D-through-D2H medians of `93.793/88.566` ms. Across 20 paired synchronous transfer-only runs, pageable/pinned H2D medians were `42.758/37.085` ms and D2H medians `42.133/35.176` ms; staging copies erased the isolated transfer gain in complete one-shot calls. Five sequences per resident count gave effective event-path medians of `125.403, 67.459, 39.324, 29.425, 24.587` ms/filter at 1, 2, 5, 10, 20 complete operations on the same resident input, with one initial H2D and final D2H. At 20 operations, complete device work was `20.197` ms/filter and transfers were 17.64% of the GPU path. Fresh serial/OpenMP-16 filter medians were `3275.932/424.181` ms, so pageable one-shot native wall was `13.967×/1.809×` faster; resident GPU-path-only comparisons exclude host allocation/setup and are not application end-to-end speedups. The earlier full sequence confirmed the trend but showed laptop timing drift. Canonical resident device buffers total 803,005,125 bytes (~765.8 MiB), including 47,228,125 dense flag bytes. This motivated a persistent adaptive CUDA Python owner, not pinned staging or another kernel experiment.

The Phase B persistent adaptive CUDA Python owner was completed on 2026-09-24 without kernel or algorithm changes. `CudaAdaptiveMedianBuffer(shape)` owns the same four device buffers (~765.8 MiB canonical), exposes explicit validated `upload()`, complete synchronized `filter()`, and independent-array `download()`, and keeps repeated calls on the same resident input rather than chaining outputs. Public (196), ignored local (4,096), representative (143,360), and canonical (47,228,125) Python outputs matched the established native/Python references bit for bit. Input/output ownership, replacement upload, lifecycle errors, and Phase A/B regressions passed. In the more stable five-run AC-powered Python sequence, a newly constructed one-call owner took `184.567` ms median; one upload, 20 complete filters, and one download on an already allocated owner took `28.221` ms/filter (`6.540×` lower effective wall time). The complete synchronized Python filter call was `20.312` ms median at 20 repetitions; upload/download medians were `89.719/66.490` ms. An earlier complete sequence confirmed the direction (`193.526` to `28.268` ms/filter). Python wall timing includes NumPy validation/output allocation and differs from native CUDA-event boundaries. With scientific reference, profiled serial/OpenMP/CUDA progression, transfer evidence, persistent Python ownership, and exact canonical validation complete, Phase B closes for the established Windows finite-`float64`, `s=3`, `sMax=7` contract. Broader problem-size and Linux validation remain future portability studies. The next active project is general-purpose CUDA K-means; no further Project 1 optimization is planned.

### Intended contribution

Demonstrate a credible progression from a controlled fixed-window exercise to a measured adaptive bottleneck, then through modern C++, CPU optimization, CUDA, GPU profiling, validation, Python integration, and clear performance reporting.

## Project 2 — General-Purpose CUDA K-Means

The existing placeholder directory prefixes still reflect the earlier order; no files are being renamed or moved in this transition. K-means is the next active project, while the old 3D-median concept is deferred and matrix/tensor multiplication follows as Project 3.

### Purpose

Demonstrate that the acquired performance-engineering skills generalize beyond microscopy. Unlike Project 1's scan-space neighborhoods, K-means has point-to-centroid distance work, assignment, many-to-one centroid reductions, synchronization, iterative convergence, and changing CPU/GPU economics across `N`, `D`, `K`, and iteration count.

### Initial algorithmic contract — approved and frozen in Python

- Input `X` is a dense, finite, C-contiguous, row-major `float32` array of shape `(N, D)`. The first supported range is `2 <= K <= min(N,32)`, `K <= N <= 2^20`, `1 <= D <= 32`, and `|X[i,d]| <= 1024`; these bounds keep the initial floating-point and memory domain explicit. `K` is fixed during a fit but supplied at runtime. Public fits permit at most 100 update/reassignment passes; a reduced cap exists only as an internal test hook. No weights, sparse data, normalization, multiple restarts, or random initialization. Input is unchanged.
- Initialize centroid `k` from a copy of row `((2k + 1) * N) // (2 * K)` of `X`, using integer arithmetic for `k = 0..K-1`. This gives one deterministic run and stable cluster identities. Assign every sample to the centroid minimizing squared Euclidean distance, `sum_d (X[i,d] - C[k,d])^2`; subtraction, squaring, and accumulation are separate `float32` operations, with feature contributions added in ascending index order. Exact computed-distance ties select the lowest cluster index.
- Make an initial assignment `A0`. Each update pass accumulates assigned samples in `float64`, divides by the integer cluster count in `float64`, rounds each nonempty centroid coordinate to `float32`, and retains the preceding `float32` centroid for an empty cluster. Reassign all points against these new centroids. Stop after the first pass whose new labels equal the preceding labels, or after 100 passes. No movement tolerance or implicit reseeding; no hidden additional update at the cap.
- Return a new `int32[N]` label array, a new C-contiguous `float32[K,D]` centroid array, the number of update passes, and a convergence Boolean. Returned labels are the final assignment against returned centroids. If the update cap is reached without convergence, those centroids may reflect the preceding labels; this is reported as nonconverged. Inertia is a separately recomputed `float64` diagnostic, `sum_i ||X[i] - C_final[A_final[i]]||^2`, not a hidden part of timed fit calls.

The authoritative [Python reference](03_CUDA_KMeans/python/reference.py) uses explicit `float32` assignment arithmetic and `float64` centroid sums/means over the supplied `float32` samples. It processes distances in bounded sample chunks rather than materializing an `N × K × D` tensor. Validation inertia is recomputed separately in `float64` and is not an implicit fit stage. Future optimized reductions need not be bitwise identical, but must satisfy the frozen validation rules below.

### Validation and public workloads

Hand-authored public fixtures (`N=16–64`, small `D`/`K`) cover lowest-index ties, duplicate initial centroids and empty clusters, singleton clusters, stable convergence, and a nonconverged final reassignment through the internal reduced-cap test hook. On every published fixture and benchmark input, require exact initialization indices, labels, update counts, convergence flags, shapes, dtypes, and input immutability; a near-boundary label mismatch fails this gate and its distance margin is diagnostic, not an exemption. Compare centroid coordinate `d` with the oracle within `5e-6 × S_d`, where `S_d=max(1,max_i|X[i,d]|)`; compare independently recomputed inertia within `2e-5 × max(1,I_reference)`. An exactly representable constant-point fixture must yield exactly zero inertia. These are preregistered starting limits, not values to widen merely to pass CUDA. Bitwise equality is not generally required for parallel floating-point sums; behavior on arbitrary untested near-boundary inputs cannot be guaranteed by a finite test suite.

Use only public deterministic synthetic data: hand-authored tiny fixtures; a seeded, version-recorded moderate `(N,D,K)=(65,536,8,16)` profiling workload; and `(262,144,16,16)` plus an optional `(1,048,576,32,32)` GPU-scale stress workload, all defined by the [public generator](03_CUDA_KMeans/python/benchmark_data.py). Vary `N`, `D`, and `K` independently after the first baseline. Do not require private microscopy data. For like-for-like stage timing, plan a benchmark-only fixed-pass mode that runs exactly `T` update/reassignment passes without early exit; measure normal convergence separately and report actual passes. Fixed-pass mode is not part of the public fit contract.

The straightforward serial C++ baseline was profiled on 2026-09-25 without changing its normal Release hot loops. All three approved separated workloads converged after one update; an additional reproducible `(16,384,8,8)` overlapping-mixture diagnostic (PCG64 seed `20260927`) naturally took 22 updates, matched the Python reference, and does not replace them. In fresh normal `/O2 /fp:precise` timing, the approved moderate workload's seven-run median was `14.3782` ms; its earlier `9.8469` ms median remains historical, with the environmental difference unexplained. Coarse phase timers in a separate build assigned about 92–93% of complete-fit time to assignment on both the one-update and 22-update workloads, versus about 2% and 6% to centroid updates. Main-thread IP sampling corroborated assignment at 91.27% of 12,568 primary samples and 92.09% of 9,512 iterative samples. Source attribution concentrated on distance accumulation, combined address/load/subtract, and loop/selection work; inlining prevents exact operation-level separation. Centroid accumulation becomes more relevant with repeated updates but does not displace assignment. This evidence selected the isolated address-calculation experiment below. Full phase and sample breakdowns are in the [Project 2 report](03_CUDA_KMeans/README.md).

The first serial CPU experiment retained a separate address-aware assignment path while preserving the straightforward baseline, ordered FP32 distance arithmetic, and FP64 updates. Both paths passed all six public fixtures and exact bitwise comparison on the approved `(65,536,8,16)` and 22-update `(16,384,8,8)` workloads; Project 1 smoke tests passed. One warm-up each and three seven-pair alternating normal-Release primary blocks yielded pooled baseline/candidate medians of `17.4447/16.3819` ms (`1.064876×`, 6.0924% reduction). A bounded second 21-pair confirmation yielded `17.5925/16.4837` ms (`1.067266×`), despite several timing outliers. The 22-update workload showed `32.8941/30.0874` ms in its first five pairs and `31.0433/28.1952` ms pooled across three confirming five-pair blocks; no repeated regression appeared. A fresh symbol-only sample comparison left assignment dominant (91.11% baseline, 89.79% candidate), and the original combined address/load/subtract source line's 29.61% attribution redistributed among row setup, difference/load, and loop lines. Inlining precludes a precise instruction-level savings claim; paired timings support retention. This evidence selected the two-feature experiment below; detailed raw results and caveats are in the [Project 2 report](03_CUDA_KMeans/README.md).

The isolated two-feature FP32 distance-pipeline experiment preserved row-pointer addressing and sequential feature-order additions, passed all six public fixtures plus D=1/2/3 and ordered-addition checks, and matched the addressed control bit for bit on both performance workloads. In three alternating seven-pair `/O2 /fp:precise` primary blocks, addressed/pipeline pooled medians were `9.4789/10.5889` ms: `0.895173×` control/candidate speedup, or 11.7102% longer candidate runtime; all block medians favored the control. Three five-pair blocks on the 22-update workload gave `16.3654/17.7841` ms (`0.920226×`, 8.6689% longer). A bounded assembly check showed ordered scalar additions without FMA or reassociation; MSVC had already unrolled the control by four features, while the candidate emitted a two-feature loop. The candidate source/test changes were discarded and normal Release rebuilt. Absolute host times vary across sessions, so only paired same-session comparisons support this decision. A further cluster-selection scalar experiment has a modest ~9% sampled ceiling; portable OpenMP decomposition is the next milestone instead of chasing another micro-optimization. Full raw timings are in the [Project 2 report](03_CUDA_KMeans/README.md).

### Engineering path and decisions to defer

Build the Python reference and fixtures first, then a straightforward serial C++ implementation, native profiling and measured scalar changes, OpenMP, correctness-first CUDA, Nsight-guided stage optimization, transfer/device-residency characterization, and a Python interface only when useful. Separate assignment, centroid update, convergence, transfers, and whole-fit time. Do not predetermine CUDA point mapping, AoS versus SoA layout, atomics versus hierarchical reductions, centroid caching, kernel fusion, or convergence-reduction strategy before baseline measurements.

Later compare with the [scikit-learn CPU KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and, only if practical, [RAPIDS cuVS/cuML GPU K-means](https://docs.rapids.ai/api/cuvs/stable/c_api/cluster_kmeans_c/). Supply the same initial centroids and disclose differences in empty-cluster and stopping semantics, actual iteration counts, precision, and included setup/transfer costs. Report objective quality, whole-fit time and per-stage time, CPU/GPU crossover, and the custom implementation's fraction of library throughput where comparable; beating a mature library is not required.

### Intended contribution

Show a different, iterative reduction-heavy GPU workload and explain measured design tradeoffs rather than reproduce Project 1's neighborhood optimization sequence.

## Project 3 — CUDA Matrix / Tensor Multiplication

### Purpose

Connect the portfolio explicitly to AI and deep-learning workloads by studying a fundamental dense computational primitive rather than inventing a neural network.

### Currently known problem

Progress from clear CPU and CUDA baselines toward increasingly optimized dense matrix multiplication and/or a closely related tensor multiplication. Potential study areas include coalesced access, shared-memory tiling, thread/block design, arithmetic intensity, numerical validation, and dimension scaling. The precise initial scope, shapes, data types, and optimization sequence are TBD.

### Intended contribution

Explain why mature AI libraries are fast. Comparison with NVIDIA cuBLAS should eventually provide a relevant performance reference; beating cuBLAS is not the objective. A successful result will quantify improvement over the custom baseline, report the fraction of relevant mature-library performance reached, and use profiling to explain the remaining gap. NumPy, PyTorch, or TensorFlow comparisons may be added later when methodologically appropriate.

## Deferred concept — 4D-STEM 3D Median Filter

The former Project 2 remains a research concept, not an active numbered deliverable. Its transformation, neighborhood, and scientific reference are still undefined. It could become valuable if that operation proves scientifically distinct and exposes a new memory/layout question; otherwise another median would overlap Project 1's fixed/adaptive CPU, OpenMP, CUDA, profiling, residency, and Python evidence. Its existing directory is retained without moving files during this roadmap transition.

## How the projects complement one another

1. **Project 1:** Apply the complete workflow to an authentic scientific bottleneck.
2. **Project 2:** Transfer the same reasoning to general-purpose, iterative K-means assignment and reductions.
3. **Project 3:** Study a core primitive underlying AI workloads and compare with a highly optimized NVIDIA library.

Together, the projects retain the credibility of real scientific problems while demonstrating that the resulting C++/CUDA skills are not confined to microscopy.

## Expected learning progression

Project 1 establishes foundational C++, multidimensional memory representation, measurement, CUDA, and integration. Project 2 adds a general-purpose iterative algorithm with distinct assignment and reduction phases, convergence, and workload parameters. Project 3 studies a dense computational primitive against a mature performance ceiling. The 3D-median idea remains deferred pending a distinct scientific contract.

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
- [x] Test dimension-aware CUDA mapping; discard after no measurable improvement
- [x] Stabilize steady-state CUDA kernel benchmarking
- [x] Complete the CUDA optimization checkpoint and discard unhelpful candidates
- [x] Characterize pageable/pinned transfers, residency amortization, and CPU/GPU crossover
- [x] Add a correctness-first Python interface with explicit host-copy behavior and optional CUDA
- [x] Remove NumPy/vector copies from CPU/OpenMP through validated direct-buffer entry points
- [x] Add explicit persistent CUDA ownership for repeated device-resident work
- [x] Consolidate final Phase A validation, reproduction guidance, and headline results

#### Phase B — Adaptive Median

- [x] Locate the current public/private implementation and derive its actual algorithm
- [x] Establish the finite-`float64`, `s=3`, `sMax=7`, real-space contract
- [x] Create deterministic branch coverage and a small four-dimensional real-data fixture
- [x] Profile the unmodified Python implementation on representative prepared data
- [x] Compare adaptive and fixed filters fairly on the same prepared subset
- [x] Establish Windows benchmark hardware, repetition policy, and representative/canonical workloads
- [x] Implement straightforward C++
- [x] Validate C++ output
- [x] Benchmark and profile C++
- [x] Remove per-window heap allocation with exact fixed-stack storage
- [x] Specialize and exhaustively verify nine-value selection for the common path
- [x] Replace common-path generic gathering with direct padded-row loads
- [x] Measure portable OpenMP scaling of the retained optimized adaptive path
- [x] Optimize and re-measure the serial CPU implementation
- [x] Implement initial CUDA
- [x] Validate and profile CUDA
- [x] Optimize CUDA in response to measured behavior, retaining only beneficial changes
- [x] Characterize adaptive CUDA one-shot transfers, pinned staging, and repeated device residency
- [x] Add persistent adaptive CUDA Python ownership with explicit transfer semantics
- [x] Add the adaptive Python binding and validate exact public/local/subset/canonical outputs
- [x] Benchmark retained native and Python-facing workflows on representative/canonical workloads
- [x] Document the measured engineering story and close Phase B for the established contract
- [x] Final pre-native repository/data organization

### Project 2 — General-Purpose CUDA K-Means

- [x] Define the first deterministic scope, algorithm, and validation contract
- [x] Create the authoritative NumPy reference and public deterministic fixtures
- [x] Implement and validate straightforward serial C++
- [x] Establish initial native/Python whole-fit timing on the moderate public workload
- [x] Profile the serial C++ baseline on the moderate public workload and a separate natural multi-update diagnostic
- [x] Reduce serial assignment address work in a separate, exactly validated candidate
- [x] Test and reject exact two-feature distance scheduling after paired regression
- [ ] Implement and measure portable OpenMP CPU decomposition
- [ ] Implement and validate initial CUDA
- [ ] Profile CUDA and optimize measured assignment/update bottlenecks
- [ ] Analyze transfer, iteration, residency, and CPU/GPU crossover costs
- [ ] Add a Python interface if it improves the demonstrated workflow
- [ ] Benchmark separate `N`, `D`, `K`, and iteration-count scaling
- [ ] Compare fairly with available established CPU/GPU libraries
- [ ] Document results, limitations, and lessons
- [ ] Final repository cleanup

### Project 3 — CUDA Matrix / Tensor Multiplication

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

### Deferred — 4D-STEM 3D Median Filter

- [ ] Reconsider only after defining a distinct scientific transformation and performance question

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
