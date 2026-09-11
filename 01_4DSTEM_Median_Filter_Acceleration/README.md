# Project 1 — 4D-STEM Median Filter Acceleration

## Objective

Project 1 develops one performance-engineering workflow through two related real-space median filters on 4D-STEM data. Phase A is a controlled fixed-window warm-up; Phase B is the main adaptive-median performance target.

The straightforward Phase A C++ baseline applies the fixed `3 × 3` scan-space median and matches the Python reference bit for bit. It remains available beside the optimized-serial, OpenMP, and correctness-first CUDA implementations. The full-input Release baseline, CPU profiles, isolated serial and CUDA experiments, Windows multicore scaling, CUDA profiling, stabilized kernel benchmark, transfer/residency characterization, and Python interface experiments are recorded below. CPU bindings use direct validated NumPy buffers, and the CUDA binding offers both one-shot execution and explicit persistent device ownership. Phase B has now begun with a separate correctness-first native adaptive-median baseline.

## Performance boundary

Both phases start from the same scientifically prepared array:

```text
ripple_data_reduced.npy
    -> center-disk alignment fitted from the current data
    -> elliptical correction fitted from the current mean diffraction pattern
    -> median_filter_input
    -> Phase A fixed median or Phase B adaptive median
```

Source loading, center alignment, ellipse fitting, and elliptical correction are excluded from median-kernel timing. The project begins at `median_filter_input`.

The general array contract is `(scan_y, scan_x, detector_y, detector_x)`. Numerical dimensions are discovered at runtime. Project 1 is not tied to one experimental array shape.

## Phase A closeout summary

Phase A is complete. The authoritative progression is the 3.756627 s straightforward C++ baseline; CPU profiles that motivated isolated `1.135×` median-of-nine and `1.105×` direct-address improvements against fresh same-session baselines; a 319.583 ms OpenMP-20 median (`8.729×` versus its fresh optimized-serial baseline); and an exact CUDA baseline with a stabilized 6.888448 ms kernel median. Nsight Compute classified that kernel as mixed instruction/latency limited. Dimension-aware mapping, shared-memory tiling, and an interior reflection fast path were tested exactly and discarded when measurements did not justify their complexity.

The native transfer study measured a 99.865630 ms pageable one-call median and 17.083 ms/filter across ten resident operations. These are distinct from Python wall-time results: direct NumPy buffers improved copied serial and OpenMP-20 binding medians by `1.096774×` and `1.658099×`, while `CudaMedianBuffer` reduced effective time from a 0.835481 s copied one-shot median to 0.043582 s/filter across ten resident calls (`19.170×`) in its variable pageable-transfer session, with approximately 6.59 ms/kernel. Historical sparse-invocation CUDA values, including the initial 38.191 ms kernel median, remain documented below but are not the steady-state optimization baseline. Public and canonical correctness evidence is bit-for-bit under the finite-`float64` contract.

## Phase A — fixed `3 × 3` median warm-up

Phase A applies a centered `3 × 3` median over scan axes 0 and 1, independently for every detector coordinate, through:

```python
fd.HyperData(data).denoise(
    method="median",
    domain="real",
    window_size=3,
    mode="reflect",
    cval=0.0,
    origin=0,
    return_array=True,
)
```

It establishes the initial C++ implementation, multidimensional indexing, exact validation, CPU measurement, and first CUDA workflow. SciPy already provides a mature fixed median, so outperforming it is not the primary learning objective.

The scientific reference is [python_reference_2d_median_filter.ipynb](python_reference_2d_median_filter.ipynb). Its published outputs are cleared because the experimental arrays remain local. A separate deterministic synthetic fixture supports the public native validation run.

### Native build

The native structure remains intentionally small:

```text
cpp/
    CMakeLists.txt
    include/
        adaptive_median.hpp
        fixed_median.hpp
        fixed_median_cuda.hpp
    src/
        adaptive_median.cpp
        adaptive_benchmark.cpp
        adaptive_validation.cpp
        cuda_benchmark.cpp
        cuda_kernel_benchmark.cpp
        cuda_transfer_characterization.cpp
        cuda_validation.cpp
        main.cpp
        fixed_median.cpp
        fixed_median_cuda.cu
        python_bindings.cpp
    third_party/
        libnpy/
            include/npy.hpp
            LICENSE
            README.md
    build/              # generated; ignored by Git
python/
    validate_bindings.py
```

`fixed_median.cpp` contains the correctness-first fixed `3 × 3` implementation: half-sample symmetric reflection on the two scan axes, nine-value median selection, and separate input/output storage. `main.cpp` loads and validates an input/reference pair and compares every `double` by its `uint64_t` bit representation. The public default is a deterministic synthetic fixture; alternate compatible arrays may be supplied explicitly. General shape is discovered at runtime.

The CPU targets depend only on compiler-supported OpenMP and the vendored, header-only [libnpy](cpp/third_party/libnpy/README.md) `v1.0.1`, pinned to commit `890ea4fcda302a580e633c624c6a63e2a5d422f6` under its MIT license. CUDA is opt-in through `PHASE_A_ENABLE_CUDA`; CMake then enables the CUDA language and links the standard `CUDA::cudart` target. Python bindings are independently opt-in through `PHASE_A_ENABLE_PYTHON` and require pybind11 in the selected Python environment. The source-tree fixture paths are embedded, so default validation runs do not depend on the working directory.

From a Visual Studio 2022 x64 Developer Command Prompt, configure and build with:

```bat
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Debug
cmake --build cpp/build --config Release
cpp\build\Release\phase_a_fixed_median.exe
cpp\build\Release\phase_b_adaptive_median_validation.exe
cpp\build\Release\phase_b_adaptive_median_benchmark.exe
```

On this Windows system, CUDA 12.9 is selected explicitly because its Visual Studio build-customization files are installed with the toolkit but not registered under the Visual Studio directory:

```bat
cmake -S cpp -B cpp/out/cuda -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" -DPHASE_A_ENABLE_CUDA=ON -DPHASE_A_CUDA_ARCHITECTURES=89
cmake --build cpp/out/cuda --config Release --target phase_a_cuda_validation phase_a_cuda_benchmark phase_a_cuda_kernel_benchmark phase_a_cuda_transfer_characterization phase_b_adaptive_cuda_validation phase_b_adaptive_cuda_benchmark
cpp\out\cuda\Release\phase_a_cuda_validation.exe
cpp\out\cuda\Release\phase_a_cuda_benchmark.exe
cpp\out\cuda\Release\phase_a_cuda_kernel_benchmark.exe
cpp\out\cuda\Release\phase_a_cuda_transfer_characterization.exe
```

Other installations with registered CUDA integration may omit the explicit `-T` selection; `PHASE_A_CUDA_ARCHITECTURES` remains overridable for other GPUs. No Visual Studio or toolkit installation was modified.

The correctness-first Python extension can be built CPU-only or together with CUDA. This Windows example uses the `hanlab` interpreter and enables both:

```bat
cmake -S cpp -B cpp/out/python -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" -DPHASE_A_ENABLE_CUDA=ON -DPHASE_A_ENABLE_PYTHON=ON -DPython_EXECUTABLE=C:\Users\haloe\anaconda3\envs\hanlab\python.exe
cmake --build cpp/out/python --config Release --target fourdstem_median
set PYTHONPATH=%CD%\cpp\out\python\Release
C:\Users\haloe\anaconda3\envs\hanlab\python.exe python\validate_bindings.py
```

Omit `PHASE_A_ENABLE_CUDA` and the CUDA toolset selection for a CPU-only extension. The build copies the CUDA runtime DLL beside a CUDA-enabled Windows module so modern Python can load it without a process-wide `PATH` change. Linux compilation remains unverified.

CMake reads `CMakeLists.txt`, adds the project and libnpy include directories, applies the C++17 target requirements, and generates Visual Studio build files under `cpp/build/`. The Phase A and Phase B validation executables are separate targets; the adaptive target remains CPU-only and does not require CUDA. Debug favors diagnosis, while Release enables the toolchain's normal optimized configuration.

Everything under `cpp/build/` is generated and may be deleted and recreated from `CMakeLists.txt`; it is ignored by Git.

### Phase A baseline benchmark — 2026-09-08

This initial single-machine baseline used the canonical C-contiguous `float64` input with runtime shape `(85, 35, 127, 125)`: 47,228,125 output elements and 377,825,000 logical output bytes. The machine was a 64-bit Windows 11 Home 10.0.26200 system with a 13th Gen Intel Core i7-13700H (14 cores, 20 logical processors) and 68,408,406,016 bytes of physical memory. The native target used C++17, MSVC tools 14.44.35207, CMake 3.31.6-msvc6, x64 Release, and the generated `MaxSpeed` compiler setting. The Python baseline used the `hanlab` environment: Python 3.12.7, NumPy 1.26.4, and SciPy 1.15.1.

Each filter received one untimed warm-up followed by three timed runs in one process. Loading was timed separately. No profiling or algorithm changes were made.

| Path | NPY load (ms) | Timed filter runs (ms) | Min / median / max (ms) | Approx. effective output data rate |
| --- | ---: | --- | ---: | ---: |
| C++ `fixed_median_3x3(...)` | 296.972 | 3756.627, 3674.788, 3797.662 | 3674.788 / 3756.627 / 3797.662 | 12.572 Moutput/s; 0.101 output GB/s |
| `HyperData.denoise(...)` | 102.746 | 4112.253, 4136.414, 4042.099 | 4042.099 / 4112.253 / 4136.414 | 11.485 Moutput/s; 0.092 output GB/s |

The median-time ratio is `4Denoise wrapper / C++ = 1.095×`. Effective output data rate is output size divided by median elapsed time; it is not an estimate of total memory bandwidth.

The C++ filter interval contains only `fixed_median_3x3(...)`, including its internal output-vector allocation. The Python interval contains the complete public `HyperData.denoise(method="median", domain="real", window_size=3, mode="reflect", cval=0.0, origin=0, return_array=True)` call, including dispatch, filtering, output allocation, and return-array handling. Python import, NPY loading, and initial `HyperData(data)` construction are excluded. Source inspection confirmed that `domain="real"` routes the median method over axes `(0, 1)`; no direct SciPy call was benchmarked.

The three shared boundary/interior sanity coordinates agreed bit for bit between paths and remained bitwise-consistent across each path's warm-up and timed runs.

Reproduce the measurement from this project directory with:

```bat
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Release --target phase_a_fixed_median phase_a_baseline_benchmark
cpp\build\Release\phase_a_fixed_median.exe
cpp\build\Release\phase_a_baseline_benchmark.exe
python benchmarks\benchmark_fourdenoise.py --fourdenoise-root path\to\4denoise
```

### Phase A CPU profile — 2026-09-08

The canonical input was profiled on the same x64 Release implementation. The normal `MaxSpeed` (`/O2`) setting and `AnySuitable` inlining were retained. Configuration with `-DPHASE_A_ENABLE_PROFILING_SYMBOLS=ON` adds only `/Zi /Zo` compiler symbols and `/DEBUG:FULL /OPT:REF /OPT:ICF` linker settings to the benchmark target.

Visual Studio DiagnosticsHub CPU sampling was attempted first but its local collector COM class was not registered (`REGDB_E_CLASSNOTREG`). Windows Performance Recorder and Nsight Systems sampling were also unavailable because this session lacks the Windows **Profile system performance** privilege. The fallback was a user-mode instruction-pointer sampler using `SuspendThread`/`GetThreadContext`, with DbgHelp resolving the optimized MSVC PDB to source lines. It requested a 1 ms interval and collected 20,170 main-thread samples, including 17,996 within `fixed_median_3x3`. Sampling perturbed wall time, so only sample shares are used; approximate time equivalents below apply those shares to the established unprofiled 3.756627 s median.

| Operation | In-filter samples | Share | Baseline-equivalent time |
| --- | ---: | ---: | ---: |
| `std::nth_element` median selection call site | 11,239 | 62.45% | ≈2.346 s |
| input/output flat indexing plus reflection call sites | 3,376 | 18.76% | ≈0.705 s |
| neighborhood value gathering | 2,446 | 13.59% | ≈0.511 s |
| loop bookkeeping plus output store | 935 | 5.20% | ≈0.195 s |

Within the indexing group, the two flat-index call sites account for 15.19% and reflection for 3.57%. Output allocation and the pre-loop shape/bounds validation received zero samples, so both are below this profile's resolution and are not current priorities. The attribution includes operations inlined by MSVC at each call site.

The first two evidence-ranked experiments, fixed median-of-nine selection and reduced flat-address generation, are completed below. Remaining candidates include isolating reflection-index reuse, simplifying the nine neighborhood loads, and only later evaluating parallel decomposition over independent outputs. Every candidate requires correctness validation and before/after measurement.

### Phase A serial CPU optimization experiment 1 — fixed median-of-nine — 2026-09-08

This isolated experiment retained `fixed_median_3x3(...)` as the `std::nth_element` baseline and added `fixed_median_3x3_median9(...)`. Both use the same validation, output allocation, loop order, reflection, flat indexing, and neighborhood gathering. The candidate changes only selection: a fixed 19-comparator compare/swap network places the fifth ordered value at index 4 without averaging, heap allocation, SIMD, or intrinsics. The network was also exhaustively checked over all `9! = 362,880` distinct-rank permutations.

The optimized candidate matched all 65,536 frozen Python-fixture values bit for bit. Its untimed canonical warm-up also matched the baseline across all 47,228,125 outputs before timing. The normal MSVC x64 Release `/O2` configuration remained active. Each implementation received one untimed warm-up and three filter-only timed calls, paired baseline-then-optimized by repetition; internal output allocation was included and file I/O was excluded.

| Implementation | Timed filter runs (ms) | Min / median / max (ms) | Median output rate |
| --- | --- | ---: | ---: |
| Baseline `std::nth_element` | 3515.558, 3625.202, 3449.657 | 3449.657 / 3515.558 / 3625.202 | 13.434 Moutput/s |
| Fixed median-of-nine | 3096.183, 3015.708, 3111.240 | 3015.708 / 3096.183 / 3111.240 | 15.254 Moutput/s |

The fresh median speedup is `1.135449×`, corresponding to an 11.93% runtime reduction from this selector-only change. This is consistent with selection being the dominant sampled hotspot: the fixed network reduces selection work but cannot eliminate it. Without re-profiling, an Amdahl-style estimate based on the earlier 62.45% selection share attributes about 42.6% of the new runtime to work outside selection; that estimate is not a new profile.

### Phase A serial CPU optimization experiment 2 — reduced address generation — 2026-09-08

This experiment retained `fixed_median_3x3_median9(...)` as the current optimized baseline and added `fixed_median_3x3_median9_direct_addressing(...)`. The candidate derives the detector-plane stride, scan-row stride, detector offset, and reusable scan-plane bases directly from the validated C-order dimensions. It removes checked `flat_index(...)` calls from the hot filter body while leaving `flat_index(...)` available elsewhere. Reflection calls, loop order, neighborhood membership and gathering order, the 19-comparator median network, and output allocation are unchanged.

The candidate matched all 65,536 frozen Python-fixture values and all 47,228,125 current-baseline canonical outputs bit for bit. The normal MSVC x64 Release `/O2` configuration remained active. Each implementation received one untimed warm-up and three filter-only timed calls, paired current-baseline-then-candidate by repetition; output allocation was included and file I/O was excluded.

| Implementation | Timed filter runs (ms) | Min / median / max (ms) | Median output rate |
| --- | --- | ---: | ---: |
| Current median-of-nine baseline | 3036.010, 3143.945, 3098.906 | 3036.010 / 3098.906 / 3143.945 | 15.240 Moutput/s |
| Direct-address candidate | 2749.857, 2965.834, 2803.897 | 2749.857 / 2803.897 / 2965.834 | 16.844 Moutput/s |

The fresh median speedup is `1.105214×`, corresponding to a 9.52% runtime reduction. Every candidate run was faster than every baseline run, so this is a measurable improvement worth retaining. Its magnitude is broadly consistent with the earlier approximately 15% flat-index hotspot: the candidate removes repeated checked flattening work but does not remove reflection, neighborhood loads, or the remaining direct address arithmetic. No re-profiling was performed as part of that isolated experiment.

### Phase A optimized serial CPU reprofile — 2026-09-08

The current direct-address median-of-nine implementation was reprofiled on the canonical input using the accepted low-overhead Windows user-mode sampler. The temporary profiling target retained C++17, MSVC x64 Release `/O2`, and normal `AnySuitable` inlining; `/Zi /Zo` and full linker debug information enabled DbgHelp source attribution, with no instrumentation or `/PROFILE`. The sampler requested an approximately 1 ms interval and recorded the main worker thread through `SuspendThread`/`GetThreadContext`. The run produced 13,872 main-thread samples, of which 13,308 were attributed to the current filter source or its median helper.

| Optimized-filter category | Samples | Share | Equivalent contribution from 2.803897 s |
| --- | ---: | ---: | ---: |
| Fixed median-of-nine selection | 9,316 | 70.00% | ≈1.963 s |
| Reflection and boundary handling | 778 | 5.85% | ≈0.164 s |
| Neighborhood load plus inseparable direct-address arithmetic | 2,168 | 16.29% | ≈0.457 s |
| Loop bookkeeping, local initialization, and output store | 1,046 | 7.86% | ≈0.220 s |

The optimized source line combines the remaining address expression with the neighborhood load, so sampling cannot separate those operations reliably. The 564 main-thread samples outside the relevant source include runtime memory movement, file I/O, and benchmark bookkeeping; allocation and validation are therefore not assigned independent filter shares. Profiler-run wall time is not used as performance evidence. The percentages are directional, and the equivalent contributions only apply those shares to the established unprofiled 2.803897 s median.

Compared with the pre-optimization profile, general `std::nth_element` and repeated checked flat indexing have disappeared as categories. Median selection now has a larger relative share because other work was reduced, while address generation and gathering are no longer cleanly separable after compiler optimization. The next roadmap step is a bounded CPU-parallelism experiment over independent outputs, followed by CUDA. Further scalar median-network tuning is deferred because it would be a narrower, microarchitecture-sensitive exercise despite selection remaining dominant.

### Phase A portable OpenMP CPU scaling — 2026-09-08

A separate OpenMP implementation now parallelizes the current direct-address median-of-nine kernel over independent `scan_y` output slabs. Each slab is contiguous, retains the serial `scan_x → detector_y → detector_x` loop order and scalar operations, and is assigned with `schedule(static)`; output writes are disjoint and no nested parallelism is used. The optimized serial implementation remains unchanged and callable. CMake discovers OpenMP with `find_package(OpenMP REQUIRED COMPONENTS CXX)` and links the standard `OpenMP::OpenMP_CXX` target, supporting MSVC on Windows and GCC or Clang with an installed OpenMP runtime on Linux.

The Windows laptop reports 20 OpenMP processors and a 20-thread maximum (13th Gen Intel Core i7-13700H, 14 physical cores and 20 logical processors). The benchmark disabled dynamic teams and requested 1, 2, 4, 8, 16, and 20 threads; the runtime supplied every requested team size. Before timing, the 20-thread output matched all 47,228,125 optimized-serial canonical outputs bit for bit. The OpenMP fixture path also matched all 65,536 frozen Python-reference values and the optimized serial output bit for bit.

Each configuration received one untimed warm-up. Three filter-only timed calls followed in interleaved serial-then-ascending-thread-count order; internal output allocation was included and file I/O was excluded. Speedup uses the fresh same-session optimized-serial median.

| Configuration | Timed filter runs (ms) | Min / median / max (ms) | Median output rate | Speedup | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: |
| Optimized serial | 2789.567, 2788.755, 2817.341 | 2788.755 / 2789.567 / 2817.341 | 16.930 Moutput/s | 1.000× | — |
| OpenMP, 1 thread | 2693.901, 2624.091, 2662.186 | 2624.091 / 2662.186 / 2693.901 | 17.740 Moutput/s | 1.048× | 104.8% |
| OpenMP, 2 threads | 1360.080, 1373.642, 1352.348 | 1352.348 / 1360.080 / 1373.642 | 34.725 Moutput/s | 2.051× | 102.6% |
| OpenMP, 4 threads | 736.454, 776.241, 745.482 | 736.454 / 745.482 / 776.241 | 63.352 Moutput/s | 3.742× | 93.5% |
| OpenMP, 8 threads | 489.667, 493.842, 491.779 | 489.667 / 491.779 / 493.842 | 96.035 Moutput/s | 5.672× | 70.9% |
| OpenMP, 16 threads | 343.042, 356.179, 334.009 | 334.009 / 343.042 / 356.179 | 137.675 Moutput/s | 8.132× | 50.8% |
| OpenMP, 20 threads | 295.810, 332.889, 319.583 | 295.810 / 319.583 / 332.889 | 147.781 Moutput/s | 8.729× | 43.6% |

The one-thread OpenMP result is close to serial and slightly faster in this run, plausibly from compiler/code-layout effects and ordinary measurement variation rather than parallel work. Scaling is strong through four threads, remains useful at eight, and progressively flattens beyond eight; 16 to 20 threads improves the median by only 7.3%. Likely contributors are cache and memory-system pressure, scheduling overhead, and this hybrid CPU's logical-versus-physical-core topology. The measured 8.729× best speedup makes the OpenMP implementation worth retaining as the multicore CPU comparison before CUDA, but it is a single-laptop result. GCC/Clang build validation and scaling on Linux/Lambda remain future work.

### Phase A correctness-first CUDA baseline — 2026-09-09

The first CUDA implementation is deliberately direct: a one-dimensional grid assigns one output element to each thread, with 256 threads per block. Every thread decodes its C-order `(scan_y, scan_x, detector_y, detector_x)` coordinate, gathers the reflected `3 × 3` scan-space neighborhood at fixed detector coordinates, applies the same 19-comparator median-of-nine network, and writes a distinct output. It uses synchronous copies and no tiling, shared memory, streams, warp primitives, intrinsics, or other GPU optimization.

CUDA 12.9.86, MSVC 19.44.35228, CMake 3.31.6, and x64 Release generated an `sm_89` binary for the RTX 4070 Laptop GPU (36 multiprocessors). The public 840-element deterministic fixture matched both its authoritative output and optimized serial C++ bit for bit. The canonical CUDA output likewise matched all 47,228,125 optimized-serial outputs bit for bit before timing.

The AC-powered Windows laptop used one untimed warm-up and three timed calls per path. CPU times use `steady_clock` around the filter call and include output allocation. CUDA events measure synchronous H2D, kernel, and D2H stages; their sum excludes allocation, event setup, and file I/O. Component medians are calculated independently and therefore need not sum to the median of the per-run totals.

| Path / stage | Raw runs (ms) | Min / median / max (ms) | Median output rate |
| --- | --- | ---: | ---: |
| Optimized serial CPU | 2696.036, 2677.485, 2761.642 | 2677.485 / 2696.036 / 2761.642 | 17.518 Moutput/s |
| OpenMP CPU, 20 threads | 332.295, 318.777, 337.153 | 318.777 / 332.295 / 337.153 | 142.127 Moutput/s |
| CUDA H2D | 33.045, 115.121, 115.287 | 33.045 / 115.121 / 115.287 | — |
| CUDA kernel | 36.621, 52.715, 38.191 | 36.621 / 38.191 / 52.715 | 1236.620 Moutput/s |
| CUDA D2H | 107.876, 53.858, 122.051 | 53.858 / 107.876 / 122.051 | — |
| CUDA total path | 177.542, 221.693, 275.530 | 177.542 / 221.693 / 275.530 | 213.034 Moutput/s |

In this original sparse-invocation session, kernel-only speedup was `70.593×` over optimized serial and `8.701×` over OpenMP; the transfer-inclusive path was `12.161×` and `1.499×`, respectively. Transfers occupied 76–86% of each measured total, so this baseline is transfer-sensitive for the canonical one-call workflow. These raw historical results are preserved, but the later timing audit below supersedes 38.191 ms as the baseline for isolated kernel optimization. At this milestone, every thread repeated coordinate decoding and reflection, read nine global values without cooperative reuse, and the one-shot path transferred the full input and output per call. The later sections record the CUDA experiments and completed Python bindings, including persistent device ownership.

### Phase A CUDA baseline profile — 2026-09-09

Nsight Compute 2025.2.1 profiled one post-warm-up canonical kernel launch from the unchanged CUDA 12.9, `sm_89` Release implementation. A focused Basic pass and targeted compute, memory, instruction, scheduler, warp-state, and source-counter sections used kernel replay; an otherwise identical ignored build added CUDA line information only for source attribution. Replay duration is not benchmark evidence. The profile was initially interpreted against the then-established unprofiled 38.191 ms result; the later timing audit below provides the steady-state reference for future kernel experiments.

| Metric | Result |
| --- | ---: |
| SM throughput / SM busy | 83.87% / 83.89% |
| DRAM / peak-memory throughput | 40.57% / 41.80% |
| Achieved occupancy | 95.57% (45.88 of 48 active warps/SM) |
| Registers / local-memory traffic | 40 per thread / none |
| L1/TEX / L2 hit rate | 8.38% / 89.24% |
| Global-load / store sector use | 29.54 / 32.00 bytes per sector |
| Scheduler cycles with no eligible warp | 70.16% |
| Dominant sampled stalls | L1TEX throttle 56.01%; short scoreboard 26.19% |
| Branch efficiency | 100.00% |

The kernel is a mixed instruction/latency workload: its FP64 pipeline reached 83.9% utilization while nine-load dependency chains produced L1TEX queue pressure and short-scoreboard stalls. It is not primarily DRAM-bandwidth, occupancy/register, or divergence limited. Loads were already close to fully coalesced; 8,853,037 excess sectors were 7.0% of the measured total, and Nsight estimated only 1.883% improvement from ideal coalescing.

Source attribution is intentionally conservative because inlined caller/callee rows overlap. The entry/decode region accounted for 123.97 million attributed warp instructions, reflection for 205.33 million, and the fused neighborhood address/load line for 159.39 million plus all excess sectors. Comparator-line samples were dominated by load-related stalls, so they cannot be interpreted as pure median-network time; the output store was fully coalesced and not a meaningful hotspot.

Ranked optimization candidates were: (1) dimension-aware grid/thread mapping that removes repeated flat-index-to-4D decode and simplifies address generation while preserving detector-x coalescing; (2) mapping-aware scan-space shared-memory tiling to reduce the nine global loads and L1TEX dependency pressure; and (3) boundary specialization or precomputed reflection coordinates. The first candidate was subsequently tested, matched all 47,228,125 outputs bit for bit, but produced no measurable gain (`6.538816` versus `6.538336` ms median) and was discarded.

### Phase A CUDA kernel timing audit — 2026-09-09

The original and mapping-experiment paths both placed CUDA events immediately before the unchanged baseline launch and immediately after it, then synchronized on the ending event. Neither kernel interval included `cudaMalloc`/`cudaFree`, H2D/D2H copies, host-output allocation, validation, runtime initialization, or event creation/destruction. The historical harness created buffers and events for every call and placed approximately three seconds of serial/OpenMP work before each timed CUDA invocation; the mapping experiment issued GPU calls continuously. Source and CMake history confirm the same kernel, 256-thread one-dimensional launch, canonical input, CUDA 12.9 `sm_89` Release build, reflection, and median network.

A dedicated kernel benchmark now initializes the runtime, allocates buffers, creates events, and copies the input before measurement. It reuses those resources for five diagnostic warm-up launches, 20 individually recorded steady-state launches, a bounded three-second idle, and five recovery launches. The AC-powered GPU reported P8/210 MHz before the sequence, so the warm-up observations include its natural transition without changing device settings.

```text
warm-up: 7.147200, 6.913024, 6.901760, 6.905632, 6.860608 ms
steady:  6.877184, 6.858752, 6.881280, 6.920192, 6.863872,
         6.906880, 6.886400, 6.895616, 6.872064, 6.907904,
         6.945792, 6.857728, 6.900736, 6.944768, 6.969344,
         6.961152, 6.888448, 6.689792, 6.501376, 6.506496 ms
post-idle: 6.857536, 6.981632, 6.936576, 6.968320, 6.882304 ms
```

The steady-state min/median/max was `6.501376 / 6.888448 / 6.969344` ms, the mean was `6.851789` ms, the population coefficient of variation was `1.878692%`, and throughput was 6856.134 Moutput/s. Output again matched optimized serial C++ bit for bit. A current rerun of the exact historical interleaved harness produced `6.881920, 6.919520, 45.636513` ms, directly demonstrating that the sparse protocol can admit large event-duration outliers; a three-second idle with persistent buffers alone did not reproduce them.

The 38.191 ms historical median was not caused by broader timing boundaries, a different kernel, or a different build. The evidence supports transient scheduling/contention associated with sparse invocations and per-call resource/copy cadence, but does not isolate a single driver or WDDM mechanism or recover the condition that made all three historical samples high; simple clock ramping is insufficient to explain it. Future isolated CUDA experiments therefore use persistent buffers, at least five warm-ups, at least 20 individual event-timed launches, raw-sequence reporting, median/min/max/CV, and separate transfer measurements.

Two further isolated kernel experiments completed the optimization checkpoint. Scan-space shared-memory tiling remained bitwise exact but regressed the median by approximately 1.3%; an interior reflection fast path covered 92.067% of canonical outputs but improved the median by only 0.635%, within 2.34–2.45% run-to-run CV. Both were discarded, as was the correct but neutral dimension-aware mapping experiment. The retained kernel is unchanged.

### Phase A CUDA transfer and residency characterization — 2026-09-09

The dedicated `phase_a_cuda_transfer_characterization` target reuses the retained kernel with persistent device buffers and events. For the canonical input, five complete pageable-path warm-ups preceded 20 CUDA-event-timed runs. Each run measured synchronous H2D, kernel, and D2H separately; the reported total is their per-run sum. Allocation, event creation, validation, host allocation, and file I/O are excluded. The output again matched all 47,228,125 optimized-serial values bit for bit.

```text
H2D:    46.279552, 47.947582, 73.867165, 53.492352, 33.413185, 54.386398, 51.509918, 40.087200, 44.956097, 43.994785, 36.467903, 46.300034, 52.247646, 45.358784, 33.288479, 32.649506, 32.475681, 43.193569, 40.167809, 41.090015 ms
kernel: 6.581984, 6.538336, 6.599136, 6.534816, 6.530912, 6.586944, 6.565472, 6.559552, 6.532480, 6.527072, 6.568512, 6.591424, 6.687040, 6.549760, 6.533472, 6.528736, 6.524192, 6.531008, 6.582624, 6.557568 ms
D2H:    48.917343, 46.924446, 79.606941, 39.838463, 43.254112, 55.531071, 54.039616, 32.850945, 57.024513, 43.983521, 43.037025, 50.029568, 62.753471, 34.107521, 30.511871, 31.364128, 34.032799, 40.164738, 60.745502, 43.202751 ms
total:  101.778880, 101.410364, 160.073242, 99.865630, 83.198209, 116.504413, 112.115006, 79.497697, 108.513090, 94.505378, 86.073441, 102.921025, 121.688158, 86.016065, 70.333822, 70.542370, 73.032672, 89.889315, 107.495935, 90.850335 ms
```

| Pageable stage | Min / median / max (ms) | Mean (ms) | Population CV |
| --- | ---: | ---: | ---: |
| H2D | 32.475681 / 44.956097 / 73.867165 | 44.658683 | 21.354% |
| Kernel | 6.524192 / 6.557568 / 6.687040 | 6.560552 | 0.572% |
| D2H | 30.511871 / 43.983521 / 79.606941 | 46.596017 | 25.998% |
| Total | 70.333822 / 99.865630 / 160.073242 | 97.815252 | 20.862% |

Transfers accounted for a median 93.456% of one-call time and were substantially more variable than the kernel. Fresh CPU medians were 2622.180 ms optimized serial and 336.586 ms at 20 OpenMP threads, so the 99.866 ms pageable CUDA total was `26.257×` faster than serial and `3.370×` faster than OpenMP on this machine.

The residency experiment copied the input once, launched the same filter repeatedly against that unchanged device input while overwriting the same device output, and copied the final output once. Three warm-ups preceded ten timed sequences at each iteration count.

| Resident filter count | Median total (ms) | Effective ms/filter | Median transfer fraction | Speedup vs serial / OpenMP-20 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 88.788420 | 88.788420 | 92.640% | 29.533× / 3.791× |
| 2 | 106.650784 | 53.325392 | 87.750% | 49.173× / 6.312× |
| 5 | 123.348515 | 24.669703 | 73.664% | 106.292× / 13.644× |
| 10 | 170.825756 | 17.082576 | 61.962% | 153.500× / 19.703× |

```text
1 iteration:  123.204061, 95.317442, 82.106210, 103.229632, 97.835805, 73.607809, 85.938048, 76.101058, 72.153438, 88.788420 ms
2 iterations: 117.125441, 107.886592, 107.393633, 77.824194, 86.703265, 89.978367, 84.615970, 106.650784, 92.422240, 108.408609 ms
5 iterations: 152.331779, 132.603550, 105.787903, 110.765663, 115.457245, 122.004318, 106.257183, 131.130623, 153.293919, 123.348515 ms
10 iterations: 170.825756, 160.657249, 210.825409, 215.411423, 148.588642, 159.967167, 247.711288, 156.640541, 183.565376, 138.199429 ms
```

Even ten operations did not make pageable transfers minor. Using the observed approximately 6.50 ms/kernel and 103 ms combined transfer medians, a simple linear estimate requires roughly 64 resident operations before transfers fall below 20%; the corresponding estimate with pinned staging is roughly 37. These are explanatory extrapolations, not measured thresholds.

A separate synchronous transfer-only diagnostic interleaved pageable storage with RAII-managed `cudaMallocHost` buffers; it did not add streams or overlap.

| Transfer diagnostic | Min / median / max (ms) | Population CV |
| --- | ---: | ---: |
| Pageable H2D | 32.388992 / 50.255390 / 105.614014 | 34.633% |
| Pinned H2D | 30.010529 / 30.337120 / 30.743168 | 0.581% |
| Pageable D2H | 31.007168 / 38.627647 / 86.187294 | 32.128% |
| Pinned D2H | 28.763519 / 28.912161 / 29.127712 | 0.302% |
| Pageable H2D + D2H pair | 63.782049 / 93.732064 / 174.655968 | 31.450% |
| Pinned H2D + D2H pair | 58.830846 / 59.254273 / 59.648447 | 0.383% |

Pinned storage made the transfer pair `1.582×` faster, a 36.783% reduction, and far less variable. It is therefore worth considering at the Python boundary, but avoiding repeated transfers through explicit device residency has the larger architectural value.

A bounded synthetic size check compared optimized serial, OpenMP at 4/8/20 threads, and the pageable CUDA total. OpenMP was fastest through 4,096,000 outputs; CUDA was fastest at 10,000,000 and at the 47,228,125-output canonical workload. The observed crossover therefore lies between approximately 4.1 and 10 million outputs for these shapes on this laptop. Because pageable-copy variability was high and shape affects cache behavior, this bracket is directional rather than universal.

These results motivated a simple copying Python call first, followed later by explicit device-resident ownership for repeated or multi-stage GPU work.

### Phase A correctness-first Python interface — 2026-09-09

The opt-in pybind11 `fourdstem_median` module exposes `fixed_median_serial(array)`, `fixed_median_openmp(array, thread_count)`, and, in a CUDA-enabled build, `fixed_median_cuda(array)` plus `cuda_device_info()`. It requires an actual four-dimensional, nonempty, finite, exactly `float64`, C-contiguous NumPy array in `(scan_y, scan_x, detector_y, detector_x)` order; invalid inputs are rejected rather than cast or copied into compliance. Each filter returns a new C-contiguous `float64` array of the same shape and leaves its input unchanged.

This baseline deliberately performs `NumPy → std::vector<double>` before the native call and `std::vector<double> → NumPy` afterward. The CUDA path additionally performs its existing one-shot device allocation, H2D copy, kernel, and D2H copy. The GIL remains released only while native filtering or CUDA device discovery runs; all Python/NumPy validation, allocation, and copying occurs with the GIL held.

The committed public 840-element fixture matched its expected output bit for bit through the serial binding; OpenMP matched serial, and CUDA matched serial. Contract tests also verified output ownership, input immutability, and rejection of non-array, wrong-rank, wrong-dtype, noncontiguous, empty, and nonfinite inputs. On the local canonical input, all 47,228,125 serial, OpenMP-20, and CUDA outputs matched bit for bit.

One warm-up and three Python-call wall-time trials produced:

| Bound path | Raw calls (s) | Min / median / max (s) |
| --- | --- | ---: |
| Optimized serial | 8.871872, 8.901303, 9.016662 | 8.871872 / 8.901303 / 9.016662 |
| OpenMP, 20 threads | 2.400907, 2.306518, 2.392543 | 2.306518 / 2.392543 / 2.400907 |
| One-shot CUDA | 1.059146, 1.192683, 1.118562 | 1.059146 / 1.118562 / 1.192683 |

These are interface-level baselines, not replacements for the established native and kernel timings. They include both host-vector copies; CUDA also includes one-shot native allocation and H2D/D2H transfers. CPU performance in this session was substantially slower than earlier milestone sessions—a same-session native serial check measured 9.909451 s—so the full historical difference cannot be attributed to binding copies. The following isolated experiment measures removal of the CPU host-vector copies directly.

### Phase A direct NumPy-buffer CPU bindings — 2026-09-10

New caller-owned-buffer C++ entry points accept separate `const double*` input and `double*` output storage. The existing vector-returning serial and OpenMP APIs remain callable and delegate to the same factored scan-`y` slab computational core, preserving the direct addressing, reflection, loop order, 19-comparator median, and static OpenMP decomposition without duplicating the algorithm.

The normal Python serial and OpenMP functions now validate the NumPy metadata and finite values, allocate a new same-shape NumPy output, capture both pointers with the GIL held, and release the GIL only while the buffer API runs. Both Python owners remain alive throughout the native call. No NumPy/vector copies occur. The CUDA function is unchanged and still follows `NumPy → vector → one-shot CUDA allocation/H2D/kernel/D2H → vector → NumPy`.

The public 840-element direct serial and OpenMP results matched the expected output bit for bit; input immutability, independent output ownership, and all invalid-input checks still passed. On the canonical workload, both direct paths matched the existing vector paths across all 47,228,125 outputs bit for bit. Private copied CPU functions were exposed only with `PHASE_A_ENABLE_PYTHON_COPY_BENCHMARK=ON` for this experiment and are absent from the normal module.

One warm-up per path preceded five alternating-order Python-call trials on the same input and in the same session:

| Python path | Raw calls (s) | Min / median / max (s) | Direct speedup |
| --- | --- | ---: | ---: |
| Copied serial | 5.922454, 5.500569, 6.059027, 6.166293, 5.291195 | 5.291195 / 5.922454 / 6.166293 | — |
| Direct serial | 5.910830, 5.425366, 5.011050, 5.399885, 5.361272 | 5.011050 / 5.399885 / 5.910830 | `1.096774×` |
| Copied OpenMP-20 | 1.314089, 1.289512, 1.094861, 1.165806, 1.334648 | 1.094861 / 1.289512 / 1.334648 | — |
| Direct OpenMP-20 | 0.805443, 0.747581, 0.739212, 0.777705, 0.780805 | 0.739212 / 0.777705 / 0.805443 | `1.658099×` |

Removing the two copies saved approximately 0.523 s for serial and 0.512 s for OpenMP, reducing median call time by 8.82% and 39.69%, respectively. The retained finite-value scan had a 0.158454 s median (`0.126317 / 0.158454 / 0.167888` s min/median/max), about 2.9% of direct serial time but 20.4% of direct OpenMP time; it is now meaningful for the parallel path but remains required by the public contract.

A fresh three-run native vector-API check measured `4.062016 / 5.218265 / 7.737714` s serial and `0.794332 / 0.827192 / 2.335497` s OpenMP-20 at min/median/max. The direct Python medians were within 3.5% of the native serial median and 6.0% below the variable native OpenMP median, so no stable residual binding penalty beyond validation/output ownership is established. The direct CPU architecture is retained; the following section records the subsequently completed persistent CUDA experiment.

### Phase A persistent CUDA Python ownership — 2026-09-10

The CUDA-enabled module now also exposes `CudaMedianBuffer(array)`. Construction validates the same strict NumPy contract, allocates one device input and one device output through a non-copyable RAII owner, and synchronously uploads directly from the NumPy buffer. The Python input may be released after construction. `filter()` launches the unchanged baseline kernel against the originally uploaded resident input and overwrites the same resident output; repeated calls do not silently chain output back to input. `download()` is valid after filtering and returns a new independently owned NumPy array. The read-only `shape` property reports the recorded four-dimensional shape. No raw device pointer or global persistent state is exposed, and the existing `fixed_median_cuda(array)` one-shot API remains unchanged.

Python validation and NumPy allocation hold the GIL. Device discovery, resident allocation/upload, filtering/synchronization, and download release it. Destruction releases each device allocation exactly once; copying and moving the native owner are disabled. The public fixture passed one and repeated resident filters, download independence, source lifetime/immutability, download-before-filter, and all invalid-input cases. Its one-shot and resident CUDA results matched serial bit for bit. One canonical resident result also matched the established serial/OpenMP/one-shot CUDA reference across all 47,228,125 outputs.

One warm-up per workflow preceded five interleaved, alternating-order Python wall-time trials. A persistent total includes construction/validation/device allocation/upload, the requested synchronous `filter()` calls, NumPy allocation/download, and RAII device release; deletion of the returned NumPy result is outside both endpoints. These workflow times are not kernel-only measurements.

| Python CUDA workflow | Raw total calls (s) | Min / median / max total (s) | Median effective time/filter | Speedup vs repeated one-shot |
| --- | --- | ---: | ---: | ---: |
| Existing one-shot | 0.354173, 0.900909, 0.835481, 0.750910, 0.904223 | 0.354173 / 0.835481 / 0.904223 | 0.835481 s | — |
| Persistent, 1 filter | 0.150375, 0.332802, 0.352179, 0.370421, 0.444610 | 0.150375 / 0.352179 / 0.444610 | 0.352179 s | `2.372×` |
| Persistent, 2 filters | 0.162077, 0.367729, 0.320582, 0.332064, 0.337658 | 0.162077 / 0.332064 / 0.367729 | 0.166032 s | `5.032×` |
| Persistent, 5 filters | 0.177683, 0.179381, 0.413448, 0.353763, 0.436906 | 0.177683 / 0.353763 / 0.436906 | 0.070753 s | `11.808×` |
| Persistent, 10 filters | 0.209745, 0.210104, 0.435821, 0.502478, 0.497004 | 0.209745 / 0.435821 / 0.502478 | 0.043582 s | `19.170×` |

| Resident filters | Median creation/upload | Median kernel sequence | Median download | Median release |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.176977 s | 0.006638 s | 0.140738 s | 0.008448 s |
| 2 | 0.168699 s | 0.013195 s | 0.141638 s | 0.008118 s |
| 5 | 0.177678 s | 0.032845 s | 0.135660 s | 0.007465 s |
| 10 | 0.172477 s | 0.065855 s | 0.148792 s | 0.007011 s |

Even a one-filter persistent workflow was 57.85% shorter than the copied one-shot call in this session; this difference combines avoided NumPy/vector copies with the simpler retained-buffer path rather than isolating one overhead. At ten filters, the resident kernel sequence remained about 6.59 ms per launch and fixed ownership/transfer costs were amortized to a 43.582 ms effective workflow time. Pageable host timings were visibly variable, so these Python totals do not supersede the native CUDA-event results. They do confirm the earlier residency conclusion: explicit device lifetime materially improves repeated Python-facing CUDA work. Phase A should now receive a final consolidated validation/report pass rather than another kernel micro-optimization, then native Phase B can begin.

## Phase B — adaptive median performance target

Phase B preserves and will later accelerate the existing 4Denoise behavior:

```python
fd.HyperData(data).denoise(
    method="adaptive_median_filter",
    domain="real",
    s=3,
    sMax=7,
    return_array=True,
)
```

For each independent detector-coordinate plane, the Python implementation pads the scan image with its global minimum, loops over scan pixels, computes local minimum/median/maximum values, and conditionally grows from `3 × 3` through `5 × 5` to `7 × 7`. This measured repeated neighborhood work makes Phase B the main performance-engineering target.

The reference is [python_reference_adaptive_median_filter.ipynb](python_reference_adaptive_median_filter.ipynb). The initial adaptive benchmark contract requires `scan_y >= 7` and `scan_x >= 7` so a complete maximum-size neighborhood is meaningful away from boundaries.

### Phase B correctness-first native C++ baseline — 2026-09-10

`adaptive_median_s3_smax7(...)` is a deliberately direct C++17 translation of the established finite-`float64` contract. It constructs a global-minimum-padded scan plane for each detector coordinate, extracts each `3 × 3`, `5 × 5`, or `7 × 7` window explicitly, selects the odd-count median with `std::nth_element`, applies strict Stage A and Stage B inequalities, and returns the original center after a failed `7 × 7` Stage A. Input validation, C-order indexing, output allocation, and the validation executable reuse existing native infrastructure without changing any Phase A implementation.

The committed `(7, 7, 1, 4)` public fixture covers retain-center, replace-with-median, equality-driven `3 × 3` to `5 × 5` expansion, `7 × 7` expansion, maximum-window fallback, and global-minimum border padding. All 196 native outputs matched both the generated authoritative array and a fresh public 4Denoise call bit for bit; input remained unchanged. The ignored local `(16, 16, 4, 4)` fixture also matched all 4,096 values exactly.

For a bounded starting measurement only, three Release native validation executions after an untimed setup check took 0.5819, 0.5261, and 0.4950 ms (0.5261 ms median). One in-process warm-up followed by three corresponding public 4Denoise calls took 66.4967, 67.8741, and 65.7886 ms (66.4967 ms median). This small workload is a correctness and operability baseline, not the formal Phase B benchmark or a profiling result. Algorithm optimization, OpenMP, CUDA, and Python binding work remain future steps.

### Phase B native benchmark and CPU profile — 2026-09-10

The native benchmark reconstructs the earlier Python profiling workload exactly from the ignored canonical input: centered slices `[10:74, 0:35, 59:67, 58:66]` produce a C-contiguous `(64, 35, 8, 8)` finite-`float64` subset with 143,360 outputs. Loading the `(85, 35, 127, 125)` source and extracting the subset occur before timing. On AC power, the unchanged MSVC x64 Release `/O2` filter received one warm-up and seven `steady_clock` calls; each timing contains the filter's internal output and temporary allocations but no file I/O, subset preparation, or diagnostic counters.

| Implementation | Raw times | Median | Min–max | Throughput |
| --- | --- | ---: | ---: | ---: |
| native C++ baseline | 15.3791, 15.2987, 15.3929, 15.3165, 15.3034, 15.2896, 15.4852 ms | 15.3165 ms | 15.2896–15.4852 ms | 9.3598 Moutput/s |
| public 4Denoise reference | 16.9017, 17.6682, 17.8093, 17.7459, 18.4166 s | 17.7459 s | 16.9017–18.4166 s | 0.008078 Moutput/s |

The native mean was 15.3522 ms with 0.430% CV. The Python comparison used the identical in-memory subset, one warm-up, and five public `HyperData.denoise(...)` calls. Its approximately `1,159×` native/Python ratio is specific to this workload and these call boundaries, not a full-canonical extrapolation.

The separate diagnostic path reproduced the timed output bit for bit and reported 143,872 median computations. Of 143,360 outputs, 143,104 (99.8214%) finished at `3 × 3`; 256 (0.1786%) expanded to `5 × 5`, the same 256 expanded to `7 × 7`, and all 256 used maximum-window fallback. No pixel finished successfully at `5 × 5` or `7 × 7`. Stage B retained 115,240 centers (80.3850%) and replaced 27,864 values (19.4364%). Input remained unchanged.

The unchanged baseline was then sampled on its main worker thread with the accepted low-overhead Windows instruction-pointer method at approximately 1 ms. An otherwise normal optimized build added `/Zi /Zo` and full linker symbols; 1,000 repeated calls supplied a stable sampling interval. Stack attribution assigned 10,482 of 10,517 samples to native filter source. Profiler wall time is perturbed and is not benchmark evidence.

| Conservative source/runtime group | Attributed samples | Share |
| --- | ---: | ---: |
| `std::nth_element` median selection | 4,629 | 44.16% |
| per-window temporary-vector allocation/teardown | 3,903 | 37.24% |
| window gathering plus min/max | 827 | 7.89% |
| plane minimum/padding/indexing | 592 | 5.65% |
| validation, adaptive control, loops, and output work | 531 | 5.07% |

Optimized unwind/source records map runtime heap calls to `window.reserve(...)` and the implicit scope-end destruction point, so the allocation/teardown group is reliable in aggregate but not separable more precisely. Plane construction and border padding are minor; adaptive expansion is also rare. This differs from the earlier Python `cProfile`, where `np.median` represented 68.75% and median/min/max together 88.28% of profiled time. Native selection remains largest, but per-output heap traffic emerges as a nearly co-dominant cost that Python function-level profiling did not expose.

The evidence ranked the next serial candidates as: (1) replace the 143,872 temporary window-vector allocations with fixed stack storage while retaining `std::nth_element` and gather order—low-to-medium complexity, low semantic risk, and the cleanest first experiment against the 37.24% heap group; (2) specialize the overwhelmingly common nine-value selection while retaining general 25/49-value paths—medium complexity and medium exactness risk, targeting the 44.16% selection group; (3) use direct padded-row addressing for the common `3 × 3` gather/min/max path—medium complexity and low-to-medium indexing risk, targeting the smaller 13.54% gather/preparation/index group. Padding-specific or general adaptive-branch optimization was not justified by this baseline profile.

### Phase B fixed-stack window storage — 2026-09-10

The first isolated adaptive optimization keeps the heap-backed baseline callable and adds `adaptive_median_s3_smax7_stack(...)`. A `std::array<double, 49>` plus a populated-length counter replaces only the temporary window vector; the `3 × 3` to `5 × 5` to `7 × 7` progression, gathering order, min/max work, `std::nth_element`, strict decisions, padding, and fallback are shared unchanged. Output and per-plane padded storage remain normal owned allocations, but the optimized path performs no dynamic allocation per output window.

The 196-value public branch fixture and ignored 4,096-value local fixture matched both Python references and the heap baseline bit for bit. On the representative 143,360-output workload, the heap and stack paths also matched bit for bit, input was unchanged, and all ten diagnostic fields were identical: 143,104 outputs finished at `3 × 3`, 256 reached maximum-window fallback, Stage B retained/replaced 115,240/27,864 values, and 143,872 medians were computed.

Short unpinned trials were visibly bimodal on the hybrid CPU. The authoritative AC-powered A/B sequence therefore pinned the unchanged benchmark process to logical processor 0, alternated implementation order, and used one warm-up plus seven filter-only Release `/O2` calls per path.

| Implementation | Raw times | Median | Min–max | Throughput |
| --- | --- | ---: | ---: | ---: |
| heap-window baseline | 15.5671, 15.7173, 15.4751, 15.6822, 15.5931, 15.7079, 16.6181 ms | 15.6822 ms | 15.4751–16.6181 ms | 9.1416 Moutput/s |
| fixed-stack candidate | 11.0475, 11.3876, 11.2685, 11.1164, 11.0871, 11.2427, 11.1307 ms | 11.1307 ms | 11.0475–11.3876 ms | 12.8797 Moutput/s |

The retained stack path is `1.408914×` faster, removing 29.0234% of median wall time. That is directionally consistent with the original 37.24% sampled heap group, but not expected to match it exactly because sampling is approximate and the remaining work becomes a larger fraction after removal.

The optimized-symbol reprofile collected 6,630 main-thread samples and attributed 6,603 to filter source. Profiler wall time remained perturbed and was not used as benchmark evidence.

| Conservative source/runtime group | Attributed samples | Share |
| --- | ---: | ---: |
| `std::nth_element` median selection | 4,574 | 69.27% |
| window gathering plus min/max | 781 | 11.83% |
| plane preparation and index/address work | 728 | 11.02% |
| adaptive control, loops, and output store | 309 | 4.68% |
| validation and output allocation | 211 | 3.20% |

Per-window allocation/teardown no longer formed a measurable source group; no unexpected replacement hotspot appeared. Because 99.8214% of outputs still terminate at `3 × 3` and median selection is now dominant, the next isolated experiment is a fixed nine-value selector for that common path while retaining `std::nth_element` for the rare `5 × 5` and `7 × 7` windows.

### Phase B specialized nine-value median selection — 2026-09-10

The second isolated adaptive optimization keeps the fixed-stack-plus-`std::nth_element` path callable and adds `adaptive_median_s3_smax7_specialized_3x3(...)`. Only a nine-value window enters the same 19-comparator compare/swap network established in Phase A; 25- and 49-value windows still use `std::nth_element`. Stack storage, gathering order, min/max work, padding, strict Stage A/B decisions, expansion, fallback, indexing, and output behavior are shared unchanged.

Before adaptive integration was accepted, the selector matched a sorted reference for all 362,880 permutations of nine distinct ranks and matched `std::nth_element` for ten representative duplicate-value patterns. The 196-value public fixture, ignored 4,096-value local fixture, and all 143,360 representative outputs then matched the fixed-stack baseline bit for bit. Input remained unchanged and every diagnostic field matched, including the 143,104/256 initial-completion/fallback split, 115,240/27,864 Stage B retain/replace counts, and 143,872 median computations. Phase A validation also remained exact.

The authoritative benchmark used AC power, Release `/O2`, the established logical-processor-0 affinity, alternating implementation order, one warm-up, and seven filter-only calls per implementation.

| Implementation | Raw times | Median | Min–max | Throughput |
| --- | --- | ---: | ---: | ---: |
| fixed stack + `std::nth_element` | 11.3971, 11.4973, 11.3629, 11.5004, 11.3384, 11.3670, 11.4625 ms | 11.3971 ms | 11.3384–11.5004 ms | 12.5786 Moutput/s |
| specialized `3 × 3` selector | 8.6412, 8.6053, 9.1272, 9.6448, 8.6485, 8.6934, 8.7995 ms | 8.6934 ms | 8.6053–9.6448 ms | 16.4907 Moutput/s |

The retained specialized path is `1.311006×` faster, a 23.7227% runtime reduction. The optimized-symbol reprofile collected 5,365 main-thread samples, with one failed observation, and attributed 5,328 to filter source; profiler wall time was perturbed and excluded from benchmark evidence.

| Conservative source/runtime group | Attributed samples | Share |
| --- | ---: | ---: |
| fixed network plus rare larger-window median selection | 3,291 | 61.77% |
| window gathering plus min/max | 757 | 14.21% |
| plane preparation and index/address work | 726 | 13.63% |
| adaptive control, loops, and output store | 358 | 6.72% |
| validation and output allocation | 196 | 3.68% |

Median selection fell by 7.50 percentage points from 69.27% but remains the largest category. The fixed network already removes general-purpose selection from 99.8214% of outputs, so another median micro-optimization is not the next priority. Gathering/min-max and plane/index work now total 27.84%; the next isolated experiment is direct padded-row addressing for the common `3 × 3` gather while preserving all values and decisions.

### Phase B direct padded-row `3 × 3` gathering — 2026-09-11

The third isolated adaptive optimization keeps the specialized stack/network implementation callable and adds an internal A/B candidate without changing the public API. For `3 × 3` only, the candidate computes three padded-row bases once and gathers three consecutive values from each row into the existing stack window in the same row-major order. Min/max comparisons, the 19-comparator selector, all `5 × 5`/`7 × 7` work, padding, adaptive decisions, diagnostics, and output indexing are unchanged.

The 196-value public fixture, ignored 4,096-value local fixture, and all 143,360 representative outputs matched bit for bit. Every diagnostic counter matched, input remained unchanged, and Phase A validation passed. The primary AC-powered, logical-processor-0-controlled sequence alternated implementations after one warm-up and recorded seven filter-only calls each.

| Implementation | Raw times | Median | Min–max | Throughput |
| --- | --- | ---: | ---: | ---: |
| specialized `3 × 3` baseline | 8.9269, 9.2407, 8.9210, 9.2412, 9.3963, 9.1518, 8.7797 ms | 9.1518 ms | 8.7797–9.3963 ms | 15.6647 Moutput/s |
| direct padded-row gather | 8.8916, 8.3730, 8.3097, 8.4542, 8.6883, 9.2289, 8.5972 ms | 8.5972 ms | 8.3097–9.2289 ms | 16.6752 Moutput/s |

The candidate produced a `1.064509×` speedup and 6.0600% median runtime reduction. Two additional controlled sequences repeated the benefit at 8.2231% and 6.6462%, so the change is retained despite individual timing outliers. The optimized-symbol sampler recorded 6,630 main-thread samples with one failed observation and attributed 6,600 to relevant source.

| Conservative source/runtime group | Attributed samples | Share |
| --- | ---: | ---: |
| fixed network plus rare larger-window median selection | 3,971 | 60.17% |
| window gathering plus min/max | 636 | 9.64% |
| plane preparation and index/address work | 1,242 | 18.82% |
| adaptive control, loops, and output store | 444 | 6.73% |
| validation and output allocation | 307 | 4.65% |

The targeted gathering share fell by 4.57 percentage points; unchanged plane preparation and generic indexing became a larger relative share, so percentage totals should not be read as exact component-time accounting. Selection remains dominant but is already a fixed network on the common path. The next experiment is portable OpenMP scaling over independent adaptive work rather than another small scalar micro-optimization.

### Phase B portable OpenMP scaling — 2026-09-11

The retained optimized serial path remains unchanged and callable. `adaptive_median_s3_smax7_openmp(...)` adds an explicit positive thread count and statically schedules the flattened `(detector_y, detector_x)` plane range. Each iteration owns one complete padded scan plane and writes one independent detector-coordinate output plane. The diagnostic path accumulates all ten counters per thread and merges them in thread-index order after the parallel region; no atomics or shared counter updates occur in the pixel path. CMake links the existing portable `OpenMP::OpenMP_CXX` target, with no Windows-specific threading code.

The 196-value public fixture and ignored 4,096-value local fixture matched the optimized serial result and Python reference bit for bit at 1, 2, 4, 8, 16, and 20 threads. The historical 143,360-output subset and the full 47,228,125-output canonical workload also matched serial bit for bit at every count; diagnostic outputs and counters matched, input remained unchanged, zero and negative thread counts were rejected, and Phase A validation passed.

The canonical workload was selected for scaling because the historical subset is too small for stable parallel conclusions. On AC power, the Release `/O2` benchmark disabled dynamic teams, warmed every configuration once, then recorded five filter-only wall-clock calls per configuration with internal output allocation and no file I/O.

| Configuration | Raw times | Median | Throughput | Speedup | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: |
| optimized serial | 3318.3376, 3320.4155, 3302.4510, 3332.9285, 3260.2945 ms | 3318.3376 ms | 14.2325 Moutput/s | `1.000×` | — |
| OpenMP 1 | 3227.0973, 3342.8974, 3271.3952, 3264.2994, 3250.1974 ms | 3264.2994 ms | 14.4681 Moutput/s | `1.017×` | 101.7% |
| OpenMP 2 | 1676.9486, 1731.8583, 1716.0141, 1705.5419, 1740.9183 ms | 1716.0141 ms | 27.5220 Moutput/s | `1.934×` | 96.7% |
| OpenMP 4 | 955.1541, 932.1422, 934.3536, 944.6297, 941.5989 ms | 941.5989 ms | 50.1574 Moutput/s | `3.524×` | 88.1% |
| OpenMP 8 | 612.7705, 604.4137, 590.8537, 578.7943, 588.0546 ms | 590.8537 ms | 79.9320 Moutput/s | `5.616×` | 70.2% |
| OpenMP 16 | 426.5120, 419.9604, 426.5593, 446.4439, 435.5400 ms | 426.5593 ms | 110.7188 Moutput/s | `7.779×` | 48.6% |
| OpenMP 20 | 450.1220, 468.2333, 428.2539, 413.9876, 449.6720 ms | 449.6720 ms | 105.0279 Moutput/s | `7.379×` | 36.9% |

OpenMP-1 differed from serial by only 1.66%, within the observed variability, so no meaningful framework penalty is established. Scaling is near-linear through two threads, remains strong at four, and shows diminishing returns from eight onward. Sixteen threads was best; 20 threads was 5.42% slower, so the final logical-thread region provided no benefit in this run. Static scheduling assigns 992 or 993 detector planes per thread at 16 threads, while only 0.1472% of canonical outputs expand beyond `3 × 3` and extra median computations are 0.2877% of output count. Those measurements do not indicate meaningful adaptive-work imbalance. Cache, memory-system, and hybrid-core effects are plausible explanations for flattening but were not profiled. The OpenMP path is retained, and the next experiment is a correctness-first adaptive CUDA baseline.

### Phase B correctness-first adaptive CUDA baseline — 2026-09-11

The opt-in CUDA baseline preserves the CPU numerical contract in two deliberately direct stages. A one-thread-per-detector-coordinate kernel scans each complete scan plane for its global minimum. A second one-thread-per-output kernel gathers `3 × 3`, `5 × 5`, or `7 × 7` values directly from the C-order input and substitutes that plane minimum for coordinates in the conceptual three-pixel constant border. It applies the same 19-comparator nine-value selector, straightforward larger-window median selection, strict Stage A/B comparisons, and original-center fallback. Both kernels use one-dimensional 256-thread blocks; the canonical launch uses 63 plane-minimum blocks and 184,485 adaptive-filter blocks. No shared memory, persistent application ownership, streams, or GPU-specific adaptive shortcut is present.

The 196-value public Python fixture and ignored 4,096-value local fixture matched the straightforward CPU baseline, retained optimized serial implementation, OpenMP, and CUDA bit for bit. The historical `(64, 35, 8, 8)` subset and all 47,228,125 canonical outputs also matched optimized serial exactly. A separate CUDA outcome byte per output reproduced all ten CPU diagnostic counters exactly; diagnostic collection was excluded from timing. Input remained unchanged. Phase A CPU/CUDA and Phase B CPU-only validation also passed.

Kernel timing uses benchmark-local persistent buffers/events only to prevent allocation and sparse-launch cadence from contaminating the GPU-compute baseline: five warm-up pairs preceded seven individually event-timed launches. The adaptive-filter raw times were 19.757919, 19.807072, 19.804031, 19.746656, 19.768320, 19.658752, and 19.673857 ms, giving a 19.757919 ms median, 19.658752–19.807072 ms range, 0.274% CV, and 2,390.339 Moutput/s. The corresponding plane-minimum median was 1.808224 ms. Against the established 3318.3376 ms serial and 426.5593 ms OpenMP-16 medians, adaptive-kernel-only speedups were `167.950×` and `21.589×`; these intentionally exclude the separate plane-minimum stage and transfers.

The same session also measured seven true one-shot calls with fresh internal device resources after five warm-ups. CUDA-event H2D/plane-minimum/adaptive/D2H medians were 102.079391/1.745600/30.969856/109.048515 ms, while per-run H2D-through-D2H totals were 319.693756, 238.147354, 280.826447, 208.080353, 304.672058, 237.531006, and 264.060638 ms (264.060638 ms median, 208.080353–319.693756 ms, 13.940% CV). That total is `12.567×` faster than established serial and `1.615×` faster than OpenMP-16. An optional native wall interval, which additionally includes validation, host/device allocations, event lifecycle, and cleanup, had a 751.1741 ms median and is not the GPU-path headline.

One-shot pageable copies and sparse launches were visibly variable, and a later read-only device-state check found substantial background desktop GPU activity. The stable back-to-back kernel sequence is therefore the defensible kernel baseline; transfer-inclusive values remain a truthful result for this session rather than a universal estimate. This unoptimized baseline is retained regardless of performance. The next single experiment is focused Nsight Compute profiling of both kernels before proposing any GPU optimization.

## Data

Four data roles are deliberately separate:

- `reference_data/public_synthetic/` is committed and contains the deterministic, nonexperimental default correctness fixture.
- `reference_data/public_adaptive/` is committed and contains the deterministic Phase B branch fixture.
- `../ripple_data_reduced.npy` is the replaceable local upstream experimental source.
- `benchmark_data/median_filter_input.npy` is the local fully prepared, unfiltered experimental performance input.
- the historical fixed/adaptive experimental correctness slices remain local for scientific regression work.

Experimental arrays and their detailed local provenance are excluded from Git because redistribution rights have not been established. [Public benchmark-data guidance](benchmark_data/README.md) explains how to supply a compatible replacement. The output-cleared reference notebooks retain the scientific pipeline and validate that a local source is a nonempty numeric finite 4D array before preparing and checking the canonical input.

If the replaceable source changes only in scan extent, detector calibration can remain compatible. If detector pixels are cropped, shifted, or resampled, the NPY file alone cannot prove that the historical detector-pixel annulus remains scientifically valid. The notebooks reject an annulus that no longer fits and reject invalid ellipse fits; a change in detector sampling still requires scientific review of `r=43.5` and `R=51.5`.

No full fixed or adaptive filtered arrays are stored under `benchmark_data`.

## Correctness fixtures

```text
reference_data/
    public_adaptive/
        generate_fixture.py
        reference_input.npy
        reference_output_python.npy
        README.md
    public_synthetic/
        generate_fixture.py
        reference_input.npy
        reference_output_python.npy
        README.md
```

Both public pairs are finite, C-contiguous `float64` data generated solely from deterministic integer-valued constructions. The Phase A executable checks its retained serial and OpenMP implementations; the Phase B executable checks the straightforward adaptive baseline, fixed-stack baseline, specialized selector, and the selector's exhaustive permutation test. Both compare expected output bit for bit. The larger historical fixtures are not distributed.

## Regenerating the canonical input

For authorized local scientific work, place the experimental source outside version control and run either reference notebook from top to bottom. Each notebook:

1. derives current dimensions and verifies the explicit upstream axis contract;
2. derives current disk alignment with the established call;
3. validates and fits the current ellipse with the established annulus;
4. creates `median_filter_input` before any median operation;
5. writes `benchmark_data/median_filter_input.npy` only when bytes changed;
6. reloads it and checks shape, dtype, contiguity, finite values, and bit-for-bit identity;
7. updates the ignored local dataset manifest with current metadata and hashes.

## Engineering progression

Phase A completed the short infrastructure and learning path: clear C++, validation, benchmarking, CPU profiling and optimization, portable multicore execution, CUDA profiling and controlled experiments, transfer characterization, and Python integration.

Phase B is the main progression: the exact adaptive contract is reproduced in clear C++, its native baseline is profiled, and fixed stack storage plus specialized nine-value selection have addressed the two largest measured costs without changing numerical behavior. Direct padded-row gathering is next; OpenMP and CUDA remain later stages.

## Status

Completed reference and organization work:

- two-phase project scope and broad directory name;
- fixed and adaptive public-API reference notebooks;
- local experimental correctness fixtures plus a public deterministic Phase A fixture;
- deterministic adaptive branch coverage;
- shape-general source validation and bounded current-data subset selection;
- local canonical unfiltered benchmark input with provenance and exact reload verification;
- Phase A `float64` NPY fixture loading, metadata validation, and verified 4D C-order indexing.
- straightforward Phase A fixed `3 × 3` C++ baseline with zero bitwise mismatches across the established local fixture;
- reproducible Phase A canonical-input baseline against the public `HyperData.denoise(...)` path, with loading and filtering measured separately.
- initial optimized-Release CPU profile identifying median selection as the dominant cost, without changing the algorithm.
- isolated fixed median-of-nine serial optimization with exact fixture/canonical equivalence and a measured `1.135449×` median speedup.
- isolated direct-address serial optimization with exact fixture/canonical equivalence and a measured `1.105214×` median speedup over the fresh median-of-nine baseline.
- optimized-serial CPU reprofile showing median selection at 70.00% of relevant samples and selecting CPU parallelism as the next checkpoint.
- portable OpenMP CPU implementation with exact fixture/canonical equivalence and a best measured `8.729×` speedup at 20 threads on the Windows laptop.
- correctness-first CUDA baseline with exact fixture/canonical equivalence, separate event-based transfer/kernel timings, and a measured `1.499×` transfer-inclusive speedup over 20-thread OpenMP on the RTX 4070 Laptop GPU system.
- focused Nsight Compute baseline profile identifying a mixed FP64-instruction and L1TEX/dependency-latency bottleneck and selecting dimension-aware thread/grid mapping as the next controlled experiment.
- CUDA timing audit establishing a reusable-buffer 20-launch kernel protocol and a 6.888448 ms steady-state baseline; the correct dimension-aware mapping candidate produced no measurable speedup and was discarded.
- CUDA optimization checkpoint completing three exact isolated experiments: dimension-aware mapping was neutral, shared-memory tiling regressed approximately 1.3%, and the reflection fast path gained only 0.635% within variability; all candidates were discarded.
- CUDA transfer/residency characterization establishing a 99.865630 ms pageable one-call median, `3.370×` speedup over fresh OpenMP-20, material pinned-transfer benefit, and a directional 4.1–10 million-output CPU/GPU crossover bracket.
- pybind11 serial/OpenMP/optional-CUDA bindings with strict finite, four-dimensional, C-contiguous `float64` validation and exact public/canonical results.
- direct NumPy-buffer CPU/OpenMP paths that removed the two intermediate vector copies and improved copied binding medians by `1.096774×` and `1.658099×`, respectively.
- non-copyable `CudaMedianBuffer` RAII ownership with unchanged-kernel, original-input repeated-call semantics and a measured `19.170×` effective ten-operation speedup over repeated copied one-shot Python calls.
- final public validation covering the regenerated Python reference, retained serial variants, OpenMP, one-shot CUDA, direct Python CPU paths, and persistent CUDA ownership.
- correctness-first Phase B C++17 adaptive median with exact public 196-value and local 4,096-value validation against the established Python behavior.
- reproducible Phase B `(64, 35, 8, 8)` native benchmark, separate branch diagnostics, and optimized-symbol CPU sampling that select temporary-window allocation removal as the first isolated optimization.
- retained Phase B fixed-stack window storage with exact fixture/workload and branch-counter equivalence, a measured `1.408914×` speedup, and a reprofile showing median selection at 69.27%.
- retained Phase B specialized `3 × 3` selection with exhaustive 362,880-permutation verification, exact adaptive equivalence, a measured `1.311006×` speedup, and a reprofile selecting direct-row gathering next.
- retained Phase B direct padded-row gathering with exact fixture/workload/counter equivalence, a repeatable 6–8% runtime reduction, and a reprofile selecting portable CPU parallelism next.
- portable Phase B OpenMP detector-plane decomposition with exact public/local/subset/canonical equivalence and a best measured `7.779×` speedup at 16 threads.
- correctness-first Phase B CUDA with separate global-minimum/adaptive kernels, exact public/local/subset/canonical and diagnostic equivalence, and a 19.757919 ms steady adaptive-kernel median.

Phase A status: **complete**. Phase B correctness-first adaptive CUDA status: **complete**. Next: profile the retained adaptive CUDA kernels with Nsight Compute before considering optimization.

## Remaining TBDs

- Broader cross-platform repetition and problem-size validation beyond the current Windows laptop.
- Native dtype expansion, allocation policy, and nondefault odd adaptive windows remain outside the fixed `float64`, `s=3`, `sMax=7` baseline contract.
- Linux/Lambda GCC-or-Clang OpenMP build validation and multicore scaling.
- CUDA-array interoperability and any asynchronous or multi-GPU design remain outside the Phase A interface scope.
- Scientific validation on replacement datasets whose detector sampling differs from the current source.
