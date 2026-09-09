# Project 1 — 4D-STEM Median Filter Acceleration

## Objective

Project 1 develops one performance-engineering workflow through two related real-space median filters on 4D-STEM data. Phase A is a controlled fixed-window warm-up; Phase B is the main adaptive-median performance target.

The straightforward Phase A C++ baseline applies the fixed `3 × 3` scan-space median and matches the Python reference bit for bit. It remains available beside the optimized-serial and OpenMP implementations. The full-input Release baseline, CPU profiles, two isolated serial experiments, and Windows multicore scaling are recorded below. CUDA, bindings, and native Phase B work have not begun.

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
        fixed_median.hpp
    src/
        main.cpp
        fixed_median.cpp
    third_party/
        libnpy/
            include/npy.hpp
            LICENSE
            README.md
    build/              # generated; ignored by Git
```

`fixed_median.cpp` contains the correctness-first fixed `3 × 3` implementation: half-sample symmetric reflection on the two scan axes, nine-value median selection, and separate input/output storage. `main.cpp` loads and validates an input/reference pair and compares every `double` by its `uint64_t` bit representation. The public default is a deterministic synthetic fixture; alternate compatible arrays may be supplied explicitly. General shape is discovered at runtime.

The only native dependencies are compiler-supported OpenMP and the vendored, header-only [libnpy](cpp/third_party/libnpy/README.md) `v1.0.1`, pinned to commit `890ea4fcda302a580e633c624c6a63e2a5d422f6` under its MIT license. CMake discovers OpenMP through its standard package target and embeds the source-tree public-fixture paths, so the default run does not depend on the working directory.

From a Visual Studio 2022 x64 Developer Command Prompt, configure and build with:

```bat
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Debug
cmake --build cpp/build --config Release
cpp\build\Release\phase_a_fixed_median.exe
```

CMake reads `CMakeLists.txt`, adds the project and libnpy include directories, applies the C++17 target requirements, and generates Visual Studio build files under `cpp/build/`. MSVC's `cl.exe` compiles the two source files; the linker combines their object files with the required runtime libraries to produce the executable. Debug favors diagnosis, while Release enables the toolchain's normal optimized configuration.

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

## Data

Four data roles are deliberately separate:

- `reference_data/public_synthetic/` is committed and contains the deterministic, nonexperimental default correctness fixture.
- `../ripple_data_reduced.npy` is the replaceable local upstream experimental source.
- `benchmark_data/median_filter_input.npy` is the local fully prepared, unfiltered experimental performance input.
- the historical fixed/adaptive experimental correctness slices remain local for scientific regression work.

Experimental arrays and their detailed local provenance are excluded from Git because redistribution rights have not been established. [Public benchmark-data guidance](benchmark_data/README.md) explains how to supply a compatible replacement. The output-cleared reference notebooks retain the scientific pipeline and validate that a local source is a nonempty numeric finite 4D array before preparing and checking the canonical input.

If the replaceable source changes only in scan extent, detector calibration can remain compatible. If detector pixels are cropped, shifted, or resampled, the NPY file alone cannot prove that the historical detector-pixel annulus remains scientifically valid. The notebooks reject an annulus that no longer fits and reject invalid ellipse fits; a change in detector sampling still requires scientific review of `r=43.5` and `R=51.5`.

No full fixed or adaptive filtered arrays are stored under `benchmark_data`.

## Correctness fixtures

```text
reference_data/
    public_synthetic/
        generate_fixture.py
        reference_input.npy
        reference_output_python.npy
        README.md
```

The public Phase A pair is finite, C-contiguous `float64` data generated solely from integer coordinates. The native executable checks both optimized serial and OpenMP results against its expected output bit for bit. The larger historical Phase A fixture and the Phase B experimental fixture are not distributed. The adaptive notebook retains a deterministic synthetic branch test in source form.

## Regenerating the canonical input

For authorized local scientific work, place the experimental source outside version control and run either reference notebook from top to bottom. Each notebook:

1. derives current dimensions and verifies the explicit upstream axis contract;
2. derives current disk alignment with the established call;
3. validates and fits the current ellipse with the established annulus;
4. creates `median_filter_input` before any median operation;
5. writes `benchmark_data/median_filter_input.npy` only when bytes changed;
6. reloads it and checks shape, dtype, contiguity, finite values, and bit-for-bit identity;
7. updates the ignored local dataset manifest with current metadata and hashes.

## Planned engineering progression

Phase A remains the short infrastructure and learning path: clear C++, validation, benchmarking, CPU profiling, first CUDA, and introductory GPU profiling.

Phase B is the main progression: reproduce the exact adaptive contract in clear C++, validate it, profile and optimize CPU behavior, implement and profile CUDA, optimize from evidence, integrate with Python, and report a reproducible benchmark matrix.

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

Not started: adaptive C++, CUDA, bindings, and final performance claims.

## Remaining TBDs

- Expanded repetition policy and problem-size matrix beyond this initial canonical-input baseline.
- Supported native dtypes and whether nondefault odd adaptive windows belong in the first implementation.
- Linux/Lambda GCC-or-Clang OpenMP build validation and multicore scaling.
- Native dtype expansion and allocation policy beyond the current `float64` fixed-filter path.
- CUDA mapping, tiling, layout/transposition, divergence, selection, transfer, and crossover studies.
- Python binding technology and copy/ownership behavior.
- Scientific validation on replacement datasets whose detector sampling differs from the current source.
