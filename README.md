# CUDA C++ Performance Engineering Portfolio

This is an in-progress portfolio in correctness-first performance engineering for scientific and high-performance computing. **Project 1 Phase A is complete**, having taken a four-dimensional fixed median from an authoritative Python/4Denoise contract through C++17, CPU profiling and optimization, OpenMP, profiled CUDA experiments, transfer/residency analysis, and Python integration. Phase B now has an exact native adaptive-median baseline, three retained serial optimizations, and a portable OpenMP implementation.

The next technical milestone is a correctness-first Phase B adaptive CUDA baseline. Projects 2–4 remain planned work.

## Current status

| Stage | Status | Evidence |
| --- | --- | --- |
| Python/4Denoise behavioral reference | Complete | Fixed and adaptive contracts documented in output-cleared notebooks |
| Correctness-first C++17 fixed median | Complete | Exact finite-`float64` semantics and bitwise validation |
| CPU profiling and serial optimization | Complete | Median-of-nine and direct-address experiments measured independently |
| Portable OpenMP multicore | Complete | Static coarse-grained decomposition with measured Windows scaling |
| Correctness-first CUDA baseline | Complete | Exact validation plus separate kernel and transfer-inclusive measurement |
| CUDA baseline profiling | Complete | Nsight Compute identified a mixed instruction/latency bottleneck rather than DRAM bandwidth, occupancy, or divergence |
| CUDA kernel timing audit | Complete | Reusable-buffer steady-state protocol separates kernel optimization from sparse one-call behavior |
| CUDA optimization checkpoint | Complete | Three exact isolated candidates were neutral, regressive, or within variability and were discarded |
| CUDA transfer/residency characterization | Complete | Pageable, pinned, repeated-residency, and bounded CPU/GPU crossover behavior measured |
| Correctness-first Python interface | Complete | pybind11 serial/OpenMP/optional-CUDA API validated bit for bit |
| Direct NumPy-buffer CPU bindings | Complete | Removed both NumPy/vector copies from serial and OpenMP calls |
| Persistent CUDA Python ownership | Complete | Non-copyable RAII owner amortizes allocation and transfer costs across repeated calls |
| Phase A final consolidation | Complete | Public validation matrix, claims, build options, and repository hygiene audited |
| Phase B adaptive median native baseline | Complete | Public branch fixture and local fixture match the established Python behavior bit for bit |
| Phase B native benchmark and profile | Complete | Reproducible native timing, branch counters, and source-attributed CPU samples established |
| Phase B fixed-stack storage | Complete | Removed per-window heap allocation with exact semantics and a measured `1.409×` speedup |
| Phase B specialized 3×3 selection | Complete | Exhaustively verified 19-comparator network produced a measured `1.311×` speedup |
| Phase B direct-row gathering | Complete | Three padded-row bases reduced common-path gather/index work by a repeatable 6–8% |
| Phase B portable OpenMP scaling | Complete | Detector-plane decomposition reached `7.779×` at 16 threads on the canonical workload |
| Phase B adaptive CUDA baseline | **Next** | Implement the established adaptive contract without premature GPU optimization |
| Projects 2–4 | Planned | Problem statements and validation/performance questions only |

## Project 1 Phase A results

The canonical workload contains 47,228,125 outputs with shape `(85, 35, 127, 125)`. All reported measurements below came from one Windows 11 laptop with an Intel Core i7-13700H (14 physical cores, 20 logical processors) and an RTX 4070 Laptop GPU. CPU measurements used MSVC x64 Release `/O2` and included internal output allocation; CUDA values distinguish kernel execution from the H2D-plus-kernel-plus-D2H path. They are machine-specific results, not universal performance claims.

- The original correctness-first C++ median was 3.756627 s.
- Sampling identified general-purpose `std::nth_element` as the dominant original hotspot.
- A fixed 19-comparator median-of-nine network produced a measured `1.135×` speedup against its fresh same-session baseline.
- Direct reusable C-order addressing produced another measured `1.105×` speedup against its fresh median-of-nine baseline.
- Reprofiling found selection still dominant at approximately 70% of relevant samples.
- Static OpenMP decomposition reached 319.583 ms at 20 threads: `8.729×` versus the fresh 2.789567 s optimized-serial baseline and 147.781 million outputs/s.
- The CUDA baseline matched all 47,228,125 canonical optimized-serial outputs bit for bit. A timing audit established a 6.888 ms reusable-buffer steady-state kernel median; three subsequent kernel candidates did not justify replacing it. In the native transfer study, the pageable one-call median was 99.866 ms versus a fresh 336.586 ms OpenMP-20 median (`3.370×`). Transfers consumed a median 93.456% of that path; pinned staging cut the transfer pair by about one-third, while ten native device-resident operations reduced effective time to 17.083 ms/filter.
- The pybind11 interface exposes optimized serial, explicit-thread-count OpenMP, optional one-shot CUDA, and `CudaMedianBuffer` resident workflows with strict finite, four-dimensional, C-contiguous `float64` validation. Direct NumPy buffers made serial `1.097×` and OpenMP-20 `1.658×` faster than their copied binding baselines. Persistent CUDA reduced median effective Python workflow time from 0.835 s/filter for repeated one-shot calls to 0.0436 s/filter across ten resident calls (`19.170×`) in the measured session. All 47,228,125 canonical outputs remained bit-for-bit exact.

The optimized candidates matched the established Python and preceding C++ outputs bit for bit. Raw runs, benchmark boundaries, full scaling data, profiling caveats, and implementation progression are in the [Project 1 technical report](01_4DSTEM_Median_Filter_Acceleration/README.md).

## Engineering progression

```text
Python reference
  → correctness-first C++17
  → profiled and optimized serial C++
  → portable OpenMP multicore CPU
  → correctness-first CUDA C++  [complete]
  → CUDA baseline profiling     [complete]
  → CUDA optimization checkpoint [complete]
  → transfer/residency analysis  [complete]
  → correctness-first Python interface [complete]
  → direct NumPy-buffer CPU bindings [complete]
  → persistent-GPU Python ownership [complete]
  → final Phase A validation/report [complete]
  → Phase B correctness-first native C++ [complete]
  → Phase B native benchmark and profile [complete]
  → Phase B fixed-stack storage [complete]
  → Phase B specialized nine-value selection [complete]
  → Phase B direct-row gathering [complete]
  → Phase B portable OpenMP scaling [complete]
  → Phase B correctness-first adaptive CUDA [next]
```

The work follows a controlled loop: define numerical behavior, validate exactly, establish a fresh baseline, profile, change one meaningful variable, and remeasure.

## Repository structure

```text
01_4DSTEM_Median_Filter_Acceleration/
    cpp/                         C++17 serial/OpenMP/CUDA source and CMake build
        include/adaptive_median.hpp
                                 correctness-first Phase B native API
        include/adaptive_median_detail.hpp
                                 tested selector primitive and internal experiment hooks
        include/fixed_median_cuda.hpp
                                 CUDA interface, RAII owner, and timing result types
        src/fixed_median_cuda.cu baseline kernel, one-shot path, and resident buffers
        src/python_bindings.cpp  pybind11 CPU/one-shot/resident-CUDA module
        src/cuda_validation.cpp  exact fixture-validation executable
        src/cuda_benchmark.cpp   serial/OpenMP/CUDA benchmark executable
        src/cuda_kernel_benchmark.cpp
                                 reusable-buffer steady-state kernel benchmark
        src/cuda_transfer_characterization.cpp
                                 pageable/pinned, residency, and crossover benchmark
        src/adaptive_median.cpp  retained adaptive baselines and optimized path
        src/adaptive_benchmark.cpp
                                 native timing and diagnostic-counter harness
        src/adaptive_validation.cpp
                                 public exact Phase B validation executable
    reference_data/public_adaptive/
                                 public deterministic adaptive branch fixture
    reference_data/public_synthetic/
                                 public deterministic correctness fixture
    benchmark_data/              instructions for local compatible inputs
    benchmarks/                  Python/4Denoise benchmark harness
    python/validate_bindings.py  public binding smoke and correctness test
    *.ipynb                      output-cleared scientific reference notebooks
02_4DSTEM_3D_Median_Filter/      planned
03_CUDA_KMeans/                  planned
04_CUDA_Matrix_Multiplication/   planned
ROADMAP.md                       staged development plan and completion state
```

## Build and validate the public fixture

The default CPU build requires CMake 3.24+, a C++17 compiler, and a compiler-supported OpenMP runtime. CUDA targets are opt-in and additionally require a compatible CUDA toolkit. The pybind11 extension is separately opt-in and requires pybind11 in the selected Python environment.

Windows with Visual Studio 2022:

```bat
cmake -S 01_4DSTEM_Median_Filter_Acceleration/cpp -B 01_4DSTEM_Median_Filter_Acceleration/cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build 01_4DSTEM_Median_Filter_Acceleration/cpp/build --config Release
01_4DSTEM_Median_Filter_Acceleration\cpp\build\Release\phase_a_fixed_median.exe
```

Linux with GCC or Clang and an installed OpenMP runtime:

```bash
cmake -S 01_4DSTEM_Median_Filter_Acceleration/cpp -B 01_4DSTEM_Median_Filter_Acceleration/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build 01_4DSTEM_Median_Filter_Acceleration/cpp/build
./01_4DSTEM_Median_Filter_Acceleration/cpp/build/phase_a_fixed_median
```

The default executable validates the retained straightforward, median-of-nine, optimized-serial, and OpenMP implementations against a small deterministic synthetic fixture. Linux compilation and scaling are intended portability targets but have not yet been verified on the planned Lambda environment.

The CUDA validation, comparison, stabilized-kernel, and transfer-characterization targets are enabled with `PHASE_A_ENABLE_CUDA=ON`; platform-specific CUDA configuration and reproduction commands are kept in the [Project 1 technical report](01_4DSTEM_Median_Filter_Acceleration/README.md).

## Benchmark with a local input

Experimental arrays are intentionally not distributed. The benchmark accepts another little-endian `float64`, C-contiguous, nonempty four-dimensional NumPy array with axis order `(scan_y, scan_x, detector_y, detector_x)`:

```text
phase_a_baseline_benchmark <compatible-input.npy> --threads 1,2,4,8
```

Use thread counts supported by the current machine. The program performs a full serial-versus-OpenMP bitwise comparison before timing, then runs one warm-up and three filter-only trials per configuration. See [benchmark data guidance](01_4DSTEM_Median_Filter_Acceleration/benchmark_data/README.md).

## Data and reproducibility boundary

The upstream experimental array, prepared canonical benchmark input, and experimental correctness slices remain local and are ignored by Git because redistribution rights are not established. The committed synthetic fixture verifies executable behavior but does not reproduce the documented experimental benchmark. Notebook outputs were cleared before publication to avoid embedding local paths or experimental renderings; their source cells document the reference methodology and require separately obtained 4Denoise software and local data to rerun.

## Development disclosure

AI coding agents were used as development tools. I remained responsible for problem specification, scientific and numerical contracts, algorithm and optimization decisions, code review, debugging, correctness validation, benchmark design, and performance interpretation. This repository does not claim that every generated line was manually authored.

See [ROADMAP.md](ROADMAP.md) for completed and planned milestones and [LEARNING_NOTES.md](LEARNING_NOTES.md) for the evolving technical study outline.
