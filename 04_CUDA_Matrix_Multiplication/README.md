# Project 3 — CUDA Matrix Multiplication (FP32 GEMM)

## Status and purpose

**Proposed contract; awaiting human approval.** No reference, fixtures, implementation, or benchmark exists yet. The legacy `04_CUDA_Matrix_Multiplication` directory remains Project 3; no files are being moved. After approval, the first task is the authoritative reference and deterministic correctness fixtures, not an optimized kernel.

Study a dense AI/HPC primitive through arithmetic intensity, coalescing, thread/block mapping, shared-memory reuse, register tiling, resource tradeoffs, numerical error, and comparison with mature GEMM. Large GEMM offers compute-heavy work; small and naive implementations need not be compute-bound. Beating cuBLAS is not the goal.

## Proposed Phase A operation and data contract

`C = A @ B`, with `A[M,K]`, `B[K,N]`, and `C[M,N]`. Shape tuples below are always **`(M,N,K)`**, not input-axis order.

- Dense, row-major, native-endian IEEE binary32 (`float32`), exactly two dimensions and C-contiguous; `1 <= M,N,K <= 4096`, matching inner dimensions, finite entries, and `|A[i,k]|, |B[k,j]| <= 1`.
- A Python reference must accept actual NumPy ndarrays and reject invalid dtype/layout/shape/value inputs without silent casting or contiguity repair. Native implementations must enforce the equivalent buffer/dimension contract with checked size arithmetic.
- Return a new, independently owned, C-contiguous `float32[M,N]` result; neither input changes. Output must be finite. Signed-zero sign is not an equivalence requirement.
- No transposed-input flags, batching, sparse matrices, bias, activation, user alpha/beta, accumulation into existing C, or general tensor framework. Library-internal `alpha=1,beta=0` only implements this operation.

The narrow SGEMM-like scope is deliberate: reuse and reduction behavior, rather than a broad API, are the engineering question. The bound prevents overflow even at the largest K: `sum_k |A[i,k] B[k,j]| <= K <= 4096`.

## Proposed arithmetic and acceptance rule

The mathematical target is `C*[i,j] = sum_{k=0}^{K-1} A[i,k] B[k,j]`. A straightforward C++ baseline will use ascending-k, separate FP32 multiplication/addition. Optimized **classical dot-product** implementations may reorder/block/tree-reduce and use FP32 FMA; they need not reproduce serial bits. Operands/output and accumulator precision remain FP32; FMA may retain an unrounded internal product. Strassen-style transformed algorithms, higher-precision custom accumulation, TF32, FP16/BF16, and Tensor Core arithmetic are outside Phase A.

Use round-to-nearest, ties-to-even, with gradual underflow (no flush-to-zero/denormals-are-zero and no fast math). Record relevant CPU/CUDA arithmetic flags. Tiny exact fixtures must verify subnormal preservation; the error inequality alone cannot enforce it. NVIDIA explains the relevant [floating-point compiler settings](https://docs.nvidia.com/cuda/floating-point/index.html#compiler-flags).

The authoritative high-accuracy oracle is `R = A.astype(float64) @ B.astype(float64)`, without rounding R back to FP32. Conversions and FP64 products of binary32 inputs are exact; FP64 accumulation is not exact and receives its own bound. Record NumPy/BLAS versions. Define, in mathematical arithmetic:

```text
u32 = 2^-24; u64 = 2^-53; eta32 = 2^-149
gamma_K(u) = K*u / (1 - K*u)
S_hat = abs(A.astype(float64)) @ abs(B.astype(float64))
S_upper[i,j] = S_hat[i,j] / (1 - gamma_K(u64))
E[i,j] = abs(float64(C[i,j]) - R[i,j])
T[i,j] = (gamma_K(u32) + gamma_K(u64)) * S_upper[i,j]
         + (2*K + 1) * eta32
accept only if every E[i,j] <= T[i,j], plus structural/exact-case checks
```

Evaluate `S_upper` and T conservatively, using outward rounding or exact/high-precision bound arithmetic; resolve borderline comparisons without rounding a mathematical upper bound downward. The formulas specify the gate, not an already implemented validator. Zero-input products must return numerical zero exactly; identity and exactly representable K=1 cases, including subnormals, have exact-value checks in addition to this general gate.

Rationale: each term traverses at most K rounding stages in a classical dot-product reduction, giving the standard `gamma_K * sum(abs(products))` bound; FMA or balanced sums may improve actual error. Add the FP64-oracle bound and a conservative gradual-underflow allowance for at most `2K-1` rounded operations. This proposal applies the [dot-product analysis of Castaldo, Whaley and Chronopoulos](https://www.cs.utsa.edu/faculty/atc/pub/J42.pdf) and the [rounding/underflow model described by Rump](https://www.tuhh.de/ti3/paper/rump/Ru10b.pdf). At K=4096, `gamma_K(u32)` is about `2.442e-4` (0.02442% of the absolute-product sum), a worst-case ceiling, not expected observed error or an arbitrary `allclose` tolerance. Cancellation does not cause division by a nearly zero R. Do not loosen the gate to pass an optimization.

Report these diagnostics; none replaces the per-entry acceptance test:

- Maximum absolute error: `max(E)`.
- Maximum product-scaled error: `max(E / S_upper)`.
- Maximum budget utilization: `rho = max(E / T)`; acceptance requires `rho <= 1`.
- Relative Frobenius error: `||C64-R||_F / ||R||_F`; also report the absolute numerator.

For diagnostic divisions, define `0/0 = 0` and positive/zero as infinity. Report failures with shape, index, candidate/reference values, E and T. Check input immutability, output ownership/dtype/shape/layout, deterministic fixture reproduction, and invalid inputs separately. Bitwise agreement between different reduction orders is not promised.

## Proposed deterministic correctness suite

Use tuples `(M,N,K)`:

`(1,1,1), (3,3,3), (3,5,2), (7,9,5), (31,17,13), (33,65,29), (65,3,17), (3,65,17), (17,19,1)`.

These cover scalar/K=1, all-unequal rectangular dimensions, odd/non-multiple edges, tall-skinny and short-wide products. Reuse small shapes for zero-A/zero-B, identity, positive/negative values, cancellation/alternating signs, duplicate values, and exact subnormal preservation. Include rejected empty, wrong-rank, mismatched-K, noncontiguous, non-float32, nonfinite and out-of-range inputs. The later CPU/CUDA paths must pass the same fixtures.

For random cases propose NumPy `Generator(PCG64(20261005 + case_index))`: draw A then B in C order from FP64 uniform `[-1,1)`, cast each once to native FP32/C order. Freeze case ordering, generator version, seed and NumPy version when fixtures are created; explicit structured cases may include endpoints -1/+1. No matrices are generated in this proposal.

## Proposed performance suite and measurement

Squares `M=N=K=128,256,512,1024,2048`; rectangles `(2048,256,1024)` and `(256,2048,1024)`. This spans overhead-sensitive, medium and compute-heavy work on the RTX 4070 Laptop 8 GB. Device A/B/C storage is `4*(M*K+K*N+M*N)` bytes: 48 MiB for the largest square and 11 MiB per rectangle, excluding workspace/reference storage. Defer 4096 from the initial suite: its 192 MiB device footprint is practical, but its 137.44 GFLOP and serial/naive runtime have not been assessed. FP64 oracle/error arrays consume additional host memory and are outside timers.

Propose Release builds in one AC-powered session, with recorded clocks/state where available but no power tuning. CPU: one warm-up and at least seven wall-clock calls. GPU: persistent device buffers/events, five warm-ups, twenty individual event timings, synchronization after each. Generate/validate inputs outside timers; retain raw runs and median/min/max/CV, no cherry-picked fastest run. If small-matrix cadence/timer resolution needs a different protocol, disclose and validate it before comparison.

Report CPU/native wall time (output allocation included), CUDA kernel-only time (allocations/initial H2D excluded), and a separate transfer-inclusive `H2D(A+B) + GEMM + D2H(C)` path with persistent allocations and complete synchronization. A later allocating one-shot interface must be labeled separately. Profiler replay durations are not performance results. Timing follows [NVIDIA's event and host-synchronization guidance](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#timing).

Conventional useful work is `F = 2*M*N*K` FLOPs, counting multiply-add as two and excluding indexing/memory operations. `GFLOP/s = F/(seconds*10^9)`. For equal shape and timing boundary, fraction of cuBLAS performance is `GFLOP/s_custom / GFLOP/s_cuBLAS = t_cuBLAS/t_custom`. Report kernel-only and transfer-inclusive CPU/GPU crossover separately; the library fraction is not a fraction of Tensor Core hardware peak.

## Trusted references and fair cuBLAS policy

NumPy FP64 is the correctness oracle; straightforward serial C++ FP32 is the inspectable native baseline; **cuBLAS SGEMM is the primary mature GPU comparator**. Add a NumPy/CPU BLAS or framework comparator only for a concrete additional question, not quantity.

Propose checked `cublasSgemm` with `CUBLAS_PEDANTIC_MATH`, `alpha=1`, `beta=0`, FP32 A/B/C, and no TF32/Tensor Cores/reduced-precision emulation. Defaults do not explicitly exclude Tensor Cores; [cuBLAS documents pedantic modes as the exclusion](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#tensor-core-usage). `cublasGemmEx` with `CUDA_R_32F` and `CUBLAS_COMPUTE_32F_PEDANTIC` is an explicit alternative, not an additional comparator. Verify math-mode configuration and exact tiny/edge fixtures later; library subnormal behavior is not claimed tested here.

For row-major buffers, interpret `C^T = B^T A^T`: formal cuBLAS `(m,n,k)=(N,M,K)`, operands B then A, `OP_N/OP_N`, leading dimensions `N,K,N`. No physical transpose or transpose copy. Keep the same inputs, arithmetic policy, device, shapes, warm-up, repetition, synchronization and timing boundaries. Handle/events/setup precede kernel timing; disclose allocation and H2D/D2H inclusion separately. Record toolkit/library/driver versions, compiler flags, queried math mode and any TF32/emulation environment overrides. Compare measured quality under the same gate as custom kernels.

## Evidence-driven progression and deferred design

After approval: freeze reference/fixtures -> serial C++ -> correctness-first CUDA -> non-multiple validation and kernel/transfer timing -> focused Nsight Compute -> coalesced-access experiment only if needed -> shared-memory reuse experiment -> reprofile -> register/block tiling or another measured change -> same-mode cuBLAS comparison -> size/crossover characterization -> Python interface only if it adds value. Preserve baselines, isolate experiments and reject unsupported complexity. Do not force every optimization, an optimized CPU detour, or a predetermined final kernel architecture.

Defer exact block/tile/shared/register shapes, vector loads, double buffering, warp MMA, WMMA, CUTLASS-like designs and persistent kernels until evidence supports them. Phase A studies **conventional CUDA-core FP32**. Optional later Phase B may study TF32/mixed-precision/Tensor Cores only under a separately approved numerical/performance contract; it is not silently comparable to Phase A.

## Portfolio distinction and remaining decisions

Project 1: scientific neighborhood processing and exact semantics. Project 2: iterative assignment/reduction/convergence and explicit residency. Project 3: dense arithmetic, reuse, arithmetic intensity and a mature-library ceiling. The stories are complementary, not interchangeable CUDA demos.

Outstanding: human approval of this proposed scope/gate/suite, then reproducible fixtures; actual library subnormal behavior, hardware/toolchain records and implementation choices will be validated in later tasks. Linux portability and optional Python/Tensor Core extensions are not claimed complete. No Project 3 implementation has begun.
