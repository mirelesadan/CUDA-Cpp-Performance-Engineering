# Learning Notes

This file is an evolving knowledge notebook for concepts encountered while building the portfolio. Entries begin as short prompts and should be expanded with explanations, experiments, profiler evidence, mistakes, and project-specific examples as they are learned.

## A. C++ Foundations

### Compilation model
Track how source files, headers, translation units, object files, and executables relate.

### Compiler vs linker
Record which responsibilities and errors belong to compilation and which belong to linking.

### Source vs binary portability
Note what can move across platforms or toolchains and what must be rebuilt.

### Variables and types
Study type choice, representation, conversions, size, precision, and performance consequences.

### Functions
Track interfaces, value/reference parameters, return values, inlining, and separation of responsibilities.

### References
Record aliasing, mutation, lifetime, and when references communicate non-ownership.

### Pointers
Study addresses, indirection, nullability, pointer arithmetic, ownership distinctions, and safe use.

### Arrays
Connect indexing, extents, strides, contiguity, and multidimensional layout.

### Memory lifetime
Track construction, destruction, scope, ownership, borrowing, and invalidation.

### Stack vs heap
Record practical differences in lifetime, size, allocation cost, and access patterns without treating either as universally better.

### RAII
Study how object lifetime can manage memory, files, handles, and other resources safely.

### Const correctness
Use const to express and enforce whether data or interfaces may mutate state.

### Classes and structs where relevant
Learn how to group data and behavior without adding abstraction that a project does not need.

### Templates where relevant
Record how compile-time generic code can support types and dimensions, including costs and diagnostics.

### STL containers
Study ownership, layout, iterator invalidation, allocation behavior, and appropriate container selection.

### Iterators and ranges where relevant
Learn composable traversal while checking clarity and performance implications.

### Exceptions and error handling
Define failure contracts and understand how error strategies interact with library boundaries and CUDA calls.

### Move semantics where relevant
Track value categories, ownership transfer, and when moves reduce expensive copies.

## B. C++ Performance

### Optimization levels (`-O0`, `-O1`, `-O2`, `-O3`)
Compare what optimization levels enable and measure their effects under the actual compiler and workload.

### Compiler-generated assembly
Inspect generated instructions only when they help answer a concrete optimization or vectorization question.

### Cache hierarchy
Connect cache levels, cache lines, reuse, and working-set size to observed performance.

### Locality
Distinguish spatial and temporal locality and identify how algorithms create or lose each.

### Contiguous data
Record how layout, strides, and traversal order influence cache and hardware prefetch behavior.

### Memory allocation and copies
Measure allocation and data-movement costs and identify avoidable temporary storage.

### SIMD and vectorization
Learn when loops vectorize, how to verify it, and which dependencies or layouts inhibit it.

### Multithreading
Study work partitioning, synchronization, scheduling, false sharing, and scaling limits.

### Profiling
Use profiles to locate costs and validate hypotheses rather than optimize by intuition alone.

### Benchmarking methodology
Track warm-up, repetitions, timers, variance, compiler settings, workload equivalence, and reproducibility.

## C. CUDA Foundations

### Host vs device
Understand the execution contexts, accessible memory, and responsibilities on each side.

### Kernels
Learn how GPU functions are launched, parameterized, synchronized, checked, and validated.

### Threads
Connect per-thread work and indices to the problem domain.

### Blocks
Study cooperation and resource sharing among threads in a block.

### Grids
Learn how blocks cover complete workloads and handle bounds.

### Warps
Understand warp execution as a foundation for reasoning about scheduling and divergence.

### SIMT
Record how single-instruction, multiple-thread execution differs from a purely scalar mental model.

### Device memory
Map CUDA memory allocation, ownership, lifetime, and accessibility.

### Global memory
Study capacity, latency, access patterns, and transaction efficiency.

### Shared memory
Learn how block-local storage can enable reuse and cooperation, including its constraints.

### Registers
Track per-thread storage, register pressure, and spilling when profiling makes them relevant.

### Synchronization
Understand the scope, necessity, cost, and correctness implications of synchronization.

### Memory transfers
Measure host/device movement and understand synchronous, asynchronous, and pinned-memory concepts when needed.

## D. CUDA Performance

### Memory coalescing
Relate neighboring thread addresses to memory transactions and effective bandwidth.

### Occupancy
Treat occupancy as a resource and latency-hiding metric, not an automatic performance objective.

### Divergence
Identify data-dependent control flow within warps and measure whether it matters.

### Shared-memory tiling
Study when tiles increase reuse and how tile shape, synchronization, and bank behavior affect results.

### Arithmetic intensity
Relate useful computation to data movement and use it to reason about potential limits.

### Bandwidth-bound vs compute-bound workloads
Use evidence to determine the limiting resource and choose optimizations accordingly.

### Kernel-launch overhead
Measure fixed launch costs and their importance for small or fragmented workloads.

### Transfer overhead
Include data movement in end-to-end reasoning and identify when reuse can amortize it.

### Profiling with NVIDIA tools
Build a question-driven workflow using the appropriate NVIDIA system and kernel profilers.

### Scaling
Track throughput, latency, resource limits, and crossover behavior as workload dimensions change.

## E. Python Integration

### Python extension concepts
Understand how compiled modules expose native functions and represent errors and types.

### pybind11 or alternatives
Evaluate binding options against project needs before choosing one; the decision is **TBD**.

### NumPy array interoperability
Study dtype, shape, strides, contiguity, writable state, and buffer access.

### Ownership and lifetime issues
Ensure objects and buffers remain valid across the Python/native boundary.

### Minimizing copies
Make copy behavior explicit and pursue zero-copy paths only when safe and useful.

### Exposing CPU and CUDA implementations cleanly
Design a stable user-facing interface while keeping backend selection and synchronization understandable.

## F. AI/HPC Concepts

### Matrix multiplication and GEMM
Study the operation, dimension conventions, reuse, computational cost, and why GEMM is heavily optimized.

### Tensors
Connect tensor shapes, strides, layouts, contractions, and batching to lower-level kernels.

### Convolution vs neighborhood filters
Compare access patterns and computation without incorrectly treating median filtering as a linear convolution.

### K-means computational structure
Separate assignment, reduction/update, convergence, initialization, and data-movement costs.

### Why GPUs accelerate parallel workloads
Relate throughput hardware to parallelism, regularity, locality, arithmetic intensity, and sufficient work.

### What cuBLAS provides
Learn the role and interface of NVIDIA's optimized dense linear-algebra library and use it as a reference where appropriate.

### Python AI frameworks and lower-level C++/CUDA libraries
Trace how high-level operations can dispatch to compiled kernels and mature backend libraries.
