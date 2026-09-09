# Project 2 — 4D-STEM 3D Median Filter

## Objective

Apply the established performance-engineering workflow to a more computationally and memory-intensive three-dimensional median/neighborhood operation derived from 4D-STEM data.

## Why this project exists

This project raises the difficulty after Project 1 by introducing a larger neighborhood and more demanding high-dimensional memory behavior. It should demonstrate deeper reasoning about data layout, cache effects, GPU memory access, thread organization, and scaling rather than simply repeating the 2D implementation.

## Problem definition

The current concept is to reorganize, unfold, or reshape 4D-STEM information into a long three-dimensional representation and apply a local 3D median/neighborhood filter. The exact transformation, neighborhood definition, scientific intent, axis order, boundary behavior, dtype, and shapes are **TBD** and must be established from the intended research method before implementation.

## Planned workflow

Define and validate the scientific reference; profile it; implement a straightforward modern C++ version; validate, benchmark, profile, and optimize CPU execution; implement and validate CUDA; profile and optimize memory access and thread organization; add an appropriate Python interface; and document scaling and tradeoffs.

## Primary learning goals

- Reasoning about higher-dimensional indexing, layout, and working sets.
- Neighborhood-size growth and selection/computation cost.
- Cache locality, memory traffic, allocation, and reuse on CPU.
- GPU mapping, coalescing, reuse, synchronization, and resource constraints.
- Scaling analysis for increasingly large three-dimensional volumes.

## Planned performance questions

- How do volume dimensions and neighborhood size affect runtime and memory traffic?
- Which measured costs come from transformation/unfolding versus filtering?
- How do layout and traversal choices affect CPU caches and GPU coalescing?
- Is the workload bandwidth-bound, compute-bound, or limited by another resource?
- Which intermediate data should be materialized, streamed, or avoided?
- At what sizes does CUDA provide an end-to-end benefit?
- How does the optimized approach differ fundamentally from Project 1?

## Validation strategy

Correctness criteria will be defined before optimization after the transformation and scientific operation are specified. Trusted small cases and representative research cases will be compared against the reference, with tolerance and edge behavior documented explicitly.

## Benchmarking strategy

Comparisons will cover reference, straightforward C++, optimized CPU, initial CUDA, and optimized CUDA implementations over multiple later-defined volume and neighborhood sizes. Transformation, allocation, transfer, kernel, and end-to-end costs will be separated when they answer meaningful questions.

## Status

Planned

## Open questions / TBD

- **TBD:** Exact scientific transformation/unfolding operation.
- **TBD:** Definition and shape of the 3D neighborhood.
- **TBD:** Axis order, layout, strides, dtype, and boundary behavior.
- **TBD:** Whether transformed data is materialized or accessed through an alternative representation.
- **TBD:** Reference implementation and correctness criteria.
- **TBD:** Representative test and benchmark datasets.
- **TBD:** Hardware, toolchain, profiling tools, and Python binding approach.
