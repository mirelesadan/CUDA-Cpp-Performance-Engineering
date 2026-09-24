"""Reproducible public workload *generator* for later Project 2 benchmarks.

This module performs no timing and writes no arrays. The fixed PCG64 seeds,
generation order, chunk size, prototype formula, Gaussian scale, and float32
conversion point are part of the dataset specification (version 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GENERATOR_VERSION = 1
GENERATOR_API = "numpy.random.Generator(numpy.random.PCG64(seed))"
CHUNK_ROWS = 16384
PROTOTYPE_MAGNITUDE = 8.0
NOISE_STD = 0.25


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    n: int
    d: int
    k: int
    seed: int


@dataclass(frozen=True)
class GeneratedWorkload:
    samples: np.ndarray
    planted_labels: np.ndarray
    planted_centers: np.ndarray
    spec: WorkloadSpec


WORKLOADS = {
    "profiling": WorkloadSpec("profiling", 65_536, 8, 16, 20_260_924),
    "gpu": WorkloadSpec("gpu", 262_144, 16, 16, 20_260_925),
    "stress_optional": WorkloadSpec("stress_optional", 1_048_576, 32, 32, 20_260_926),
}


def _prototype_centers(k: int, d: int) -> np.ndarray:
    """Repeated cluster-index bits make every feature informative and separated."""
    bit_count = (k - 1).bit_length()
    cluster_bits = np.arange(k, dtype=np.int64)[:, None]
    feature_bits = np.arange(d, dtype=np.int64)[None, :] % bit_count
    bits = (cluster_bits >> feature_bits) & 1
    return np.where(bits == 0, -PROTOTYPE_MAGNITUDE,
                    PROTOTYPE_MAGNITUDE).astype(np.float32)


def generate_workload(spec: WorkloadSpec) -> GeneratedWorkload:
    """Generate bounded, separated synthetic clusters with fixed FP32 seeds.

    PCG64 first shuffles cyclic planted labels. Then, in ascending row chunks,
    it draws float64 Gaussian noise. Prototype plus noise is cast to float32
    once per chunk, then clipped to the input contract. Finally, exact FP32
    prototypes overwrite the deterministic seed rows, and their planted labels
    are corrected to match. The output is not a benchmark result.
    """
    if not (2 <= spec.k <= min(spec.n, 32) and spec.n <= 1 << 20 and
            1 <= spec.d <= 32):
        raise ValueError("workload dimensions are outside the Project 2 contract")
    rng = np.random.Generator(np.random.PCG64(spec.seed))
    centers = _prototype_centers(spec.k, spec.d)
    planted = np.arange(spec.n, dtype=np.int32) % spec.k
    rng.shuffle(planted)
    samples = np.empty((spec.n, spec.d), dtype=np.float32, order="C")
    for begin in range(0, spec.n, CHUNK_ROWS):
        end = min(begin + CHUNK_ROWS, spec.n)
        noise = rng.normal(0.0, NOISE_STD, size=(end - begin, spec.d))
        block = (centers[planted[begin:end]].astype(np.float64) +
                 noise).astype(np.float32)
        np.clip(block, -1024.0, 1024.0, out=block)
        samples[begin:end] = block

    for cluster in range(spec.k):
        seed_row = ((2 * cluster + 1) * spec.n) // (2 * spec.k)
        samples[seed_row] = centers[cluster]
        planted[seed_row] = cluster
    return GeneratedWorkload(samples, planted, centers, spec)
