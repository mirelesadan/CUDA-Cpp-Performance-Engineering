"""Authoritative, deliberately clear Project 2 Lloyd K-means reference.

Assignment uses separate NumPy float32 subtract, square, and add operations
for each feature in ascending order. Centroid sums and means use float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


CONTRACT_VERSION = "lloyd-fp32-assignment-fp64-update-v1"
MAX_N = 1 << 20
MAX_D = 32
MAX_K = 32
MAX_MAGNITUDE = 1024.0
ASSIGNMENT_CHUNK_ROWS = 2048
UPDATE_CHUNK_ROWS = 65536


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    update_count: int
    converged: bool


@dataclass(frozen=True)
class Comparison:
    max_centroid_absolute_error: float
    max_centroid_scaled_error: float
    reference_inertia: float
    candidate_inertia: float


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a Boolean")
    return int(value)


def _validate_input(x: object, k: object, max_updates: object) -> tuple[np.ndarray, int, int]:
    if not isinstance(x, np.ndarray):
        raise TypeError("X must be a NumPy ndarray")
    if x.ndim != 2:
        raise ValueError("X must have shape (N, D)")
    if x.dtype != np.dtype("float32"):
        raise TypeError("X must have exact dtype float32")
    if not x.flags.c_contiguous:
        raise ValueError("X must be C-contiguous")

    n, d = x.shape
    clusters = _integer(k, "K")
    updates = _integer(max_updates, "max_updates")
    if not (2 <= clusters <= min(n, MAX_K)):
        raise ValueError("K must satisfy 2 <= K <= min(N, 32)")
    if not (clusters <= n <= MAX_N and 1 <= d <= MAX_D):
        raise ValueError("X must satisfy K <= N <= 2^20 and 1 <= D <= 32")
    if not (1 <= updates <= 100):
        raise ValueError("max_updates must be between 1 and 100")

    for begin in range(0, n, UPDATE_CHUNK_ROWS):
        block = x[begin : begin + UPDATE_CHUNK_ROWS]
        if not np.isfinite(block).all():
            raise ValueError("X must contain only finite values")
        if np.any(np.abs(block) > MAX_MAGNITUDE):
            raise ValueError("every absolute input coordinate must be <= 1024")
    return x, clusters, updates


def initial_row_indices(n: int, k: int) -> np.ndarray:
    """Return deterministic seed rows using integer arithmetic only."""
    n = _integer(n, "N")
    k = _integer(k, "K")
    if not (2 <= k <= min(n, MAX_K) and n <= MAX_N):
        raise ValueError("initial indices require 2 <= K <= min(N, 32), N <= 2^20")
    return np.array([((2 * cluster + 1) * n) // (2 * k)
                     for cluster in range(k)], dtype=np.int64)


def _assign_fp32(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign by explicitly rounded float32 distances; argmin breaks ties low."""
    n, d = x.shape
    k = centroids.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for begin in range(0, n, ASSIGNMENT_CHUNK_ROWS):
        end = min(begin + ASSIGNMENT_CHUNK_ROWS, n)
        points = x[begin:end]
        distances = np.zeros((end - begin, k), dtype=np.float32)
        contribution = np.empty_like(distances)
        for feature in range(d):
            np.subtract(points[:, feature, None],
                        centroids[None, :, feature], out=contribution)
            np.multiply(contribution, contribution, out=contribution)
            np.add(distances, contribution, out=distances)
        labels[begin:end] = np.argmin(distances, axis=1).astype(np.int32)
    return labels


def _update_float64(x: np.ndarray, labels: np.ndarray,
                    centroids: np.ndarray) -> np.ndarray:
    """Accumulate in sample order as float64, then round means to float32."""
    k, d = centroids.shape
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros((k, d), dtype=np.float64)
    for begin in range(0, x.shape[0], UPDATE_CHUNK_ROWS):
        end = min(begin + UPDATE_CHUNK_ROWS, x.shape[0])
        np.add.at(sums, labels[begin:end], x[begin:end].astype(np.float64))
    updated = centroids.copy(order="C")
    nonempty = counts > 0
    updated[nonempty] = (sums[nonempty] /
                         counts[nonempty, None].astype(np.float64)).astype(np.float32)
    return updated


def _fit_with_update_cap(x: np.ndarray, k: int, max_updates: int) -> KMeansResult:
    """Internal reduced-cap hook; the public contract always permits 100 passes."""
    x, k, max_updates = _validate_input(x, k, max_updates)
    centroids = x[initial_row_indices(x.shape[0], k)].copy(order="C")
    labels = _assign_fp32(x, centroids)
    for update_count in range(1, max_updates + 1):
        next_centroids = _update_float64(x, labels, centroids)
        next_labels = _assign_fp32(x, next_centroids)
        converged = bool(np.array_equal(next_labels, labels))
        centroids, labels = next_centroids, next_labels
        if converged:
            return KMeansResult(labels, centroids, update_count, True)
    return KMeansResult(labels, centroids, max_updates, False)


def fit_kmeans(x: np.ndarray, k: int) -> KMeansResult:
    """Run one deterministic Lloyd fit with at most 100 update/reassignment passes."""
    return _fit_with_update_cap(x, k, 100)


def validation_inertia(x: np.ndarray, labels: np.ndarray,
                       centroids: np.ndarray) -> float:
    """FP64 diagnostic on final labels/centroids, separate from assignment.

    Convert operands to float64, subtract and square in float64, sum each
    bounded row chunk in float64, then add chunk totals in input order.
    """
    if labels.shape != (x.shape[0],) or centroids.shape[1:] != (x.shape[1],):
        raise ValueError("inertia operands have incompatible shapes")
    if np.any(labels < 0) or np.any(labels >= centroids.shape[0]):
        raise ValueError("inertia labels must index the centroids")
    total = np.float64(0.0)
    for begin in range(0, x.shape[0], UPDATE_CHUNK_ROWS):
        end = min(begin + UPDATE_CHUNK_ROWS, x.shape[0])
        difference = (x[begin:end].astype(np.float64) -
                      centroids[labels[begin:end]].astype(np.float64))
        np.square(difference, out=difference)
        total += np.sum(difference, dtype=np.float64)
    return float(total)


def compare_results(x: np.ndarray, reference: KMeansResult,
                    candidate: KMeansResult) -> Comparison:
    """Fail clearly on a contract mismatch; return continuous-error metrics."""
    n, d = x.shape
    k = reference.centroids.shape[0]
    if (not isinstance(candidate.labels, np.ndarray) or
            candidate.labels.shape != (n,) or candidate.labels.dtype != np.int32 or
            not candidate.labels.flags.c_contiguous or not candidate.labels.flags.owndata):
        raise AssertionError("candidate labels must be a new C-contiguous int32[N] array")
    if (not isinstance(candidate.centroids, np.ndarray) or
            candidate.centroids.shape != (k, d) or
            candidate.centroids.dtype != np.float32 or
            not candidate.centroids.flags.c_contiguous or
            not candidate.centroids.flags.owndata):
        raise AssertionError("candidate centroids must be a new C-contiguous float32[K,D] array")
    if np.shares_memory(candidate.labels, x) or np.shares_memory(candidate.centroids, x):
        raise AssertionError("candidate outputs must not share input memory")
    if not np.isfinite(candidate.centroids).all():
        raise AssertionError("candidate centroids must be finite")
    if candidate.update_count != reference.update_count or candidate.converged != reference.converged:
        raise AssertionError("candidate update count or convergence flag differs")
    mismatches = np.flatnonzero(candidate.labels != reference.labels)
    if mismatches.size:
        first = int(mismatches[0])
        raise AssertionError(f"{mismatches.size} label mismatches; first at {first}: "
                             f"expected {reference.labels[first]}, got {candidate.labels[first]}")

    scales = np.ones(d, dtype=np.float64)
    for begin in range(0, n, UPDATE_CHUNK_ROWS):
        scales = np.maximum(scales, np.max(np.abs(x[begin:begin + UPDATE_CHUNK_ROWS]), axis=0))
    errors = np.abs(candidate.centroids.astype(np.float64) -
                    reference.centroids.astype(np.float64))
    max_absolute = float(np.max(errors))
    max_scaled = float(np.max(errors / scales[None, :]))
    if max_scaled > 5e-6:
        raise AssertionError(f"centroid scaled error {max_scaled:.9g} exceeds 5e-6")

    reference_inertia = validation_inertia(x, reference.labels, reference.centroids)
    candidate_inertia = validation_inertia(x, candidate.labels, candidate.centroids)
    if not np.isfinite(reference_inertia) or not np.isfinite(candidate_inertia):
        raise AssertionError("validation inertia must be finite")
    inertia_error = abs(candidate_inertia - reference_inertia)
    if inertia_error > 2e-5 * max(1.0, abs(reference_inertia)):
        raise AssertionError(f"inertia error {inertia_error:.9g} exceeds limit")
    return Comparison(max_absolute, max_scaled, reference_inertia, candidate_inertia)
