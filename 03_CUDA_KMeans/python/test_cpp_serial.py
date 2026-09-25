"""Validate the Project 2 serial C++ baseline and optionally time its public workload.

From the repository root, after building the Release target:
    python -B 03_CUDA_KMeans/python/test_cpp_serial.py PATH_TO_EXE
    python -B 03_CUDA_KMeans/python/test_cpp_serial.py PATH_TO_EXE --benchmark

Temporary raw little-endian files only bridge NumPy and the standalone C++
executable. They are not part of either fit's timed region or committed data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from benchmark_data import WORKLOADS, generate_workload
from reference import (KMeansResult, _fit_with_update_cap, compare_results,
                       fit_kmeans, initial_row_indices, validation_inertia)
from test_reference import public_cases


def run_native(executable: Path, points: np.ndarray, k: int, directory: Path,
               stem: str, *, test_cap: int | None = None,
               timed_runs: int = 0) -> tuple[KMeansResult, dict]:
    if sys.byteorder != "little":
        raise RuntimeError("raw validation bridge requires a little-endian host")
    if points.dtype != np.float32 or not points.flags.c_contiguous:
        raise ValueError("bridge input must be C-contiguous float32")
    n, d = points.shape
    input_path = directory / f"{stem}.input.f32"
    labels_path = directory / f"{stem}.labels.i32"
    centroids_path = directory / f"{stem}.centroids.f32"
    points.tofile(input_path)
    before = points.view(np.uint32).copy()
    args = [str(executable), str(input_path), str(n), str(d), str(k),
            str(labels_path), str(centroids_path)]
    if test_cap is not None:
        args.extend(["--test-update-cap", str(test_cap)])
    if timed_runs:
        args.extend(["--timed-runs", str(timed_runs)])
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    metadata = json.loads(completed.stdout)
    np.testing.assert_array_equal(points.view(np.uint32), before)
    if input_path.stat().st_size != points.nbytes:
        raise AssertionError("C++ driver changed its input file")
    if labels_path.stat().st_size != n * np.dtype("int32").itemsize:
        raise AssertionError("native label file has wrong size")
    if centroids_path.stat().st_size != k * d * np.dtype("float32").itemsize:
        raise AssertionError("native centroid file has wrong size")
    labels = np.empty(n, dtype=np.int32)
    labels[:] = np.fromfile(labels_path, dtype=np.int32)
    centroids = np.empty((k, d), dtype=np.float32, order="C")
    centroids[:] = np.fromfile(centroids_path, dtype=np.float32).reshape(k, d)
    return (KMeansResult(labels, centroids, metadata["update_count"],
                         metadata["converged"]), metadata)


def validate_fixtures(executable: Path, directory: Path) -> None:
    total = 0
    centroid_bit_mismatches = 0
    for case in public_cases():
        points = np.array(case["points"], dtype=np.float32, order="C")
        before = points.view(np.uint32).copy()
        k = case["k"]
        np.testing.assert_array_equal(initial_row_indices(len(points), k),
                                      case["initial_rows"])
        cap = case["max_updates"]
        reference = (fit_kmeans(points, k) if cap == 100 else
                     _fit_with_update_cap(points, k, cap))
        expected_labels = np.array(case["expected_labels"], dtype=np.int32)
        expected_centroids = np.array(case["expected_centroids"], dtype=np.float32)
        np.testing.assert_array_equal(reference.labels, expected_labels)
        np.testing.assert_array_equal(reference.centroids.view(np.uint32),
                                      expected_centroids.view(np.uint32))
        native, _ = run_native(executable, points, k, directory, case["name"],
                               test_cap=cap if cap != 100 else None)
        second, _ = run_native(executable, points, k, directory,
                               case["name"] + ".repeat",
                               test_cap=cap if cap != 100 else None)
        compare_results(points, reference, native)
        np.testing.assert_array_equal(native.labels, second.labels)
        np.testing.assert_array_equal(native.centroids.view(np.uint32),
                                      second.centroids.view(np.uint32))
        np.testing.assert_array_equal(points.view(np.uint32), before)
        if native.update_count != case["expected_update_count"] or \
                native.converged is not case["expected_converged"]:
            raise AssertionError(f"{case['name']}: update/convergence mismatch")
        inertia = validation_inertia(points, native.labels, native.centroids)
        if abs(inertia - case["expected_inertia"]) > \
                2e-5 * max(1.0, abs(case["expected_inertia"])):
            raise AssertionError(f"{case['name']}: inertia mismatch")
        if case["expected_inertia"] == 0.0 and inertia != 0.0:
            raise AssertionError(f"{case['name']}: zero inertia not exact")
        bits = int(np.count_nonzero(native.centroids.view(np.uint32) !=
                                    expected_centroids.view(np.uint32)))
        centroid_bit_mismatches += bits
        total += len(points)
        print(f"{case['name']}: {len(points)} exact labels, "
              f"centroid-bit mismatches={bits}, updates={native.update_count}, "
              f"converged={native.converged}")
    print(f"Fixtures: {len(public_cases())} cases, {total} labels exact, "
          f"{centroid_bit_mismatches} centroid-bit mismatches; "
          "input/ownership/repeatability checks passed")


def benchmark(executable: Path, directory: Path) -> None:
    spec = WORKLOADS["profiling"]
    workload = generate_workload(spec)
    points = workload.samples
    native, metadata = run_native(executable, points, spec.k, directory,
                                  "profiling", timed_runs=7)
    times_ms = metadata["timings_ms"]
    if len(times_ms) != 7 or metadata["warmup"] != 1:
        raise AssertionError("native timing protocol requires one warm-up and seven runs")

    fit_kmeans(points, spec.k)  # Untimed Python-reference warm-up.
    python_times_ms = []
    for _ in range(3):
        start = perf_counter()
        python_result = fit_kmeans(points, spec.k)
        python_times_ms.append((perf_counter() - start) * 1000.0)
    comparison = compare_results(points, python_result, native)
    median_ms = statistics.median(times_ms)
    distance_evaluations = spec.n * spec.k * (1 + native.update_count)
    print(f"Profiling workload: (N,D,K)=({spec.n},{spec.d},{spec.k}), "
          f"PCG64 seed={spec.seed}")
    print("C++ complete-fit raw ms: " + ", ".join(f"{t:.6f}" for t in times_ms))
    print(f"C++ min/median/max ms: {min(times_ms):.6f} / "
          f"{median_ms:.6f} / {max(times_ms):.6f}")
    print(f"C++ update_count={native.update_count}, converged={native.converged}; "
          f"samples/s={spec.n / (median_ms / 1000):,.0f}; "
          f"distance evaluations/s={distance_evaluations / (median_ms / 1000):,.0f}")
    print("Python-reference complete-fit raw ms: " +
          ", ".join(f"{t:.6f}" for t in python_times_ms))
    print(f"Python-reference median ms: {statistics.median(python_times_ms):.6f}")
    print(f"Full-workload comparison: exact labels/count/convergence; "
          f"max centroid absolute error={comparison.max_centroid_absolute_error:.9g}, "
          f"scaled error={comparison.max_centroid_scaled_error:.9g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="Release phase2_kmeans_serial executable")
    parser.add_argument("--benchmark", action="store_true",
                        help="time the frozen profiling workload after fixture validation")
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    with tempfile.TemporaryDirectory(prefix="project2_kmeans_") as temporary:
        directory = Path(temporary)
        validate_fixtures(executable, directory)
        if args.benchmark:
            benchmark(executable, directory)


if __name__ == "__main__":
    main()
