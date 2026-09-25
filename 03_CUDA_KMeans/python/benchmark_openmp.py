"""Validate and benchmark assignment-only OpenMP against addressed serial C++.

Generated FP32 inputs and raw-file exchange live in an OS temporary directory.
The native executable loads input once, warms each configuration once, then
rotates the order of complete-fit timings; file I/O and bitwise comparisons
are outside its steady-clock timing boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np

from benchmark_data import (ITERATIVE_PROFILE_SPEC, WORKLOADS,
                            generate_iterative_workload, generate_workload)
from reference import KMeansResult, _fit_with_update_cap, compare_results, fit_kmeans
from test_reference import public_cases


COUNTS = (1, 2, 4, 8, 16, 20)


def invoke(executable: Path, points: np.ndarray, k: int, directory: Path,
           stem: str, *, runs: int = 0, update_cap: int = 100) -> tuple[KMeansResult, dict]:
    n, d = points.shape
    if points.dtype != np.float32 or not points.flags.c_contiguous:
        raise AssertionError("native bridge requires C-contiguous float32 input")
    input_path = directory / f"{stem}.input.f32"
    labels_path = directory / f"{stem}.labels.i32"
    centroids_path = directory / f"{stem}.centroids.f32"
    points.tofile(input_path)
    before = hashlib.sha256(input_path.read_bytes()).digest()
    command = [str(executable), str(input_path), str(n), str(d), str(k),
               str(labels_path), str(centroids_path),
               "--counts", ",".join(map(str, COUNTS)), "--runs", str(runs)]
    if update_cap != 100:
        command.extend(["--test-update-cap", str(update_cap)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = json.loads(completed.stdout)
    if hashlib.sha256(input_path.read_bytes()).digest() != before:
        raise AssertionError("native driver modified its input")
    if metadata["counts"] != list(COUNTS) or metadata["timed_runs_each"] != runs:
        raise AssertionError("unexpected thread counts or timing protocol")
    labels = np.fromfile(labels_path, dtype=np.int32)
    centroids = np.fromfile(centroids_path, dtype=np.float32)
    if labels.size != n or centroids.size != k * d:
        raise AssertionError("native output size mismatch")
    result = KMeansResult(labels.copy(), centroids.reshape(k, d).copy(),
                          metadata["update_count"], metadata["converged"])
    return result, metadata


def fixtures(executable: Path, directory: Path) -> None:
    labels_checked = 0
    centroid_bit_mismatches = 0
    for case in public_cases():
        points = np.array(case["points"], dtype=np.float32, order="C")
        original = points.view(np.uint32).copy()
        cap = case["max_updates"]
        oracle = (fit_kmeans(points, case["k"]) if cap == 100 else
                  _fit_with_update_cap(points, case["k"], cap))
        first, metadata = invoke(executable, points, case["k"], directory,
                                 case["name"], update_cap=cap)
        second, _ = invoke(executable, points, case["k"], directory,
                           case["name"] + ".repeat", update_cap=cap)
        compare_results(points, oracle, first)
        expected_labels = np.array(case["expected_labels"], dtype=np.int32)
        expected_centroids = np.array(case["expected_centroids"], dtype=np.float32)
        np.testing.assert_array_equal(first.labels, expected_labels)
        np.testing.assert_array_equal(first.centroids.view(np.uint32),
                                      expected_centroids.view(np.uint32))
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.centroids.view(np.uint32),
                                      second.centroids.view(np.uint32))
        np.testing.assert_array_equal(points.view(np.uint32), original)
        if first.update_count != case["expected_update_count"] or \
                first.converged is not case["expected_converged"]:
            raise AssertionError(f"{case['name']}: update/convergence changed")
        if metadata["warmup_each"] != 1:
            raise AssertionError("native driver did not warm all configurations")
        labels_checked += len(points)
        print(f"{case['name']}: {len(points)} labels and centroid bits exact; "
              f"serial/OpenMP-{','.join(map(str, COUNTS))} matched internally")
    print(f"public fixtures: six cases, {labels_checked} exact labels, "
          f"{centroid_bit_mismatches} centroid-bit mismatches")


def median_phases(metadata: dict, name: str, expected_updates: int) -> None:
    phases = metadata["phase_timings"]
    times = metadata["times_ms"]
    print(f"{name} coarse phase medians (separate timed build; directional):")
    for index, label in enumerate(("serial", *(f"OpenMP-{c}" for c in COUNTS))):
        rows = phases[index]
        if not rows or len(rows) != len(times[index]):
            raise AssertionError("phase-timing run count mismatch")
        if any(row["assignment_passes"] != expected_updates + 1 or
               row["centroid_updates"] != expected_updates for row in rows):
            raise AssertionError("missing or inconsistent native phase timings")
        assignments = [row["initial_assignment_ms"] + row["reassignment_ms"]
                       for row in rows]
        updates = [row["centroid_update_ms"] for row in rows]
        whole = statistics.median(times[index])
        assign = statistics.median(assignments)
        update = statistics.median(updates)
        print(f"  {label}: whole {whole:.6f} ms; assignment {assign:.6f} ms "
              f"({100 * assign / whole:.1f}%); serial updates {update:.6f} ms "
              f"({100 * update / whole:.1f}%); passes "
              f"{rows[0]['assignment_passes']}/{rows[0]['centroid_updates']}")


def workload(executable: Path, phase_executable: Path | None,
             which: str, directory: Path) -> None:
    generated = (generate_workload(WORKLOADS["profiling"])
                 if which == "profiling" else generate_iterative_workload())
    spec = generated.spec
    # Seven rotations place every configuration once in each call-order slot.
    runs = 7
    expected_updates = 1 if which == "profiling" else 22
    oracle = fit_kmeans(generated.samples, spec.k)
    result, metadata = invoke(executable, generated.samples, spec.k,
                              directory, which, runs=runs)
    compare_results(generated.samples, oracle, result)
    if result.update_count != expected_updates or not result.converged:
        raise AssertionError("workload iteration behavior changed")
    if metadata["warmup_each"] != 1:
        raise AssertionError("native driver omitted warm-ups")
    times = metadata["times_ms"]
    if len(times) != len(COUNTS) + 1 or any(len(row) != runs for row in times):
        raise AssertionError("native driver omitted a timed fit")
    serial_median = statistics.median(times[0])
    evaluations = spec.n * spec.k * (result.update_count + 1)
    print(f"{which}: (N,D,K)=({spec.n},{spec.d},{spec.k}), seed={spec.seed}; "
          f"updates={result.update_count}, converged={result.converged}; "
          "all native configurations bitwise identical")
    print(f"OpenMP runtime: max threads={metadata['omp_max_threads']}, "
          f"processors={metadata['omp_num_procs']}; tested={COUNTS}")
    print("variant | raw ms | min / median / max ms | speedup | efficiency | "
          "Msamples/s | Mdistances/s")
    for index, label in enumerate(("serial", *(f"OpenMP-{c}" for c in COUNTS))):
        row = times[index]
        med = statistics.median(row)
        speedup = serial_median / med
        efficiency = speedup / COUNTS[index - 1] if index else 1.0
        samples_per_s = spec.n * 1000.0 / med / 1e6
        distances_per_s = evaluations * 1000.0 / med / 1e6
        raw = ", ".join(f"{value:.6f}" for value in row)
        print(f"{label} | {raw} | {min(row):.6f} / {med:.6f} / "
              f"{max(row):.6f} | {speedup:.4f}x | {efficiency:.4f} | "
              f"{samples_per_s:.4f} | {distances_per_s:.4f}")
    if phase_executable is not None:
        phase_result, phase_metadata = invoke(
            phase_executable, generated.samples, spec.k, directory,
            which + ".phase", runs=runs)
        np.testing.assert_array_equal(phase_result.labels, result.labels)
        np.testing.assert_array_equal(phase_result.centroids.view(np.uint32),
                                      result.centroids.view(np.uint32))
        median_phases(phase_metadata, which, expected_updates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path,
                        help="normal Release phase2_kmeans_openmp_scaling executable")
    parser.add_argument("mode", choices=("fixtures", "profiling", "iterative_profile"))
    parser.add_argument("--counts", default="1,2,4,8,16,20",
                        help="comma-separated supported OpenMP counts; default is the Windows scaling set")
    parser.add_argument("--phase-executable", type=Path,
                        help="separate phase-timed Release scaling executable")
    args = parser.parse_args()
    try:
        selected_counts = tuple(int(part) for part in args.counts.split(","))
    except ValueError:
        parser.error("--counts must contain comma-separated positive integers")
    if not selected_counts or any(count <= 0 for count in selected_counts) or \
            len(set(selected_counts)) != len(selected_counts):
        parser.error("--counts must contain distinct positive integers")
    global COUNTS
    COUNTS = selected_counts
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    phase_executable = args.phase_executable.resolve() if args.phase_executable else None
    if phase_executable is not None and not phase_executable.is_file():
        parser.error(f"phase executable does not exist: {phase_executable}")
    with tempfile.TemporaryDirectory(prefix="project2_openmp_") as temporary:
        directory = Path(temporary)
        if args.mode == "fixtures":
            fixtures(executable, directory)
        else:
            workload(executable, phase_executable, args.mode, directory)


if __name__ == "__main__":
    main()
