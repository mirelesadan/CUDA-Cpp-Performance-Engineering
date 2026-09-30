"""Validate and time the correctness-first Project 2 native CUDA K-means.

The native executable reads a raw float32 file before timing. Its normal
steady-clock fits include full native allocations, H2D, device work, D2H,
and teardown. CUDA-event stage diagnostics are separate runs, not substitutes
for the fresh serial/OpenMP-8/CUDA whole-fit timing comparison.
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

from benchmark_data import WORKLOADS, generate_iterative_workload, generate_workload
from reference import KMeansResult, _fit_with_update_cap, compare_results, fit_kmeans
from test_reference import public_cases


STAGE_FIELDS = (
    "setup_ms", "h2d_ms", "initialization_ms", "initial_assignment_ms",
    "count_ms", "centroid_update_ms", "reassignment_ms", "convergence_ms",
    "final_d2h_ms", "device_algorithm_ms", "one_shot_gpu_ms", "native_wall_ms",
)


def invoke(executable: Path, points: np.ndarray, k: int, directory: Path,
           stem: str, *, runs: int = 0, diagnostic_runs: int = 0,
           update_cap: int = 100, skip_serial: bool = False) -> tuple[KMeansResult, dict]:
    n, d = points.shape
    if points.dtype != np.float32 or not points.flags.c_contiguous:
        raise AssertionError("native bridge requires C-contiguous float32 input")
    input_path = directory / f"{stem}.input.f32"
    labels_path = directory / f"{stem}.labels.i32"
    centroids_path = directory / f"{stem}.centroids.f32"
    points.tofile(input_path)
    before = hashlib.sha256(input_path.read_bytes()).digest()
    command = [
        str(executable), str(input_path), str(n), str(d), str(k),
        str(labels_path), str(centroids_path), "--runs", str(runs),
        "--diagnostic-runs", str(diagnostic_runs),
    ]
    if update_cap != 100:
        command.extend(["--test-update-cap", str(update_cap)])
    if skip_serial:
        command.extend(["--skip-serial", "1"])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = json.loads(completed.stdout)
    if hashlib.sha256(input_path.read_bytes()).digest() != before:
        raise AssertionError("native bridge modified input")
    if metadata["timed_runs_each"] != runs or metadata["diagnostic_runs"] != diagnostic_runs:
        raise AssertionError("native bridge timing protocol mismatch")
    labels = np.fromfile(labels_path, dtype=np.int32)
    centroids = np.fromfile(centroids_path, dtype=np.float32)
    if labels.size != n or centroids.size != k * d:
        raise AssertionError("native output size mismatch")
    result = KMeansResult(labels.copy(), centroids.reshape(k, d).copy(),
                          metadata["update_count"], metadata["converged"])
    return result, metadata


def verify(points: np.ndarray, oracle: KMeansResult,
           candidate: KMeansResult) -> None:
    comparison = compare_results(points, oracle, candidate)
    # CPU native results have previously matched all centroid bits. Keep
    # that stronger baseline for this correctness-first CUDA milestone.
    np.testing.assert_array_equal(candidate.centroids.view(np.uint32),
                                  oracle.centroids.view(np.uint32))
    if comparison.max_centroid_absolute_error != 0.0 or \
            comparison.max_centroid_scaled_error != 0.0 or \
            comparison.candidate_inertia != comparison.reference_inertia:
        raise AssertionError("exact centroid or inertia agreement was lost")


def run_fixtures(executable: Path, directory: Path) -> None:
    labels_checked = 0
    for case in public_cases():
        points = np.array(case["points"], dtype=np.float32, order="C")
        before = points.view(np.uint32).copy()
        cap = case["max_updates"]
        oracle = (fit_kmeans(points, case["k"]) if cap == 100 else
                  _fit_with_update_cap(points, case["k"], cap))
        candidate, metadata = invoke(executable, points, case["k"], directory,
                                     case["name"], update_cap=cap)
        again, _ = invoke(executable, points, case["k"], directory,
                          case["name"] + ".repeat", update_cap=cap)
        verify(points, oracle, candidate)
        verify(points, oracle, again)
        np.testing.assert_array_equal(candidate.labels, case["expected_labels"])
        expected_centroids = np.asarray(case["expected_centroids"], dtype=np.float32)
        np.testing.assert_array_equal(candidate.centroids.view(np.uint32),
                                      expected_centroids.view(np.uint32))
        np.testing.assert_array_equal(points.view(np.uint32), before)
        if candidate.update_count != case["expected_update_count"] or \
                candidate.converged is not case["expected_converged"]:
            raise AssertionError(f"{case['name']}: update/convergence changed")
        if metadata["cuda_warmups"] != 1:
            raise AssertionError("fixture CUDA path was not launched")
        labels_checked += len(points)
        print(f"{case['name']}: {len(points)} labels, centroid bits, "
              "update/convergence, and inertia exact; repeat exact")
    print(f"public fixtures: {len(public_cases())} cases, "
          f"{labels_checked} labels, zero centroid-bit mismatches")


def edge_cases() -> list[tuple[str, np.ndarray, int]]:
    d1 = np.array([[-4.0]] * 8 + [[4.0]] * 8, dtype=np.float32)
    odd = np.zeros((16, 3), dtype=np.float32)
    odd[4] = [1.0, 2.0 ** -12, 2.0 ** -12]
    odd[8:] = [1.0, 0.0, 0.0]
    odd[12] = [1.0, 0.0, 0.0]
    even = np.array([[-4.0, -3.0, -2.0, -1.0]] * 8 +
                    [[4.0, 3.0, 2.0, 1.0]] * 8, dtype=np.float32)
    near_limit = np.arange(64, dtype=np.float32).reshape(64, 1)
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    signed_zero = np.array([[-0.0, tiny, -tiny]] * 8 +
                           [[1.0, -0.0, tiny]] * 8, dtype=np.float32)
    subnormal_square = np.array([[0.0, 1e-19, -1e-19]] * 8 +
                                [[1e-19, 0.0, 1e-19]] * 8, dtype=np.float32)
    return [
        ("edge_d1_k2", np.ascontiguousarray(d1), 2),
        ("edge_d3_ordered", np.ascontiguousarray(odd), 2),
        ("edge_d4_k2", np.ascontiguousarray(even), 2),
        ("edge_k32", np.ascontiguousarray(near_limit), 32),
        ("edge_signed_zero_subnormal", np.ascontiguousarray(signed_zero), 2),
        ("edge_subnormal_square", np.ascontiguousarray(subnormal_square), 2),
    ]


def run_edges(executable: Path, directory: Path) -> None:
    for name, points, k in edge_cases():
        before = points.view(np.uint32).copy()
        oracle = fit_kmeans(points, k)
        candidate, _ = invoke(executable, points, k, directory, name)
        verify(points, oracle, candidate)
        np.testing.assert_array_equal(points.view(np.uint32), before)
        print(f"{name}: (N,D,K)=({len(points)},{points.shape[1]},{k}), "
              f"updates={candidate.update_count}, converged={candidate.converged}; "
              "labels/centroid bits/inertia exact")


def print_timings(name: str, metadata: dict) -> None:
    print(f"{name}: OpenMP max threads={metadata['omp_max_threads']}, "
          f"processors={metadata['omp_num_procs']}; control threads=8")
    print(f"normal-fit warm-ups: CPU={metadata['cpu_warmups']}, "
          f"CUDA={metadata['cuda_warmups']}; "
          f"timed runs each={metadata['timed_runs_each']}")
    medians = {}
    for label, key in (
        ("addressed serial", "serial_times_ms"),
        ("OpenMP-8", "openmp8_times_ms"),
        ("CUDA native wall", "cuda_wall_times_ms"),
    ):
        values = metadata[key]
        if not values:
            continue
        if len(values) != metadata["timed_runs_each"]:
            raise AssertionError(f"{label}: missing timed fits")
        medians[label] = statistics.median(values)
        raw = ", ".join(f"{value:.6f}" for value in values)
        print(f"{label}: raw [{raw}] ms; min/median/max "
              f"{min(values):.6f}/{medians[label]:.6f}/{max(values):.6f} ms")
    if "CUDA native wall" in medians:
        for label in ("addressed serial", "OpenMP-8"):
            if label in medians:
                print(f"{label} / CUDA native-wall speedup: "
                      f"{medians[label] / medians['CUDA native wall']:.6f}x")

    stages = metadata["cuda_stage_runs"]
    if len(stages) != metadata["diagnostic_runs"]:
        raise AssertionError("missing CUDA stage diagnostic runs")
    if stages:
        if any(set(stage) != set(STAGE_FIELDS) for stage in stages):
            raise AssertionError("CUDA stage field set changed")
        print("CUDA diagnostic stage runs (separate event-timed fits):")
        print(json.dumps(stages, separators=(",", ":")))
        stage_medians = {key: statistics.median(stage[key] for stage in stages)
                         for key in STAGE_FIELDS}
        print("CUDA diagnostic stage medians (ms): "
              + ", ".join(f"{key}={value:.6f}"
                          for key, value in stage_medians.items()))
        print("Diagnostic device/one-shot timings are directional and are not "
              "the normal-fit native-wall speedup denominator.")


def run_workload(executable: Path, directory: Path, mode: str,
                 diagnostic_runs: int) -> None:
    generated = (generate_iterative_workload() if mode == "iterative" else
                 generate_workload(WORKLOADS["profiling" if mode == "primary"
                                             else "gpu"]))
    spec = generated.spec
    expected_updates = 22 if mode == "iterative" else 1
    runs = 5 if mode == "iterative" else 7
    points = generated.samples
    before = points.view(np.uint32).copy()
    oracle = fit_kmeans(points, spec.k)
    candidate, metadata = invoke(
        executable, points, spec.k, directory, mode, runs=runs,
        diagnostic_runs=diagnostic_runs)
    verify(points, oracle, candidate)
    np.testing.assert_array_equal(points.view(np.uint32), before)
    if candidate.update_count != expected_updates or not candidate.converged:
        raise AssertionError(f"{mode}: iteration state changed")
    print(f"{mode}: (N,D,K)=({spec.n},{spec.d},{spec.k}), seed={spec.seed}; "
          f"updates={candidate.update_count}, converged={candidate.converged}; "
          "native CPU/OpenMP/CUDA and NumPy labels/centroid bits exact")
    print_timings(mode, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path,
                        help="Release phase2_kmeans_cuda_benchmark executable")
    parser.add_argument("mode", choices=("fixtures", "edges", "primary",
                                         "gpu", "iterative", "all"))
    parser.add_argument("--diagnostic-runs", type=int, default=3,
                        help="separate CUDA stage-timing fits per workload")
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    if not 0 <= args.diagnostic_runs <= 20:
        parser.error("--diagnostic-runs must be 0..20")
    modes = ("fixtures", "edges", "primary", "gpu", "iterative") \
        if args.mode == "all" else (args.mode,)
    with tempfile.TemporaryDirectory(prefix="project2_cuda_") as temporary:
        directory = Path(temporary)
        for mode in modes:
            if mode == "fixtures":
                run_fixtures(executable, directory)
            elif mode == "edges":
                run_edges(executable, directory)
            else:
                run_workload(executable, directory, mode, args.diagnostic_runs)


if __name__ == "__main__":
    main()
