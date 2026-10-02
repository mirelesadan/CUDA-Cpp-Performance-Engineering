"""Validate and characterize benchmark-only native CUDA K-means lifecycles.

All wall/event timings originate in the executable, excluding Python, file I/O,
and oracle validation. Event spans can include pageable-copy/control gaps; their
sums are not a claim of pure kernel execution. No Python binding is introduced.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np

from benchmark_cuda import edge_cases
from benchmark_data import WORKLOADS, generate_iterative_workload, generate_workload
from reference import KMeansResult, _fit_with_update_cap, compare_results, fit_kmeans
from test_reference import public_cases


ALGORITHM = ("initialization", "initial_assignment", "count", "centroid_update",
             "flag_reset", "reassignment")
TRANSFERS = ("h2d", "flag_d2h", "labels_d2h", "centroids_d2h")


def stats(values: list[float]) -> str:
    return "/".join(f"{value:.6f}" for value in
                    (min(values), statistics.median(values), max(values)))


def summarize(metadata: dict, runs: int, sequences: int) -> None:
    groups = defaultdict(list)
    for sample in metadata["samples"]:
        groups[sample["mode"], sample["instrumentation"], sample["fits"]].append(sample)
    if runs:
        for mode in ("A_native", "A_shadow", "B_preallocated", "C_resident_input"):
            if len(groups[mode, "none", 1]) != runs:
                raise AssertionError(f"{mode}: incomplete uninstrumented timing runs")
        if len(metadata["openmp8_ms"]) != runs:
            raise AssertionError("incomplete OpenMP-8 timings")
    if sequences:
        for fits in (1, 2, 5, 10, 20):
            if len(groups["D_amortized", "none", fits]) != sequences:
                raise AssertionError(f"D_amortized/{fits}: incomplete timing sequences")
    print("All timing summaries are min/median/max in ms; effective values divide by fits.")
    for (mode, instrumentation, fits), samples in sorted(groups.items()):
        print(f"{mode}/{instrumentation}/{fits} fits: total "
              f"{stats([s['wall_ms'] for s in samples])}; effective "
              f"{stats([s['wall_ms'] / fits for s in samples])}; fit method "
              f"{stats([s['fit_wall_ms'] / fits for s in samples])}; event collection "
              f"{stats([s['event_collection_ms'] / fits for s in samples])}")
        # Zero-filled fields in uninstrumented samples mean not collected.
        fields = (("host_ms", "event_ms") if instrumentation == "events" else
                  ("host_ms",) if instrumentation == "host" else ())
        for field in fields:
            keys = sorted({key for sample in samples for key in sample[field]})
            if keys:
                print(f"  {field} effective medians: " + ", ".join(
                    f"{key}={statistics.median(s[field].get(key, 0.0) / fits for s in samples):.6f}"
                    for key in keys))
        if instrumentation == "events":
            for label, keys in (("algorithm event spans", ALGORITHM),
                                ("transfer event spans", TRANSFERS)):
                values = [sum(s["event_ms"].get(key, 0.0) for key in keys) / fits
                          for s in samples]
                print(f"  effective {label}: {stats(values)} (not pure device execution)")
    if metadata["openmp8_ms"]:
        print(f"OpenMP-8 complete native wall: {stats(metadata['openmp8_ms'])}")
    memory = metadata["device_bytes"]
    print(f"Resident device bytes: {memory}; total={sum(memory.values())} "
          f"({sum(memory.values()) / 2**20:.6f} MiB)")
    print("Diagnostic event setup/destruction outside measured sequences: "
          f"{metadata['diagnostic_event_setup_ms']:.6f}/"
          f"{metadata['diagnostic_event_destroy_ms']:.6f} ms")


def run_case(executable: Path, directory: Path, name: str, points: np.ndarray,
             k: int, runs: int = 0, sequences: int = 0, cap: int = 100,
             expected: dict | None = None) -> None:
    if points.dtype != np.float32 or not points.flags.c_contiguous:
        raise AssertionError("bridge requires C-contiguous float32 points")
    before = points.view(np.uint32).copy()
    oracle = fit_kmeans(points, k) if cap == 100 else _fit_with_update_cap(points, k, cap)
    input_path, labels_path, centroids_path = (
        directory / f"{name}.{suffix}" for suffix in ("input.f32", "labels.i32", "centroids.f32"))
    points.tofile(input_path)
    digest = hashlib.sha256(input_path.read_bytes()).digest()
    n, d = points.shape
    command = [str(executable), str(input_path), str(n), str(d), str(k),
               str(labels_path), str(centroids_path), str(runs), str(sequences), str(cap)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = json.loads(completed.stdout)
    if (metadata["n"], metadata["d"], metadata["k"]) != (n, d, k):
        raise AssertionError("native workload dimensions changed")
    if metadata["checked_fits"] < 20:
        raise AssertionError("resident state did not pass at least 20 independent reset checks")
    if hashlib.sha256(input_path.read_bytes()).digest() != digest:
        raise AssertionError("native bridge modified serialized input")
    labels = np.fromfile(labels_path, dtype=np.int32)
    centroids = np.fromfile(centroids_path, dtype=np.float32)
    if labels.size != n or centroids.size != k * d:
        raise AssertionError("native output size mismatch")
    candidate = KMeansResult(labels.copy(), centroids.reshape(k, d).copy(),
                             metadata["update_count"], metadata["converged"])
    comparison = compare_results(points, oracle, candidate)
    np.testing.assert_array_equal(candidate.centroids.view(np.uint32), oracle.centroids.view(np.uint32))
    np.testing.assert_array_equal(points.view(np.uint32), before)
    if expected is not None:
        np.testing.assert_array_equal(candidate.labels, expected["expected_labels"])
        if (candidate.update_count != expected["expected_update_count"] or
                candidate.converged != expected["expected_converged"]):
            raise AssertionError(f"{name}: frozen fixture termination changed")
    print(f"{name}: ({n},{d},{k}), exact labels/centroid bits; "
          f"updates={candidate.update_count}, converged={candidate.converged}; "
          f"checked fits={metadata['checked_fits']}; inertia error="
          f"{abs(comparison.candidate_inertia - comparison.reference_inertia):.9g}")
    print("RAW_JSON " + json.dumps({"workload": name, **metadata}, separators=(",", ":")))
    summarize(metadata, runs, sequences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("mode", choices=("fixtures", "gpu", "iterative", "primary", "all"))
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--sequences", type=int, default=5)
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    if not 7 <= args.runs <= 30 or not 5 <= args.sequences <= 20:
        parser.error("--runs must be 7..30 and --sequences must be 5..20")
    modes = ("fixtures", "gpu", "iterative", "primary") if args.mode == "all" else (args.mode,)
    with tempfile.TemporaryDirectory(prefix="project2_lifecycle_") as temporary:
        directory = Path(temporary)
        for mode in modes:
            if mode == "fixtures":
                for case in public_cases():
                    run_case(executable, directory, case["name"],
                             np.array(case["points"], dtype=np.float32, order="C"), case["k"],
                             cap=case["max_updates"], expected=case)
                for name, points, k in edge_cases():
                    run_case(executable, directory, name, points, k)
                points = np.full((1025, 3), -0.0, dtype=np.float32)
                run_case(executable, directory, "partial_tile_identical_k32", points, 32)
            else:
                generated = (generate_iterative_workload() if mode == "iterative" else
                             generate_workload(WORKLOADS["profiling" if mode == "primary" else "gpu"]))
                run_case(executable, directory, mode, generated.samples, generated.spec.k,
                         args.runs, args.sequences)


if __name__ == "__main__":
    main()
