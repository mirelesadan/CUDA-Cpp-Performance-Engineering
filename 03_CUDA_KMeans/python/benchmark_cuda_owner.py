"""Validate and benchmark the supported native resident CUDA K-means owner.

This raw-file bridge is not a Python binding. Timings are native steady_clock
wall times: no Python, file I/O, oracle checks, CUDA events, or profiling. The
owner constructor/destructor boundaries are explicit in each reported mode.
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


MODES = ("A_one_shot", "B_construct_upload_fit_download_destroy",
         "C_resident_fit_download", "D_resident_fit_only", "E_upload_fit_download")


def stats(values: list[float]) -> str:
    mean = statistics.mean(values)
    cv = 100 * statistics.pstdev(values) / mean if mean else 0.0
    return (f"min/median/max={min(values):.6f}/{statistics.median(values):.6f}/"
            f"{max(values):.6f} ms, mean={mean:.6f} ms, CV={cv:.3f}%")


def summarize(metadata: dict, runs: int, sequences: int) -> None:
    groups = defaultdict(list)
    for sample in metadata["samples"]:
        groups[sample["mode"], sample["fits"]].append(sample["wall_ms"])
    if runs:
        for mode in MODES:
            if len(groups[mode, 1]) != runs:
                raise AssertionError(f"{mode}: incomplete timing runs")
        for key in ("openmp8_ms", "construction_ms", "destruction_ms"):
            if len(metadata[key]) != runs:
                raise AssertionError(f"{key}: incomplete timing runs")
    if sequences:
        for mode in ("R_fit_only", "R_upload_fits_download"):
            for fits in (1, 2, 5, 10, 20):
                if len(groups[mode, fits]) != sequences:
                    raise AssertionError(f"{mode}/{fits}: incomplete timing sequences")
    for (mode, fits), values in sorted(groups.items()):
        print(f"{mode}/{fits}: raw total ms={values}; {stats(values)}")
        if mode.startswith("R_"):
            effective = [value / fits for value in values]
            print(f"  effective per fit: {stats(effective)}; construction/destruction excluded")
    for key in ("construction_ms", "destruction_ms", "openmp8_ms"):
        if metadata[key]:
            print(f"{key}: raw={metadata[key]}; {stats(metadata[key])}")
    if runs:
        omp = statistics.median(metadata["openmp8_ms"])
        for mode in MODES:
            median = statistics.median(groups[mode, 1])
            print(f"{mode}: OpenMP-8/median={omp / median:.6f}x")
    print(f"Resident device storage: {metadata['device_bytes']} bytes "
          f"({metadata['device_bytes'] / 2**20:.6f} MiB)")


def run_case(executable: Path, directory: Path, name: str, points: np.ndarray,
             k: int, runs: int = 0, sequences: int = 0, cap: int = 100,
             expected: dict | None = None) -> None:
    if points.dtype != np.float32 or not points.flags.c_contiguous:
        raise AssertionError("bridge requires C-contiguous float32 points")
    before = points.view(np.uint32).copy()
    # Negation is exact for finite FP32 values (including signed zero), remains
    # in bounds, and changes the uploaded snapshot without changing its shape.
    replacement = np.negative(points)
    replacement_before = replacement.view(np.uint32).copy()
    reference = fit_kmeans(points, k) if cap == 100 else _fit_with_update_cap(points, k, cap)
    replacement_reference = (fit_kmeans(replacement, k) if cap == 100 else
                             _fit_with_update_cap(replacement, k, cap))
    input_path, replacement_path, labels_path, centroids_path = (
        directory / f"{name}.{suffix}" for suffix in
        ("input.f32", "replacement.f32", "labels.i32", "centroids.f32"))
    points.tofile(input_path)
    replacement.tofile(replacement_path)
    digests = {path: hashlib.sha256(path.read_bytes()).digest()
               for path in (input_path, replacement_path)}
    n, d = points.shape
    command = [str(executable), str(input_path), str(replacement_path),
               str(n), str(d), str(k), str(labels_path), str(centroids_path),
               str(runs), str(sequences), str(cap)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{name}: native owner check failed:\n{completed.stderr}")
    metadata = json.loads(completed.stdout)
    if (metadata["n"], metadata["d"], metadata["k"]) != (n, d, k):
        raise AssertionError("native workload dimensions changed")
    if metadata["checked_fits"] != 20 or not metadata["replacement_checked"]:
        raise AssertionError("independent fits / input replacement not verified")
    if not metadata["independent_downloads"]:
        raise AssertionError("independent host output ownership not verified")
    expected_bytes = (4*n*d + 4*k*d + 8*n + 4*k + 4 +
                      8*k*d*((n+4095)//4096) + 4*k*((n+1023)//1024))
    if metadata["device_bytes"] != expected_bytes:
        raise AssertionError("resident allocation footprint differs from eight-buffer design")
    for path, digest in digests.items():
        if hashlib.sha256(path.read_bytes()).digest() != digest:
            raise AssertionError("native bridge modified serialized input")

    candidates = []
    for suffix, prefix, x, oracle in (("", "", points, reference),
                                     (".replacement", "replacement_", replacement,
                                      replacement_reference)):
        labels = np.fromfile(str(labels_path) + suffix, dtype=np.int32)
        centroids = np.fromfile(str(centroids_path) + suffix, dtype=np.float32)
        if labels.size != n or centroids.size != k*d:
            raise AssertionError("native output size mismatch")
        candidate = KMeansResult(labels.copy(), centroids.reshape(k, d).copy(),
                                 metadata[prefix + "update_count"],
                                 metadata[prefix + "converged"])
        comparison = compare_results(x, oracle, candidate)
        # Stronger than the frozen centroid/inertia tolerances for these cases.
        np.testing.assert_array_equal(candidate.centroids.view(np.uint32),
                                      oracle.centroids.view(np.uint32))
        candidates.append(candidate)
        print(f"{name}{suffix}: exact labels/centroid bits/termination; "
              f"updates={candidate.update_count}, converged={candidate.converged}; "
              f"inertia error={abs(comparison.candidate_inertia - comparison.reference_inertia):.9g}")
    if expected is not None:
        candidate = candidates[0]
        np.testing.assert_array_equal(candidate.labels, expected["expected_labels"])
        frozen_centroids = np.asarray(expected["expected_centroids"], dtype=np.float32)
        np.testing.assert_array_equal(candidate.centroids.view(np.uint32),
                                      frozen_centroids.view(np.uint32))
        if (candidate.update_count != expected["expected_update_count"] or
                candidate.converged != expected["expected_converged"]):
            raise AssertionError("frozen fixture termination changed")
    np.testing.assert_array_equal(points.view(np.uint32), before)
    np.testing.assert_array_equal(replacement.view(np.uint32), replacement_before)
    print(f"{name}: 20 independent fits with downloads; independent output ownership; "
          "X1 -> X2 -> X1 replacement exact; input unchanged")
    print("RAW_JSON " + json.dumps({"workload": name, **metadata}, separators=(",", ":")))
    summarize(metadata, runs, sequences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("mode", choices=("fixtures", "gpu", "iterative", "primary", "all"))
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--sequences", type=int, default=7)
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    if not 7 <= args.runs <= 30 or not 7 <= args.sequences <= 30:
        parser.error("--runs and --sequences must be 7..30")
    modes = ("fixtures", "gpu", "iterative", "primary") if args.mode == "all" else (args.mode,)
    with tempfile.TemporaryDirectory(prefix="project2_owner_") as temporary:
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
