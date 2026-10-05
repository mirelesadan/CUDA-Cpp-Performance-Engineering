"""Bounded Python wall timing for the Project 2 CUDA K-means bindings.

Run validate_bindings.py --workloads first. This script rechecks the generated workloads
before timing; it is not a substitute for the public/invalid-input test suite.
All measurements use perf_counter, one warm-up and at least seven trials.
They include binding copies, result creation and blocking CUDA operations in
the named API boundary, but exclude oracle checks, file I/O and object teardown.
No reported wall time is a CUDA-event device algorithm measurement.

The native executable option runs the existing native owner benchmark on the
same generated array in this session. Its timings are steady_clock wall times;
the native construct/upload/fit/download/destroy mode includes destruction,
whereas Python construction is measured separately and teardown is excluded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from benchmark_data import WORKLOADS, generate_iterative_workload, generate_workload
from reference import KMeansResult, compare_results, fit_kmeans


ONE_CALL_MODES = ("python_one_shot", "construction", "upload", "resident_fit",
                  "download", "upload_fit_download")
REPEATS = (1, 2, 5, 10, 20)
BOUNDARIES = {
    "python_one_shot": "kmeans_cuda(X,K): complete Python call, including native ownership and copies",
    "construction": "CudaKMeansBuffer(N,D,K) only; retain object through timer stop; teardown excluded",
    "upload": "existing owner.upload(X), including NumPy-to-vector copy, native validation and H2D",
    "resident_fit": "existing resident owner.fit(), returning metadata; no bulk download",
    "download": "completed owner.download(), including D2H, vectors and NumPy output copies",
    "upload_fit_download": "existing owner: upload(X), independent fit(), download(); construction/teardown excluded",
    "R_fit_only": "R independent fits after upload, divided by R; upload/download/construction/teardown excluded",
    "R_upload_fits_download": "existing owner: one upload, R independent fits, one final download, divided by R",
}


def digest(points: np.ndarray) -> str:
    """Read the existing buffer; do not allocate another full input tensor."""
    return hashlib.sha256(memoryview(points)).hexdigest()


def result(value: dict) -> KMeansResult:
    return KMeansResult(value["labels"], value["centroids"],
                        value["update_count"], value["converged"])


def metadata_exact(expected: KMeansResult, actual: dict) -> None:
    if (actual["update_count"] != expected.update_count or
            actual["converged"] != expected.converged):
        raise AssertionError("Python fit metadata differs from validated one-shot result")


def output_exact(points: np.ndarray, expected: KMeansResult, value: dict) -> None:
    """Untimed deterministic repeat check, with no duplicate input tensor."""
    actual = result(value)
    metadata_exact(expected, value)
    for array, dtype, shape in ((actual.labels, np.int32, (points.shape[0],)),
                                (actual.centroids, np.float32, expected.centroids.shape)):
        if (not isinstance(array, np.ndarray) or array.dtype != dtype or
                array.shape != shape or not array.flags.c_contiguous or
                not array.flags.owndata or np.shares_memory(array, points)):
            raise AssertionError("Python output contract changed during timing")
    if (not np.array_equal(actual.labels, expected.labels) or
            not np.array_equal(actual.centroids.view(np.uint32),
                               expected.centroids.view(np.uint32))):
        raise AssertionError("Python deterministic output changed during timing")


def stats(values: list[float]) -> dict:
    mean = statistics.mean(values)
    return {"raw_ms": values, "min_ms": min(values),
            "median_ms": statistics.median(values), "max_ms": max(values),
            "mean_ms": mean,
            "cv_percent": 100.0 * statistics.pstdev(values) / mean if mean else 0.0}


def describe(name: str, values: list[float], fits: int = 1) -> None:
    measured = stats(values)
    print(f"{name}/{fits}: raw total ms={values}; "
          f"min/median/max={measured['min_ms']:.6f}/{measured['median_ms']:.6f}/"
          f"{measured['max_ms']:.6f} ms; mean={measured['mean_ms']:.6f} ms; "
          f"CV={measured['cv_percent']:.3f}%")
    if name.startswith("R_"):
        effective = stats([value / fits for value in values])
        print(f"  effective ms/fit: min/median/max={effective['min_ms']:.6f}/"
              f"{effective['median_ms']:.6f}/{effective['max_ms']:.6f}; "
              "construction/teardown excluded")


def native_case(executable: Path, name: str, points: np.ndarray, k: int,
                expected: KMeansResult, runs: int, sequences: int) -> dict:
    """Raw bridge setup/output validation are outside every native timer."""
    n, d = points.shape
    with tempfile.TemporaryDirectory(prefix="project2_python_native_") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.f32"
        replacement_path = directory / "replacement.f32"
        labels_path = directory / "labels.i32"
        centroids_path = directory / "centroids.f32"
        points.tofile(input_path)
        np.negative(points).tofile(replacement_path)
        command = [str(executable), str(input_path), str(replacement_path),
                   str(n), str(d), str(k), str(labels_path), str(centroids_path),
                   str(runs), str(sequences), "100"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"{name}: native benchmark failed:\n{completed.stderr}")
        record = json.loads(completed.stdout)
        if (record["n"], record["d"], record["k"]) != (n, d, k):
            raise AssertionError("native benchmark dimensions changed")
        if (record["checked_fits"] != 20 or not record["replacement_checked"] or
                not record["independent_downloads"]):
            raise AssertionError("native owner correctness checks did not complete")
        labels = np.fromfile(labels_path, dtype=np.int32)
        centroids = np.fromfile(centroids_path, dtype=np.float32).reshape(k, d).copy()
        native = {"labels": labels, "centroids": centroids,
                  "update_count": record["update_count"], "converged": record["converged"]}
        output_exact(points, expected, native)
        compare_results(points, expected, result(native))
    native_groups = defaultdict(list)
    for sample in record["samples"]:
        native_groups[sample["mode"], sample["fits"]].append(sample["wall_ms"])
    for mode in ("A_one_shot", "B_construct_upload_fit_download_destroy",
                 "C_resident_fit_download", "D_resident_fit_only", "E_upload_fit_download"):
        if len(native_groups[mode, 1]) != runs:
            raise AssertionError(f"incomplete native {mode} timing")
    for mode in ("R_fit_only", "R_upload_fits_download"):
        for fits in REPEATS:
            if len(native_groups[mode, fits]) != sequences:
                raise AssertionError(f"incomplete native {mode}/{fits} timing")
    for key in ("construction_ms", "destruction_ms", "openmp8_ms"):
        if len(record[key]) != runs:
            raise AssertionError(f"incomplete native {key} timing")
    print("RAW_NATIVE_JSON " + json.dumps({"workload": name, **record}, separators=(",", ":")))
    for (mode, fits), values in sorted(native_groups.items()):
        describe("native_" + mode, values, fits)
    for key in ("construction_ms", "destruction_ms", "openmp8_ms"):
        describe("native_" + key, record[key])
    return record


def run_case(module, name: str, points: np.ndarray, k: int,
             runs: int, sequences: int, native_executable: Path | None) -> None:
    n, d = points.shape
    before = digest(points)
    oracle = fit_kmeans(points, k)
    validated = result(module.kmeans_cuda(points, k))
    comparison = compare_results(points, oracle, validated)
    centroid_bits_exact = np.array_equal(validated.centroids.view(np.uint32),
                                         oracle.centroids.view(np.uint32))
    owner = module.CudaKMeansBuffer(n, d, k)
    owner.upload(points)
    metadata_exact(validated, owner.fit())
    output_exact(points, validated, owner.download())
    if digest(points) != before:
        raise AssertionError("Python binding modified input before timing")
    print(f"{name}: Python one-shot/owner validated before timing; "
          f"updates={validated.update_count}, converged={validated.converged}; "
          f"centroid bits exact={bool(centroid_bits_exact)}, "
          f"scaled centroid error={comparison.max_centroid_scaled_error:.9g}, "
          f"inertia error={abs(comparison.candidate_inertia - comparison.reference_inertia):.9g}")

    samples = []

    def one_call(mode: str, record: bool) -> None:
        # Fit completes before the download-only timer; no other operation or
        # validation is included in that API boundary.
        if mode == "download":
            metadata_exact(validated, owner.fit())
        start = perf_counter()
        if mode == "python_one_shot":
            value = module.kmeans_cuda(points, k)
        elif mode == "construction":
            value = module.CudaKMeansBuffer(n, d, k)
        elif mode == "upload":
            value = owner.upload(points)
        elif mode == "resident_fit":
            value = owner.fit()
        elif mode == "download":
            value = owner.download()
        elif mode == "upload_fit_download":
            owner.upload(points)
            owner.fit()
            value = owner.download()
        else:
            raise AssertionError("unknown timing mode")
        stop = perf_counter()
        if record:
            samples.append({"mode": mode, "fits": 1, "wall_ms": (stop - start) * 1000.0})
        # Object destruction and correctness checks occur after timer stop.
        if mode in ("python_one_shot", "download", "upload_fit_download"):
            output_exact(points, validated, value)
        elif mode == "resident_fit":
            metadata_exact(validated, value)
            output_exact(points, validated, owner.download())
        elif mode == "upload":
            metadata_exact(validated, owner.fit())
            output_exact(points, validated, owner.download())
        del value

    for mode in ONE_CALL_MODES:
        one_call(mode, False)
    for trial in range(runs):
        for offset in range(len(ONE_CALL_MODES)):
            one_call(ONE_CALL_MODES[(trial + offset) % len(ONE_CALL_MODES)], True)

    def repeated(fits: int, include_io: bool, record: bool) -> None:
        start = perf_counter()
        if include_io:
            owner.upload(points)
        for _ in range(fits):
            metadata = owner.fit()
        if include_io:
            downloaded = owner.download()
        stop = perf_counter()
        metadata_exact(validated, metadata)
        if not include_io:
            downloaded = owner.download()
        output_exact(points, validated, downloaded)
        if record:
            samples.append({"mode": "R_upload_fits_download" if include_io else "R_fit_only",
                            "fits": fits, "wall_ms": (stop - start) * 1000.0})

    # Construction is outside both repeated-fit boundaries. Every fit seeds
    # from the same uploaded input, independently of preceding fit results.
    owner.upload(points)
    for fits in REPEATS:
        repeated(fits, False, False)
        repeated(fits, True, False)
        for trial in range(sequences):
            repeated(fits, bool(trial % 2), True)
            repeated(fits, not bool(trial % 2), True)
    if digest(points) != before:
        raise AssertionError("Python binding modified input during timing")

    record = {"workload": name, "n": n, "d": d, "k": k,
              "update_count": validated.update_count, "converged": validated.converged,
              "runs": runs, "sequences": sequences, "timer": "time.perf_counter",
              "input_sha256": before, "centroid_bits_exact": bool(centroid_bits_exact),
              "boundaries": BOUNDARIES, "samples": samples}
    print("RAW_PYTHON " + json.dumps(record, separators=(",", ":")))
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["mode"], sample["fits"]].append(sample["wall_ms"])
    for mode in ONE_CALL_MODES:
        if len(groups[mode, 1]) != runs:
            raise AssertionError(f"incomplete Python {mode} timing")
    for mode in ("R_fit_only", "R_upload_fits_download"):
        for fits in REPEATS:
            if len(groups[mode, fits]) != sequences:
                raise AssertionError(f"incomplete Python {mode}/{fits} timing")
    for (mode, fits), values in sorted(groups.items()):
        describe(mode, values, fits)
    one_shot = statistics.median(groups["python_one_shot", 1])
    resident = statistics.median(groups["resident_fit", 1])
    upload_fit_download = statistics.median(groups["upload_fit_download", 1])
    print(f"{name}: Python one-shot/resident-fit={one_shot / resident:.6f}x; "
          f"Python one-shot/upload-fit-download={one_shot / upload_fit_download:.6f}x")
    # Release the Python owner before the native suite. Teardown is outside all
    # reported timers and the native suite uses its own existing owner boundary.
    del owner
    if native_executable is not None:
        native = native_case(native_executable, name, points, k, validated, runs, sequences)
        native_groups = defaultdict(list)
        for sample in native["samples"]:
            native_groups[sample["mode"], sample["fits"]].append(sample["wall_ms"])
        for py_mode, native_mode in (("python_one_shot", "A_one_shot"),
                                     ("resident_fit", "D_resident_fit_only"),
                                     ("upload_fit_download", "E_upload_fit_download")):
            py_median = statistics.median(groups[py_mode, 1])
            native_median = statistics.median(native_groups[native_mode, 1])
            print(f"{name}: {py_mode}: Python/native medians="
                  f"{py_median:.6f}/{native_median:.6f} ms; ratio={py_median / native_median:.6f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gpu", "iterative", "all"), default="all")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--sequences", type=int, default=7)
    parser.add_argument("--native-executable", type=Path)
    args = parser.parse_args()
    if not 7 <= args.runs <= 30 or not 7 <= args.sequences <= 30:
        parser.error("--runs and --sequences must be 7..30")
    module_dir = args.module_dir.resolve()
    if not module_dir.is_dir():
        parser.error(f"module directory does not exist: {module_dir}")
    native_executable = (args.native_executable.resolve()
                         if args.native_executable is not None else None)
    if native_executable is not None and not native_executable.is_file():
        parser.error(f"native executable does not exist: {native_executable}")
    sys.path.insert(0, str(module_dir))
    module = importlib.import_module("kmeans_native")
    modes = ("gpu", "iterative") if args.mode == "all" else (args.mode,)
    for mode in modes:
        generated = (generate_iterative_workload() if mode == "iterative" else
                     generate_workload(WORKLOADS["gpu"]))
        run_case(module, mode, generated.samples, generated.spec.k,
                 args.runs, args.sequences, native_executable)


if __name__ == "__main__":
    main()
