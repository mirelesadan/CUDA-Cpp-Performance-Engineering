"""Controlled scikit-learn/Project 2 comparison on established FP32 workloads.

Run on AC power after the installed-version semantics have been inspected.
Python timers include complete public API calls and exclude generation, oracle
checks and file I/O. Resident fit excludes construction/upload/download. The
optional native OpenMP timer includes native validation/result allocations,
but excludes this Python/raw-file bridge. No implementation is tuned here.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from benchmark_bindings import digest, metadata_exact, output_exact, result, stats
from benchmark_data import WORKLOADS, generate_iterative_workload, generate_workload
from reference import (KMeansResult, UPDATE_CHUNK_ROWS, compare_results, fit_kmeans,
                       initial_row_indices, validation_inertia)


CONFIGURATION = {"algorithm": "lloyd", "n_init": 1, "max_iter": 100,
                 "tol": 0.0, "copy_x": True, "openmp_limit": 8, "blas_limit": 1,
                 "init": "exact Project 2 fixed rows, prepared before timing"}
MODES = ("sklearn_fit", "python_cuda_one_shot", "resident_fit",
         "construction", "upload", "download")
BOUNDARIES = {
    "sklearn_fit": "KMeans(...).fit(X), constructor and normal fit validation/copy/work included",
    "python_cuda_one_shot": "kmeans_native.kmeans_cuda(X,K), all native ownership/copies included",
    "resident_fit": "uploaded existing owner.fit(), metadata only; no construction/upload/download",
    "construction": "CudaKMeansBuffer(N,D,K), retained through timer stop; destruction excluded",
    "upload": "existing owner.upload(X), native validation/copy/H2D included",
    "download": "completed owner.download(), D2H/native vectors/NumPy output creation included",
    "native_openmp8": "native steady_clock fit, raw-file/Python bridge excluded; assignment-only OpenMP-8",
}


def quality(points: np.ndarray, project: KMeansResult, model) -> dict:
    """Analyze same cluster IDs; do not assert scikit-learn contract parity."""
    labels, centers = model.labels_, model.cluster_centers_
    if (labels.shape != project.labels.shape or centers.shape != project.centroids.shape or
            centers.dtype != np.float32 or not np.isfinite(centers).all()):
        raise AssertionError("unexpected scikit-learn output shape/dtype/finite status")
    scales = np.ones(points.shape[1], dtype=np.float64)
    for begin in range(0, len(points), UPDATE_CHUNK_ROWS):
        scales = np.maximum(scales, np.max(np.abs(points[begin:begin + UPDATE_CHUNK_ROWS]), axis=0))
    error = np.abs(centers.astype(np.float64) - project.centroids.astype(np.float64))
    agreement = int(np.count_nonzero(labels == project.labels))
    project_inertia = validation_inertia(points, project.labels, project.centroids)
    sklearn_inertia = validation_inertia(points, labels, centers)
    absolute = abs(sklearn_inertia - project_inertia)
    relative = absolute / abs(project_inertia) if project_inertia else (0.0 if not absolute else None)
    return {"sklearn_n_iter": int(model.n_iter_), "project_update_count": project.update_count,
            "project_converged": project.converged, "cluster_id_permutation_applied": False,
            "label_agreement_count": agreement,
            "label_agreement_percent": 100.0 * agreement / len(points),
            "labels_identical": agreement == len(points),
            "max_centroid_absolute_difference": float(error.max()),
            "max_centroid_feature_scaled_difference": float((error / scales).max()),
            "project_recomputed_inertia": project_inertia,
            "sklearn_recomputed_inertia": sklearn_inertia,
            "inertia_absolute_difference": absolute,
            "inertia_relative_difference": relative,
            "sklearn_reported_inertia": float(model.inertia_),
            "sklearn_effective_dtype": str(centers.dtype)}


def native_openmp(executable: Path, points: np.ndarray, k: int,
                  expected: KMeansResult, runs: int) -> dict:
    """Reuse the existing scaling executable; all serialization is untimed."""
    n, d = points.shape
    with tempfile.TemporaryDirectory(prefix="project2_sklearn_") as temporary:
        directory = Path(temporary)
        input_path, labels_path, centers_path = (directory / suffix for suffix in
                                                ("input.f32", "labels.i32", "centroids.f32"))
        points.tofile(input_path)
        completed = subprocess.run(
            [str(executable), str(input_path), str(n), str(d), str(k),
             str(labels_path), str(centers_path), "--counts", "8", "--runs", str(runs)],
            capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError("native OpenMP comparison failed:\n" + completed.stderr)
        measured = json.loads(completed.stdout)
        if (measured["counts"] != [8] or measured["timed_runs_each"] != runs or
                len(measured["times_ms"]) != 2 or len(measured["times_ms"][1]) != runs):
            raise AssertionError("native OpenMP-8 timing settings changed")
        actual = KMeansResult(np.fromfile(labels_path, dtype=np.int32),
                              np.fromfile(centers_path, dtype=np.float32).reshape(k, d).copy(),
                              measured["update_count"], measured["converged"])
        compare_results(points, expected, actual)
    return measured


def run_case(module, name: str, points: np.ndarray, k: int, runs: int,
             openmp_executable: Path | None, environment: dict) -> dict:
    from sklearn.cluster import KMeans

    n, d = points.shape
    before = digest(points)
    initial = points[initial_row_indices(n, k)].copy(order="C")
    initial_before = digest(initial)
    oracle = fit_kmeans(points, k)
    project = result(module.kmeans_cuda(points, k))
    compare_results(points, oracle, project)
    owner = module.CudaKMeansBuffer(n, d, k)
    owner.upload(points)
    metadata_exact(project, owner.fit())
    output_exact(points, project, owner.download())
    timings = {mode: [] for mode in MODES}
    sklearn_iterations = []
    sklearn_qualities = []
    sklearn_last = None

    def one_call(mode: str, record: bool) -> None:
        nonlocal sklearn_last
        if mode == "download":
            metadata_exact(project, owner.fit())  # completion is outside download timing
        start = perf_counter()
        if mode == "sklearn_fit":
            value = KMeans(n_clusters=k, init=initial, n_init=1, algorithm="lloyd",
                           max_iter=100, tol=0.0, copy_x=True).fit(points)
        elif mode == "python_cuda_one_shot":
            value = module.kmeans_cuda(points, k)
        elif mode == "resident_fit":
            value = owner.fit()
        elif mode == "construction":
            value = module.CudaKMeansBuffer(n, d, k)
        elif mode == "upload":
            value = owner.upload(points)
        elif mode == "download":
            value = owner.download()
        else:
            raise AssertionError("unknown timing mode")
        stop = perf_counter()
        if record:
            timings[mode].append((stop - start) * 1000.0)
        if mode == "sklearn_fit":
            if value._n_threads != 8:
                raise AssertionError(f"scikit-learn used {value._n_threads} effective threads, expected 8")
            if (value.cluster_centers_.dtype != np.float32 or
                    not np.isfinite(value.cluster_centers_).all()):
                raise AssertionError("scikit-learn output dtype/finite status changed")
            sklearn_last = value
            if record:
                sklearn_iterations.append(int(value.n_iter_))
                # Parallel FP32 sums may vary without violating sklearn's
                # semantics. Record every trial's quality, not centroid parity.
                sklearn_qualities.append(quality(points, project, value))
        elif mode in ("python_cuda_one_shot", "download"):
            output_exact(points, project, value)
        elif mode == "resident_fit":
            metadata_exact(project, value)
            output_exact(points, project, owner.download())
        elif mode == "upload":
            metadata_exact(project, owner.fit())
            output_exact(points, project, owner.download())
        # Temporary CUDA owner destruction and all result comparisons are after
        # timer stop. scikit-learn models remain alive through timer stop too.
        del value

    for mode in MODES:
        one_call(mode, False)
    for trial in range(runs):
        for offset in range(len(MODES)):
            one_call(MODES[(trial + offset) % len(MODES)], True)
        if digest(points) != before or digest(initial) != initial_before:
            raise AssertionError("comparison modified input or the shared initialization")
    analyzed = quality(points, project, sklearn_last)
    analyzed["sklearn_n_iter_each_timed_fit"] = sklearn_iterations
    analyzed["sklearn_effective_threads_each_fit"] = 8
    analyzed["each_timed_sklearn_fit"] = sklearn_qualities
    analyzed["minimum_label_agreement_percent_across_trials"] = min(
        item["label_agreement_percent"] for item in sklearn_qualities)
    analyzed["maximum_scaled_centroid_difference_across_trials"] = max(
        item["max_centroid_feature_scaled_difference"] for item in sklearn_qualities)
    analyzed["maximum_absolute_inertia_difference_across_trials"] = max(
        item["inertia_absolute_difference"] for item in sklearn_qualities)
    del owner
    native = (native_openmp(openmp_executable, points, k, oracle, runs)
              if openmp_executable is not None else None)
    summarized = {mode: stats(values) for mode, values in timings.items()}
    if native is not None:
        summarized["native_openmp8"] = stats(native["times_ms"][1])
    sklearn_ms = summarized["sklearn_fit"]["median_ms"]
    ratios = {"sklearn_ms_div_python_cuda_one_shot_ms": sklearn_ms / summarized["python_cuda_one_shot"]["median_ms"],
              "sklearn_ms_div_resident_fit_ms": sklearn_ms / summarized["resident_fit"]["median_ms"]}
    if native is not None:
        ratios["sklearn_ms_div_native_openmp8_ms"] = sklearn_ms / summarized["native_openmp8"]["median_ms"]
    record = {"workload": name, "n": n, "d": d, "k": k, "input_dtype": str(points.dtype),
              "input_sha256": before, "initial_centroid_sha256": initial_before,
              "initial_rows": initial_row_indices(n, k).tolist(), "runs": runs,
              "warmup_each": 1, "timer": "time.perf_counter",
              "configuration": CONFIGURATION, "environment": environment,
              "boundaries": BOUNDARIES, "quality": analyzed, "timings": summarized,
              "ratios": ratios, "native_openmp_record": native}
    print("RAW_COMPARE " + json.dumps(record, separators=(",", ":")))
    for mode, measured in summarized.items():
        print(f"{name}/{mode}: raw={measured['raw_ms']}; "
              f"min/median/max={measured['min_ms']:.6f}/{measured['median_ms']:.6f}/"
              f"{measured['max_ms']:.6f} ms; CV={measured['cv_percent']:.3f}%")
    print(f"{name}: labels identical={analyzed['labels_identical']}; "
          f"agreement={analyzed['label_agreement_count']}/{n}; "
          f"sklearn iterations={sklearn_iterations}; Project 2 updates={project.update_count}; "
          f"inertia relative difference={analyzed['inertia_relative_difference']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--openmp-executable", type=Path)
    parser.add_argument("--mode", choices=("primary", "gpu", "iterative", "all"), default="all")
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    if not 7 <= args.runs <= 30:
        parser.error("--runs must be 7..30")
    module_dir = args.module_dir.resolve()
    if not module_dir.is_dir():
        parser.error(f"module directory does not exist: {module_dir}")
    executable = args.openmp_executable.resolve() if args.openmp_executable else None
    if executable is not None and not executable.is_file():
        parser.error(f"OpenMP executable does not exist: {executable}")
    sys.path.insert(0, str(module_dir))
    # On the tested Windows environment, importing sklearn before this CUDA
    # extension caused a native loader access violation, before any timing.
    # This import order passed the same runtime-control and regression checks;
    # the underlying DLL interaction is not established. No package is changed.
    module = importlib.import_module("kmeans_native")
    import sklearn
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    import threadpoolctl
    from threadpoolctl import threadpool_info, threadpool_limits

    modes = ("primary", "gpu", "iterative") if args.mode == "all" else (args.mode,)
    records = []
    # Limits and runtime introspection are outside every fit timer. MKL BLAS is
    # held to one thread; scikit-learn's OpenMP budget is eight, avoiding hidden
    # nested BLAS teams. Native OpenMP subprocess explicitly requests eight.
    # Setting MKL's BLAS limit may load its Intel OpenMP DLL. Apply BLAS first,
    # then discover/limit all OpenMP runtimes, including that newly loaded DLL.
    with threadpool_limits(limits=1, user_api="blas"), threadpool_limits(limits=8, user_api="openmp"):
        pools = threadpool_info()
        effective = int(_openmp_effective_n_threads())
        if effective != 8:
            raise RuntimeError(f"effective scikit-learn OpenMP budget is {effective}, expected 8")
        for pool in pools:
            if pool["user_api"] == "openmp" and pool["num_threads"] != 8:
                raise RuntimeError("an OpenMP runtime did not accept the eight-thread limit")
            if pool["user_api"] == "blas" and pool["num_threads"] != 1:
                raise RuntimeError("a BLAS runtime did not accept the one-thread limit")
        environment = {"python": platform.python_version(), "numpy": np.__version__,
                       "sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__,
                       "controlled_threadpools": pools, "sklearn_effective_openmp_threads": effective}
        print("RAW_ENVIRONMENT " + json.dumps(environment, separators=(",", ":")))
        for mode in modes:
            generated = (generate_iterative_workload() if mode == "iterative" else
                         generate_workload(WORKLOADS["profiling" if mode == "primary" else "gpu"]))
            records.append(run_case(module, mode, generated.samples, generated.spec.k,
                                    args.runs, executable, environment))
    print("workload | sklearn fit (OMP8/BLAS1) | native OpenMP8 | Python CUDA one-shot | resident fit (ms)")
    for record in records:
        timing = record["timings"]
        omp = f"{timing['native_openmp8']['median_ms']:.6f}" if "native_openmp8" in timing else "not measured"
        print(f"{record['workload']} | {timing['sklearn_fit']['median_ms']:.6f} | {omp} | "
              f"{timing['python_cuda_one_shot']['median_ms']:.6f} | {timing['resident_fit']['median_ms']:.6f}")


if __name__ == "__main__":
    main()
