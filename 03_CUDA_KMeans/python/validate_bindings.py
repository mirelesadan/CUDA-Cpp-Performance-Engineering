"""Public contract/state tests for the opt-in kmeans_native CUDA extension.

Run with --module-dir <Release extension directory>. Public fixtures and small
edges run by default; --workloads additionally checks primary/GPU/iterative
inputs and a lightweight GIL sanity test. These checks do not benchmark.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import threading
from time import perf_counter
import unittest

import numpy as np

from benchmark_cuda import edge_cases
from benchmark_data import WORKLOADS, generate_iterative_workload, generate_workload
from reference import (KMeansResult, _fit_with_update_cap, compare_results,
                       fit_kmeans, validation_inertia)
from test_reference import public_cases


native = None


def as_result(value: dict, n: int, d: int, k: int) -> KMeansResult:
    if not isinstance(value, dict) or set(value) != {
            "labels", "centroids", "update_count", "converged"}:
        raise AssertionError("result must contain exactly four documented fields")
    if type(value["update_count"]) is not int or type(value["converged"]) is not bool:
        raise AssertionError("metadata must be Python int/bool")
    for name, dtype, shape in (("labels", np.int32, (n,)),
                               ("centroids", np.float32, (k, d))):
        array = value[name]
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype) or
                array.shape != shape or not array.flags.c_contiguous or
                not array.flags.owndata):
            raise AssertionError(f"{name}: wrong type, shape, layout, or ownership")
    if np.shares_memory(value["labels"], value["centroids"]):
        raise AssertionError("result arrays must have independent allocations")
    return KMeansResult(**value)


def fit_metadata(value: dict, expected: KMeansResult) -> None:
    if not isinstance(value, dict) or set(value) != {"update_count", "converged"}:
        raise AssertionError("fit() must return only update count and convergence")
    if type(value["update_count"]) is not int or type(value["converged"]) is not bool:
        raise AssertionError("fit metadata must be Python int/bool")
    if (value["update_count"], value["converged"]) != (
            expected.update_count, expected.converged):
        raise AssertionError("fit metadata differs from reference")


def same_result(points: np.ndarray, expected: KMeansResult,
                value: dict) -> tuple[KMeansResult, int]:
    result = as_result(value, *points.shape, len(expected.centroids))
    compare_results(points, expected, result)
    bit_differences = int(np.count_nonzero(
        result.centroids.view(np.uint32) != expected.centroids.view(np.uint32)))
    return result, bit_differences


def one_shot(points: np.ndarray, k: int, cap: int = 100) -> dict:
    if cap == 100:
        return native.kmeans_cuda(points, k)
    return native._kmeans_cuda_with_update_cap(points, k, cap)


def owner_fit(owner, cap: int = 100) -> dict:
    return owner.fit() if cap == 100 else owner._fit_with_update_cap(cap)


def check_case(name: str, points: np.ndarray, k: int, *, cap: int = 100,
               expected_case: dict | None = None, repeated: int = 3) -> None:
    before = points.view(np.uint32).copy()
    expected = fit_kmeans(points, k) if cap == 100 else _fit_with_update_cap(points, k, cap)
    first, bits = same_result(points, expected, one_shot(points, k, cap))
    second, again_bits = same_result(points, expected, one_shot(points, k, cap))
    first_centroid_bits = first.centroids.view(np.uint32).copy()
    np.testing.assert_array_equal(second.centroids.view(np.uint32), first_centroid_bits)

    def check_resident(value: dict) -> tuple[KMeansResult, int]:
        result, differences = same_result(points, expected, value)
        np.testing.assert_array_equal(result.centroids.view(np.uint32), first_centroid_bits)
        return result, differences

    if (np.shares_memory(first.labels, second.labels) or
            np.shares_memory(first.centroids, second.centroids)):
        raise AssertionError("one-shot calls share output storage")
    if expected_case is not None:
        frozen = KMeansResult(np.asarray(expected_case["expected_labels"], dtype=np.int32),
                              np.asarray(expected_case["expected_centroids"], dtype=np.float32),
                              expected_case["expected_update_count"],
                              expected_case["expected_converged"])
        compare_results(points, frozen, first)
        inertia = validation_inertia(points, first.labels, first.centroids)
        if abs(inertia - expected_case["expected_inertia"]) > 2e-5 * max(
                1.0, abs(expected_case["expected_inertia"])):
            raise AssertionError("frozen inertia tolerance exceeded")

    owner = native.CudaKMeansBuffer(*points.shape, k)
    if owner.upload(points) is not None:
        raise AssertionError("upload must not return a fit or host output")
    for _ in range(repeated):
        fit_metadata(owner_fit(owner, cap), expected)
        _, resident_bits = check_resident(owner.download())
        bits += resident_bits
    # Three independent fits without an intervening bulk download.
    for _ in range(3):
        fit_metadata(owner_fit(owner, cap), expected)
    original, resident_bits = check_resident(owner.download())
    bits += resident_bits
    duplicate, resident_bits = check_resident(owner.download())
    bits += resident_bits
    if (np.shares_memory(original.labels, duplicate.labels) or
            np.shares_memory(original.centroids, duplicate.centroids)):
        raise AssertionError("repeated downloads share host storage")
    original.labels.fill(-1)
    original.centroids.fill(np.float32(1024))
    check_resident(owner.download())
    check_resident({
        "labels": duplicate.labels, "centroids": duplicate.centroids,
        "update_count": duplicate.update_count, "converged": duplicate.converged})

    # Exact negation changes the uploaded snapshot and keeps values in bounds.
    replacement = np.negative(points)
    replacement_before = replacement.view(np.uint32).copy()
    replacement_expected = (fit_kmeans(replacement, k) if cap == 100 else
                            _fit_with_update_cap(replacement, k, cap))
    replacement_one_shot, _ = same_result(
        replacement, replacement_expected, one_shot(replacement, k, cap))
    owner.upload(replacement)
    try:
        owner.download()
    except RuntimeError:
        pass
    else:
        raise AssertionError("successful upload did not invalidate prior output")
    fit_metadata(owner_fit(owner, cap), replacement_expected)
    replacement_result, replacement_bits = same_result(
        replacement, replacement_expected, owner.download())
    np.testing.assert_array_equal(replacement_result.centroids.view(np.uint32),
                                  replacement_one_shot.centroids.view(np.uint32))
    owner.upload(points)
    fit_metadata(owner_fit(owner, cap), expected)
    check_resident(owner.download())
    np.testing.assert_array_equal(points.view(np.uint32), before)
    np.testing.assert_array_equal(replacement.view(np.uint32), replacement_before)
    print(f"{name}: {len(points)} exact labels/termination; frozen centroid/inertia "
          f"tolerances passed; centroid-bit differences={bits + again_bits + replacement_bits}; "
          f"{repeated} fits with downloads + 3 fits without; resident bits match one-shot; "
          "X1/X2/X1 exact; "
          "input unchanged and output allocations independent")


class PythonCudaContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.array([[-2.0, -1.0]] * 8 + [[2.0, 1.0]] * 8,
                               dtype=np.float32)

    def test_public_fixtures(self) -> None:
        for case in public_cases():
            with self.subTest(case=case["name"]):
                check_case(case["name"], np.array(case["points"], dtype=np.float32,
                                                 order="C"), case["k"],
                           cap=case["max_updates"], expected_case=case)

    def test_existing_edges(self) -> None:
        cases = edge_cases()
        for n in (1023, 1024, 1025, 4097, 8193):
            d, k = (32, 32) if n in (1025, 4097) else (3, 2)
            points = np.empty((n, d), dtype=np.float32)
            for row in range(n):
                points[row] = (row * k // n) * 8 + np.arange(d, dtype=np.float32) * 0.03125
            cases.append((f"edge_tiled_n{n}", points, k))
        cases.append(("partial_tile_identical_k32",
                      np.full((1025, 3), -0.0, dtype=np.float32), 32))
        cases.append(("minimum_n_maximum_d_magnitude_bounds",
                      np.array([[-1024.0] * 32, [1024.0] * 32], dtype=np.float32), 2))
        for name, points, k in cases:
            with self.subTest(case=name):
                check_case(name, points, k)

    def test_lifecycle_and_snapshot(self) -> None:
        points = self.points.copy()
        before = points.view(np.uint32).copy()
        expected = fit_kmeans(points, 2)
        owner = native.CudaKMeansBuffer(*points.shape, 2)
        with self.assertRaises(RuntimeError):
            owner.fit()
        with self.assertRaises(RuntimeError):
            owner.download()
        owner.upload(points)
        with self.assertRaises(RuntimeError):
            owner.download()
        points.fill(42)
        fit_metadata(owner.fit(), expected)
        same_result(before.view(np.float32), expected, owner.download())
        # The owner retains the uploaded snapshot, not a borrowed NumPy buffer.
        replacement = self.points * np.float32(2)
        owner.upload(replacement)
        with self.assertRaises(RuntimeError):
            owner.download()
        replacement_expected = fit_kmeans(replacement, 2)
        fit_metadata(owner.fit(), replacement_expected)
        same_result(replacement, replacement_expected, owner.download())

    def test_invalid_constructor_parameters(self) -> None:
        for args in ((0, 2, 2), (1, 2, 2), (2**20 + 1, 2, 2),
                     (16, 0, 2), (16, 33, 2), (16, 2, 1), (16, 2, 17),
                     (64, 2, 33), (-16, 2, 2), (16, -2, 2), (16, 2, -2),
                     (True, 2, 2), (16, True, 2), (16, 2, True),
                     (16.0, 2, 2), (16, 2.0, 2), (16, 2, 2.0),
                     ("16", 2, 2), (2**100, 2, 2)):
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError, OverflowError)):
                    native.CudaKMeansBuffer(*args)

    def test_invalid_input_and_upload_preservation(self) -> None:
        good = self.points
        owner = native.CudaKMeansBuffer(*good.shape, 2)
        expected = fit_kmeans(good, 2)
        owner.upload(good)
        fit_metadata(owner.fit(), expected)
        invalid = [good.tolist(), good[:, 0], np.zeros((2, 2, 2), dtype=np.float32),
                   good.astype(np.float64),
                   good.astype(">f4"), good[:, ::-1], np.asfortranarray(good),
                   np.zeros((0, 2), dtype=np.float32),
                   np.zeros((16, 0), dtype=np.float32),
                   np.zeros((16, 33), dtype=np.float32),
                   np.zeros((2**20 + 1, 1), dtype=np.float32),
                   np.zeros((15, 2), dtype=np.float32)]
        for bad in (np.nan, np.inf, -np.inf, 1024.25, -1024.25):
            value = good.copy()
            value[0, 0] = bad
            invalid.append(value)
        for index, value in enumerate(invalid):
            with self.subTest(index=index):
                with self.assertRaises((TypeError, ValueError)):
                    owner.upload(value)
                same_result(good, expected, owner.download())
                # Different positive shapes are legal for one-shot; all other
                # rejected uploads also violate its documented input contract.
                if isinstance(value, np.ndarray) and value.shape == (15, 2):
                    continue
                with self.assertRaises((TypeError, ValueError)):
                    native.kmeans_cuda(value, 2)
        for k in (1, 17, True, False, 2.0, "2", -2, 2**100):
            with self.subTest(k=k):
                with self.assertRaises((TypeError, ValueError, OverflowError)):
                    native.kmeans_cuda(good, k)
        for cap in (0, 101, True, 1.0):
            with self.subTest(cap=cap):
                with self.assertRaises((TypeError, ValueError)):
                    native._kmeans_cuda_with_update_cap(good, 2, cap)
                with self.assertRaises((TypeError, ValueError)):
                    owner._fit_with_update_cap(cap)
                same_result(good, expected, owner.download())
        fit_metadata(owner.fit(), expected)
        same_result(good, expected, owner.download())

    def test_readonly_and_unaligned_contiguous_input(self) -> None:
        readonly = self.points.copy()
        readonly.flags.writeable = False
        unaligned = np.ndarray(self.points.shape, dtype=np.float32,
                               buffer=bytearray(self.points.nbytes + 1), offset=1)
        unaligned[:] = self.points
        self.assertTrue(unaligned.flags.c_contiguous)
        self.assertFalse(unaligned.flags.aligned)
        for name, points in (("readonly", readonly), ("unaligned", unaligned)):
            with self.subTest(case=name):
                check_case(name, points, 2)


def check_gil_progress(points: np.ndarray, k: int) -> None:
    """Counter progress while a resident CUDA fit releases the interpreter lock.

    The switch interval exceeds the expected call duration. The main thread
    enables the worker immediately before fit and disables it immediately on
    return, so ordinary Python scheduling outside native work cannot satisfy
    the check. The worker's 50-ms bound prevents interpreter starvation.
    """
    owner = native.CudaKMeansBuffer(*points.shape, k)
    owner.upload(points)
    owner.fit()  # Initialize CUDA before the concurrency-only sanity check.
    ready, active, stop = threading.Event(), threading.Event(), threading.Event()
    counter = [0]
    worker_errors = []
    busy_check = []

    def worker() -> None:
        ready.wait()
        deadline = perf_counter() + 0.05
        begin = perf_counter()
        try:
            owner.download()
        except RuntimeError as error:
            elapsed = perf_counter() - begin
            if "busy" not in str(error).lower():
                worker_errors.append(f"overlapping call raised the wrong error: {error}")
            elif elapsed >= 0.05:
                worker_errors.append("overlapping owner call did not reject immediately")
            else:
                busy_check.append(True)
        except Exception as error:
            worker_errors.append(f"overlapping call raised {type(error).__name__}: {error}")
        else:
            worker_errors.append("overlapping owner.download() was not rejected")
        while not stop.is_set() and active.is_set() and perf_counter() < deadline:
            counter[0] += 1

    previous_interval = sys.getswitchinterval()
    thread = threading.Thread(target=worker, name="kmeans-gil-sanity", daemon=True)
    thread.start()
    try:
        sys.setswitchinterval(1.0)
        active.set()
        ready.set()
        owner.fit()
        active.clear()
        stop.set()
    finally:
        stop.set()
        ready.set()
        sys.setswitchinterval(previous_interval)
        thread.join(timeout=2.0)
    if worker_errors:
        raise AssertionError("; ".join(worker_errors))
    if thread.is_alive() or counter[0] == 0 or not busy_check:
        raise AssertionError("CPU Python worker made no progress during resident fit")
    print(f"GIL sanity: Python worker made {counter[0]} increments during resident fit; "
          "overlapping same-owner download immediately rejected as busy; "
          "concurrency check excluded from performance measurements")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--workloads", action="store_true",
                        help="also validate primary/GPU/iterative and GIL release")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_dir.resolve()))
    global native
    native = importlib.import_module("kmeans_native")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PythonCudaContractTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    if args.workloads:
        for name in ("primary", "gpu", "iterative"):
            generated = (generate_iterative_workload() if name == "iterative" else
                         generate_workload(WORKLOADS["profiling" if name == "primary" else "gpu"]))
            check_case(name, generated.samples, generated.spec.k, repeated=20)
            if name == "gpu":
                check_gil_progress(generated.samples, generated.spec.k)
    print("Python CUDA integration validation passed")


if __name__ == "__main__":
    main()
