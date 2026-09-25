"""Public, deterministic Project 2 semantic and contract self-tests.

Run from the repository root with:
    python -B 03_CUDA_KMeans/python/test_reference.py
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from benchmark_data import (GENERATOR_VERSION, ITERATIVE_PROFILE_SPEC, WORKLOADS,
                            WorkloadSpec, generate_iterative_workload,
                            generate_workload)
from reference import (CONTRACT_VERSION, _assign_fp32, _fit_with_update_cap,
                       _update_float64, compare_results, fit_kmeans,
                       initial_row_indices, validation_inertia)


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "public_cases.json"


def public_cases() -> list[dict]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document["schema_version"] != 1:
        raise AssertionError("unsupported public fixture schema")
    if document["reference_contract"] != CONTRACT_VERSION:
        raise AssertionError("public fixture/reference contract version mismatch")
    return document["cases"]


class KMeansReferenceTests(unittest.TestCase):
    def test_iterative_profiling_workload_is_reproducible(self) -> None:
        generated = generate_iterative_workload()
        self.assertEqual(generated.spec, ITERATIVE_PROFILE_SPEC)
        self.assertEqual(generated.samples.shape, (16_384, 8))
        self.assertEqual(generated.samples.dtype, np.float32)
        self.assertTrue(generated.samples.flags.c_contiguous)
        self.assertEqual(
            hashlib.sha256(generated.samples.tobytes()).hexdigest(),
            "97359ddaf01fe506a28efc48159760324af6db9226d56e5a6624b20d62c81904",
        )
        result = fit_kmeans(generated.samples, generated.spec.k)
        self.assertEqual(result.update_count, 22)
        self.assertTrue(result.converged)

    def test_seeded_workload_generator_without_benchmarking(self) -> None:
        self.assertEqual(GENERATOR_VERSION, 1)
        self.assertEqual((WORKLOADS["profiling"].n, WORKLOADS["profiling"].d,
                          WORKLOADS["profiling"].k), (65_536, 8, 16))
        self.assertEqual((WORKLOADS["gpu"].n, WORKLOADS["gpu"].d,
                          WORKLOADS["gpu"].k), (262_144, 16, 16))
        self.assertEqual((WORKLOADS["stress_optional"].n, WORKLOADS["stress_optional"].d,
                          WORKLOADS["stress_optional"].k), (1_048_576, 32, 32))
        spec = WorkloadSpec("smoke", 256, 8, 16, 20_260_924)
        first = generate_workload(spec)
        again = generate_workload(spec)
        np.testing.assert_array_equal(first.samples.view(np.uint32),
                                      again.samples.view(np.uint32))
        np.testing.assert_array_equal(first.planted_labels, again.planted_labels)
        self.assertEqual(first.samples.shape, (256, 8))
        self.assertEqual(first.samples.dtype, np.float32)
        self.assertTrue(first.samples.flags.c_contiguous)
        self.assertTrue(np.isfinite(first.samples).all())
        self.assertLessEqual(float(np.max(np.abs(first.samples))), 1024.0)
        for cluster, row in enumerate(initial_row_indices(256, 16)):
            np.testing.assert_array_equal(first.samples[row], first.planted_centers[cluster])
            self.assertEqual(int(first.planted_labels[row]), cluster)
        fitted = fit_kmeans(first.samples, spec.k)
        np.testing.assert_array_equal(fitted.labels, first.planted_labels)
        self.assertTrue(fitted.converged)

    def test_frozen_public_fixtures(self) -> None:
        for case in public_cases():
            with self.subTest(case=case["name"]):
                x = np.array(case["points"], dtype=np.float32, order="C")
                before = x.view(np.uint32).copy()
                k = case["k"]
                np.testing.assert_array_equal(initial_row_indices(len(x), k),
                                              case["initial_rows"])
                run = (fit_kmeans if case["max_updates"] == 100 else
                       lambda points, clusters: _fit_with_update_cap(
                           points, clusters, case["max_updates"]))
                result = run(x, k)
                again = run(x, k)
                expected_labels = np.array(case["expected_labels"], dtype=np.int32)
                expected_centroids = np.array(case["expected_centroids"], dtype=np.float32)

                self.assertEqual(result.labels.shape, (len(x),))
                self.assertEqual(result.centroids.shape, (k, x.shape[1]))
                self.assertEqual(result.labels.dtype, np.dtype("int32"))
                self.assertEqual(result.centroids.dtype, np.dtype("float32"))
                self.assertTrue(result.labels.flags.c_contiguous and result.labels.flags.owndata)
                self.assertTrue(result.centroids.flags.c_contiguous and
                                result.centroids.flags.owndata)
                self.assertFalse(np.shares_memory(x, result.labels))
                self.assertFalse(np.shares_memory(x, result.centroids))
                self.assertFalse(np.shares_memory(result.labels, again.labels))
                self.assertFalse(np.shares_memory(result.centroids, again.centroids))
                np.testing.assert_array_equal(x.view(np.uint32), before)
                np.testing.assert_array_equal(result.labels, expected_labels)
                np.testing.assert_array_equal(result.centroids.view(np.uint32),
                                              expected_centroids.view(np.uint32))
                self.assertEqual(result.update_count, case["expected_update_count"])
                self.assertIs(result.converged, case["expected_converged"])
                self.assertAlmostEqual(validation_inertia(x, result.labels, result.centroids),
                                       case["expected_inertia"], delta=1e-12 *
                                       max(1.0, case["expected_inertia"]))
                comparison = compare_results(x, result, again)
                self.assertEqual(comparison.max_centroid_absolute_error, 0.0)
                self.assertEqual(comparison.max_centroid_scaled_error, 0.0)

    def test_initial_indices_use_integer_formula(self) -> None:
        np.testing.assert_array_equal(initial_row_indices(16, 2), [4, 12])
        np.testing.assert_array_equal(initial_row_indices(16, 3), [2, 8, 13])
        expected = [((2 * cluster + 1) * (1 << 20)) // 64 for cluster in range(32)]
        np.testing.assert_array_equal(initial_row_indices(1 << 20, 32), expected)

    def test_exact_tie_and_empty_cluster_are_reachable(self) -> None:
        case = next(item for item in public_cases() if item["name"].startswith("exact_tie"))
        x = np.array(case["points"], dtype=np.float32)
        seeds = x[initial_row_indices(len(x), 3)].copy()
        np.testing.assert_array_equal(seeds[:, 0], [3, 3, 9])
        self.assertEqual(int(_assign_fp32(x, seeds)[6]), 0)
        result = fit_kmeans(x, 3)
        self.assertEqual(int(np.count_nonzero(result.labels == 1)), 0)
        self.assertEqual(float(result.centroids[1, 0]), 3.0)

    def test_fp32_assignment_differs_from_fp64(self) -> None:
        case = next(item for item in public_cases() if item["name"] == "fp32_distance_rounding")
        x = np.array(case["points"], dtype=np.float32)
        seeds = x[initial_row_indices(len(x), 2)].copy()
        self.assertEqual(int(_assign_fp32(x, seeds)[7]), 0)
        double_distances = np.sum((x[7].astype(np.float64) -
                                   seeds.astype(np.float64)) ** 2, axis=1)
        self.assertLess(double_distances[1], double_distances[0])

    def test_feature_accumulation_order_is_ascending(self) -> None:
        point = np.zeros((1, 3), dtype=np.float32)
        centers = np.array([[1, 2 ** -12, 2 ** -12], [1, 0, 0]], dtype=np.float32)
        # 1 + 2^-24 rounds to 1 in FP32 twice. Adding the small terms
        # together first would instead produce 1 + 2^-23 and choose center 1.
        self.assertEqual(int(_assign_fp32(point, centers)[0]), 0)

    def test_centroid_update_accumulates_in_float64(self) -> None:
        # The seven small values vanish when added after 1.0 in FP32, but
        # affect the correctly accumulated FP64 mean after its final cast.
        x = np.array([[1.0]] + [[2 ** -25]] * 7 + [[4.0]] * 8,
                     dtype=np.float32)
        labels = np.array([0] * 8 + [1] * 8, dtype=np.int32)
        previous = np.array([[0.0], [4.0]], dtype=np.float32)
        updated = _update_float64(x, labels, previous)
        expected = np.float32((1.0 + 7 * 2 ** -25) / 8)
        self.assertEqual(updated[0, 0], expected)
        self.assertNotEqual(updated[0, 0], np.float32(1.0 / 8))

    def test_assignment_chunk_boundary(self) -> None:
        points = np.zeros((2050, 2), dtype=np.float32)
        points[2048:] = 1.0
        centers = np.array([[0, 0], [1, 1]], dtype=np.float32)
        labels = _assign_fp32(points, centers)
        np.testing.assert_array_equal(labels[:2048], 0)
        np.testing.assert_array_equal(labels[2048:], 1)

    def test_reduced_update_cap_returns_final_reassignment(self) -> None:
        case = next(item for item in public_cases() if item["name"] == "update_cap_without_hidden_update")
        x = np.array(case["points"], dtype=np.float32)
        result = _fit_with_update_cap(x, 2, max_updates=1)
        self.assertEqual(result.update_count, 1)
        self.assertFalse(result.converged)
        np.testing.assert_array_equal(result.labels, _assign_fp32(x, result.centroids))
        self.assertNotEqual(float(result.centroids[0, 0]),
                            float(np.mean(x[result.labels == 0, 0], dtype=np.float64)))

    def test_zero_inertia_exactly_representable(self) -> None:
        x = np.array([[-2.0]] * 8 + [[2.0]] * 8, dtype=np.float32)
        result = fit_kmeans(x, 2)
        self.assertEqual(validation_inertia(x, result.labels, result.centroids), 0.0)

    def test_comparator_rejects_discrete_and_numerical_mismatches(self) -> None:
        x = np.array([[0.0]] * 8 + [[2.0]] * 8, dtype=np.float32)
        reference = fit_kmeans(x, 2)
        wrong_labels = reference.labels.copy()
        wrong_labels[0] = 1
        with self.assertRaisesRegex(AssertionError, "label mismatches"):
            compare_results(x, reference, replace(reference, labels=wrong_labels))
        with self.assertRaisesRegex(AssertionError, "update count"):
            compare_results(x, reference, replace(reference, update_count=2))
        wrong_centroids = reference.centroids.copy()
        wrong_centroids[0, 0] = 0.01
        with self.assertRaisesRegex(AssertionError, "centroid scaled error"):
            compare_results(x, reference, replace(reference, centroids=wrong_centroids))
        nan_centroids = reference.centroids.copy()
        nan_centroids[0, 0] = np.nan
        with self.assertRaisesRegex(AssertionError, "centroids must be finite"):
            compare_results(x, reference, replace(reference, centroids=nan_centroids))

    def test_invalid_inputs_are_rejected_without_coercion(self) -> None:
        good = np.zeros((16, 2), dtype=np.float32)
        invalid = [
            (good.tolist(), 2),
            (good[:, 0], 2),
            (good.astype(np.float64), 2),
            (good[:, ::-1], 2),
            (np.zeros((0, 2), dtype=np.float32), 2),
            (np.zeros((16, 0), dtype=np.float32), 2),
            (np.zeros((16, 33), dtype=np.float32), 2),
            (good, 1),
            (good, 17),
            (good, True),
        ]
        for value, k in invalid:
            with self.subTest(kind=type(value).__name__, k=k):
                with self.assertRaises((TypeError, ValueError)):
                    fit_kmeans(value, k)
        with self.assertRaises(TypeError):
            fit_kmeans(good, 2, 1)
        for cap in (0, 101, False):
            with self.assertRaises((TypeError, ValueError)):
                _fit_with_update_cap(good, 2, cap)
        for bad_value in (np.nan, np.inf, -np.inf, 1024.25, -1024.25):
            value = good.copy()
            value[0, 0] = bad_value
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    fit_kmeans(value, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
