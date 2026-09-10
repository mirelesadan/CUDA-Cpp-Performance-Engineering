"""Validate and optionally time the correctness-first Phase A Python bindings."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import fourdstem_median


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
FIXTURE_DIRECTORY = PROJECT_DIRECTORY / "reference_data" / "public_synthetic"


def bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def assert_output_contract(result: np.ndarray, source: np.ndarray) -> None:
    assert isinstance(result, np.ndarray)
    assert result.shape == source.shape
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.c_contiguous
    assert result.flags.owndata
    assert not np.shares_memory(result, source)


def expect_rejection(value: object, message: str) -> None:
    try:
        fourdstem_median.fixed_median_serial(value)
    except (TypeError, ValueError):
        return
    raise AssertionError(f"binding accepted {message}")


def validate_fixture(thread_count: int) -> None:
    source = np.load(FIXTURE_DIRECTORY / "reference_input.npy", allow_pickle=False)
    expected = np.load(
        FIXTURE_DIRECTORY / "reference_output_python.npy", allow_pickle=False
    )
    source_before = source.copy()

    result_serial = fourdstem_median.fixed_median_serial(source)
    result_openmp = fourdstem_median.fixed_median_openmp(source, thread_count)
    assert_output_contract(result_serial, source)
    assert_output_contract(result_openmp, source)
    assert bitwise_equal(result_serial, expected)
    assert bitwise_equal(result_openmp, result_serial)

    if hasattr(fourdstem_median, "_benchmark_fixed_median_serial_copied"):
        copied_serial = fourdstem_median._benchmark_fixed_median_serial_copied(source)
        copied_openmp = fourdstem_median._benchmark_fixed_median_openmp_copied(
            source, thread_count
        )
        assert bitwise_equal(copied_serial, result_serial)
        assert bitwise_equal(copied_openmp, result_openmp)

    if hasattr(fourdstem_median, "fixed_median_cuda"):
        result_cuda = fourdstem_median.fixed_median_cuda(source)
        assert_output_contract(result_cuda, source)
        assert bitwise_equal(result_cuda, result_serial)
        device = fourdstem_median.cuda_device_info()
        print(f"fixture CUDA vs serial: bitwise exact; device={device['name']}")

    assert bitwise_equal(source, source_before)
    expect_rejection([0.0], "non-ndarray input")
    expect_rejection(np.zeros((2, 2, 2), dtype=np.float64), "non-4D input")
    expect_rejection(np.zeros((1, 1, 1, 1), dtype=np.float32), "non-float64 input")
    expect_rejection(
        np.zeros((2, 2, 2, 4), dtype=np.float64)[..., ::2],
        "non-C-contiguous input",
    )
    expect_rejection(np.empty((0, 1, 1, 1), dtype=np.float64), "empty dimension")
    nonfinite = np.zeros((1, 1, 1, 1), dtype=np.float64)
    nonfinite[0, 0, 0, 0] = np.nan
    expect_rejection(nonfinite, "nonfinite input")
    print(f"fixture serial/OpenMP: {source.size} values bitwise exact")
    print("input/output and invalid-input contract: passed")


def timed_paths(
    operations: dict[str, Callable[[], object]], run_count: int
) -> None:
    for operation in operations.values():
        warmup_result = operation()
        del warmup_result

    elapsed_seconds: dict[str, list[float]] = {name: [] for name in operations}
    items = list(operations.items())
    for run in range(run_count):
        ordered_items = items if run % 2 == 0 else list(reversed(items))
        for name, operation in ordered_items:
            start = perf_counter()
            result = operation()
            elapsed_seconds[name].append(perf_counter() - start)
            del result

    for name, times in elapsed_seconds.items():
        print(
            f"{name}: raw_s={times}; median_s={median(times):.6f}; "
            f"min_s={min(times):.6f}; max_s={max(times):.6f}"
        )


def validate_and_time_canonical(
    input_path: Path, thread_count: int, run_count: int
) -> None:
    source = np.load(input_path, allow_pickle=False)
    result_serial = fourdstem_median.fixed_median_serial(source)
    result_openmp = fourdstem_median.fixed_median_openmp(source, thread_count)
    assert bitwise_equal(result_openmp, result_serial)
    if hasattr(fourdstem_median, "fixed_median_cuda"):
        result_cuda = fourdstem_median.fixed_median_cuda(source)
        assert bitwise_equal(result_cuda, result_serial)
        del result_cuda

    copied_paths_available = hasattr(
        fourdstem_median, "_benchmark_fixed_median_serial_copied"
    )
    if copied_paths_available:
        copied_serial = fourdstem_median._benchmark_fixed_median_serial_copied(source)
        assert bitwise_equal(copied_serial, result_serial)
        del copied_serial
        copied_openmp = fourdstem_median._benchmark_fixed_median_openmp_copied(
            source, thread_count
        )
        assert bitwise_equal(copied_openmp, result_openmp)
        del copied_openmp

    print(f"canonical direct/vector CPU paths: {source.size} values bitwise exact")
    del result_openmp, result_serial

    operations = {
        "direct serial Python call": lambda: fourdstem_median.fixed_median_serial(source),
        f"direct OpenMP-{thread_count} Python call": lambda: (
            fourdstem_median.fixed_median_openmp(source, thread_count)
        ),
    }
    if copied_paths_available:
        operations = {
            "copied serial Python call": lambda: (
                fourdstem_median._benchmark_fixed_median_serial_copied(source)
            ),
            "direct serial Python call": operations["direct serial Python call"],
            f"copied OpenMP-{thread_count} Python call": lambda: (
                fourdstem_median._benchmark_fixed_median_openmp_copied(
                    source, thread_count
                )
            ),
            f"direct OpenMP-{thread_count} Python call": operations[
                f"direct OpenMP-{thread_count} Python call"
            ],
        }
    timed_paths(operations, run_count)

    if hasattr(fourdstem_median, "_benchmark_validate_finite"):
        timed_paths(
            {
                "finite validation only": lambda: (
                    fourdstem_median._benchmark_validate_finite(source)
                )
            },
            run_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-input", type=Path)
    parser.add_argument("--thread-count", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timing-runs", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.thread_count < 1 or arguments.timing_runs < 1:
        parser.error("thread count and timing runs must be positive")

    validate_fixture(arguments.thread_count)
    if arguments.canonical_input is not None:
        validate_and_time_canonical(
            arguments.canonical_input,
            arguments.thread_count,
            arguments.timing_runs,
        )


if __name__ == "__main__":
    main()
