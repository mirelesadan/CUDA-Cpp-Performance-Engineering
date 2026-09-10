"""Validate and optionally time the correctness-first Phase A Python bindings."""

from __future__ import annotations

import argparse
import gc
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

        resident = fourdstem_median.CudaMedianBuffer(source)
        assert resident.shape == source.shape
        try:
            resident.download()
        except RuntimeError:
            pass
        else:
            raise AssertionError("resident CUDA buffer allowed download before filter")

        resident.filter()
        resident_first = resident.download()
        assert_output_contract(resident_first, source)
        assert bitwise_equal(resident_first, expected)
        resident.filter()
        resident_second = resident.download()
        assert bitwise_equal(resident_second, expected)

        # Downloads own their storage and cannot mutate resident device output.
        resident_first.flat[0] = -123456.0
        resident_independent = resident.download()
        assert bitwise_equal(resident_independent, expected)
        del resident, resident_first, resident_second, resident_independent

        temporary_source = source.copy()
        lifetime_buffer = fourdstem_median.CudaMedianBuffer(temporary_source)
        del temporary_source
        gc.collect()
        lifetime_buffer.filter()
        lifetime_result = lifetime_buffer.download()
        assert bitwise_equal(lifetime_result, expected)
        del lifetime_buffer, lifetime_result

        device = fourdstem_median.cuda_device_info()
        print(
            "fixture one-shot/resident CUDA vs serial: bitwise exact; "
            f"device={device['name']}"
        )

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
    if hasattr(fourdstem_median, "CudaMedianBuffer"):
        invalid_resident_inputs = (
            ([0.0], "resident non-ndarray input"),
            (np.zeros((2, 2, 2), dtype=np.float64), "resident non-4D input"),
            (
                np.zeros((1, 1, 1, 1), dtype=np.float32),
                "resident non-float64 input",
            ),
            (
                np.zeros((2, 2, 2, 4), dtype=np.float64)[..., ::2],
                "resident non-C-contiguous input",
            ),
            (np.empty((0, 1, 1, 1), dtype=np.float64), "resident empty dimension"),
            (nonfinite, "resident nonfinite input"),
        )
        for value, message in invalid_resident_inputs:
            try:
                fourdstem_median.CudaMedianBuffer(value)
            except (TypeError, ValueError):
                continue
            raise AssertionError(f"binding accepted {message}")
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


def benchmark_persistent_cuda(
    source: np.ndarray, run_count: int, iteration_counts: tuple[int, ...] = (1, 2, 5, 10)
) -> None:
    if run_count < 5:
        raise ValueError("persistent CUDA benchmarking requires at least five runs")

    warmup = fourdstem_median.fixed_median_cuda(source)
    del warmup
    for iteration_count in iteration_counts:
        buffer = fourdstem_median.CudaMedianBuffer(source)
        for _ in range(iteration_count):
            buffer.filter()
        output = buffer.download()
        del buffer, output

    one_shot_times: list[float] = []
    resident_times: dict[int, dict[str, list[float]]] = {
        count: {
            "creation_upload": [],
            "kernel_sequence": [],
            "download": [],
            "release": [],
            "total": [],
        }
        for count in iteration_counts
    }
    configurations: list[int | None] = [None, *iteration_counts]
    for run in range(run_count):
        ordered = configurations if run % 2 == 0 else list(reversed(configurations))
        for iteration_count in ordered:
            if iteration_count is None:
                start = perf_counter()
                output = fourdstem_median.fixed_median_cuda(source)
                one_shot_times.append(perf_counter() - start)
                del output
                continue

            measurements = resident_times[iteration_count]
            total_start = perf_counter()
            start = perf_counter()
            buffer = fourdstem_median.CudaMedianBuffer(source)
            after_creation = perf_counter()
            for _ in range(iteration_count):
                buffer.filter()
            after_kernels = perf_counter()
            output = buffer.download()
            after_download = perf_counter()
            del buffer
            after_release = perf_counter()

            measurements["creation_upload"].append(after_creation - start)
            measurements["kernel_sequence"].append(after_kernels - after_creation)
            measurements["download"].append(after_download - after_kernels)
            measurements["release"].append(after_release - after_download)
            measurements["total"].append(after_release - total_start)
            del output
        gc.collect()

    print(
        f"one-shot CUDA Python workflow: raw_s={one_shot_times}; "
        f"median_s={median(one_shot_times):.6f}; "
        f"min_s={min(one_shot_times):.6f}; max_s={max(one_shot_times):.6f}"
    )
    for iteration_count in iteration_counts:
        measurements = resident_times[iteration_count]
        total_times = measurements["total"]
        effective_times = [value / iteration_count for value in total_times]
        print(
            f"persistent CUDA {iteration_count} operation(s): "
            f"creation_upload_raw_s={measurements['creation_upload']}; "
            f"kernel_sequence_raw_s={measurements['kernel_sequence']}; "
            f"download_raw_s={measurements['download']}; "
            f"release_raw_s={measurements['release']}; "
            f"total_raw_s={total_times}; median_total_s={median(total_times):.6f}; "
            f"effective_raw_s={effective_times}; "
            f"median_effective_s={median(effective_times):.6f}"
        )


def validate_and_time_canonical(
    input_path: Path,
    thread_count: int,
    run_count: int,
    run_persistent_cuda_benchmark: bool,
) -> None:
    source = np.load(input_path, allow_pickle=False)
    result_serial = fourdstem_median.fixed_median_serial(source)
    result_openmp = fourdstem_median.fixed_median_openmp(source, thread_count)
    assert bitwise_equal(result_openmp, result_serial)
    if hasattr(fourdstem_median, "fixed_median_cuda"):
        result_cuda = fourdstem_median.fixed_median_cuda(source)
        assert bitwise_equal(result_cuda, result_serial)
        del result_cuda
        resident = fourdstem_median.CudaMedianBuffer(source)
        resident.filter()
        result_resident = resident.download()
        assert bitwise_equal(result_resident, result_serial)
        del resident, result_resident

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

    print(f"canonical CPU/one-shot/resident paths: {source.size} values bitwise exact")
    del result_openmp, result_serial

    if run_persistent_cuda_benchmark:
        if not hasattr(fourdstem_median, "CudaMedianBuffer"):
            raise RuntimeError("persistent CUDA benchmark requires a CUDA-enabled module")
        benchmark_persistent_cuda(source, run_count)

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
    parser.add_argument("--persistent-cuda-benchmark", action="store_true")
    arguments = parser.parse_args()
    if arguments.thread_count < 1 or arguments.timing_runs < 1:
        parser.error("thread count and timing runs must be positive")

    validate_fixture(arguments.thread_count)
    if arguments.canonical_input is not None:
        validate_and_time_canonical(
            arguments.canonical_input,
            arguments.thread_count,
            arguments.timing_runs,
            arguments.persistent_cuda_benchmark,
        )


if __name__ == "__main__":
    main()
