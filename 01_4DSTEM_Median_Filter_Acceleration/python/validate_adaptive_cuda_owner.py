"""Validate and time the Phase B persistent adaptive CUDA Python owner."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

import fourdstem_median


PROJECT = Path(__file__).resolve().parents[1]
PUBLIC = PROJECT / "reference_data" / "public_adaptive"
LOCAL = PROJECT / "reference_data" / "adaptive_median"
COUNTS = (1, 2, 5, 10, 20)


def load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def require_output(output: np.ndarray, source: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(output, np.ndarray)
    assert output.shape == source.shape
    assert output.dtype == np.dtype(np.float64)
    assert output.flags.c_contiguous and output.flags.owndata
    assert not np.shares_memory(output, source)
    assert exact(output, expected)


def rejects(label: str, action: Callable[[], object]) -> None:
    try:
        action()
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return
    raise AssertionError(f"accepted invalid {label}")


def validate_contract(source: np.ndarray, expected: np.ndarray) -> None:
    buffer = fourdstem_median.CudaAdaptiveMedianBuffer(source.shape)
    rejects("filter before upload", buffer.filter)
    rejects("download before filter", buffer.download)
    rejects("non-array input", lambda: buffer.upload(source.tolist()))
    rejects("wrong rank", lambda: buffer.upload(source[..., 0]))
    rejects("wrong shape", lambda: buffer.upload(source[:, :, :, :3].copy()))
    rejects("float32", lambda: buffer.upload(source.astype(np.float32)))
    rejects("non-contiguous input", lambda: buffer.upload(source.swapaxes(0, 1)))
    rejects("empty input", lambda: buffer.upload(np.empty((0, 7, 1, 4))))
    invalid = source.copy()
    invalid.flat[0] = np.nan
    rejects("non-finite input", lambda: buffer.upload(invalid))
    rejects("three-dimensional shape", lambda: fourdstem_median.CudaAdaptiveMedianBuffer((7, 7, 4)))
    rejects("zero shape", lambda: fourdstem_median.CudaAdaptiveMedianBuffer((0, 7, 1, 4)))
    rejects("noninteger shape", lambda: fourdstem_median.CudaAdaptiveMedianBuffer((7, 7, 1, 4.0)))
    rejects("Boolean shape", lambda: fourdstem_median.CudaAdaptiveMedianBuffer((7, 7, True, 4)))

    zeros = np.zeros_like(source)
    buffer.upload(zeros)
    rejects("download after upload but before filter", buffer.download)
    buffer.filter()
    assert exact(buffer.download(), zeros)
    buffer.upload(source)
    rejects("stale output after replacement upload", buffer.download)
    buffer.filter()
    first = buffer.download()
    second = buffer.download()
    require_output(first, source, expected)
    require_output(second, source, expected)
    assert not np.shares_memory(first, second)
    first.flat[0] = -123456.0
    assert exact(buffer.download(), expected)
    buffer.filter()
    assert exact(buffer.download(), expected)


def validate_workload(label: str, source: np.ndarray, expected: np.ndarray) -> None:
    before = source.copy()
    buffer = fourdstem_median.CudaAdaptiveMedianBuffer(source.shape)
    assert buffer.shape == source.shape
    buffer.upload(source)
    buffer.filter()
    output = buffer.download()
    require_output(output, source, expected)
    buffer.filter()
    assert exact(buffer.download(), expected)
    assert exact(source, before)
    print(f"{label}: {source.size} Python output bits exact; input unchanged")


def centered_subset(source: np.ndarray) -> np.ndarray:
    sy, sx, dy, dx = source.shape
    ny, nx, ndy, ndx = min(sy, 64), min(sx, 35), min(dy, 8), min(dx, 8)
    return np.ascontiguousarray(
        source[(sy - ny) // 2 : (sy - ny) // 2 + ny,
               :nx,
               (dy - ndy) // 2 : (dy - ndy) // 2 + ndy,
               (dx - ndx) // 2 : (dx - ndx) // 2 + ndx]
    )


def report(label: str, milliseconds: list[float]) -> None:
    print(
        f"{label}: raw_ms={[round(value, 3) for value in milliseconds]} "
        f"min/median/max_ms={min(milliseconds):.3f}/"
        f"{median(milliseconds):.3f}/{max(milliseconds):.3f}"
    )


def benchmark(source: np.ndarray, expected: np.ndarray, runs: int) -> None:
    resident_owner = fourdstem_median.CudaAdaptiveMedianBuffer(source.shape)
    resident_owner.upload(source)
    resident_owner.filter()
    require_output(resident_owner.download(), source, expected)

    def one_shot_equivalent() -> tuple[float, np.ndarray]:
        start = perf_counter()
        owner = fourdstem_median.CudaAdaptiveMedianBuffer(source.shape)
        owner.upload(source)
        owner.filter()
        output = owner.download()
        del owner  # include device cleanup, as in a one-call owner lifetime
        elapsed_ms = 1000.0 * (perf_counter() - start)
        return elapsed_ms, output

    warm_ms, warm_output = one_shot_equivalent()
    require_output(warm_output, source, expected)
    del warm_ms, warm_output

    one_shot: list[float] = []
    totals = {count: [] for count in COUNTS}
    uploads = {count: [] for count in COUNTS}
    filters = {count: [] for count in COUNTS}
    downloads = {count: [] for count in COUNTS}
    for run in range(runs):
        if run % 2 == 0:
            elapsed_ms, output = one_shot_equivalent()
            require_output(output, source, expected)
            one_shot.append(elapsed_ms)
            del output

        for offset in range(len(COUNTS)):
            count = COUNTS[(run + offset) % len(COUNTS)]
            start = perf_counter()
            resident_owner.upload(source)
            after_upload = perf_counter()
            for _ in range(count):
                resident_owner.filter()
            after_filter = perf_counter()
            output = resident_owner.download()
            stop = perf_counter()
            require_output(output, source, expected)
            totals[count].append(1000.0 * (stop - start))
            uploads[count].append(1000.0 * (after_upload - start))
            filters[count].append(1000.0 * (after_filter - after_upload) / count)
            downloads[count].append(1000.0 * (stop - after_filter))
            del output

        if run % 2 != 0:
            elapsed_ms, output = one_shot_equivalent()
            require_output(output, source, expected)
            one_shot.append(elapsed_ms)
            del output

    print("Timing boundaries: one-shot-equivalent includes owner allocation/release; "
          "resident includes one upload, complete synchronized filters, and one "
          "new-array download, but excludes owner allocation/release.")
    report("one-shot-equivalent Python wall", one_shot)
    for count in COUNTS:
        report(f"resident-{count} total Python wall", totals[count])
        report(f"resident-{count} effective Python wall/filter",
               [value / count for value in totals[count]])
        report(f"resident-{count} upload", uploads[count])
        report(f"resident-{count} filter-only/filter", filters[count])
        report(f"resident-{count} download", downloads[count])
        print(f"resident-{count} effective speedup vs one-shot-equivalent: "
              f"{median(one_shot) / (median(totals[count]) / count):.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-input", type=Path)
    parser.add_argument("--subset-expected", type=Path)
    parser.add_argument("--canonical-expected", type=Path)
    parser.add_argument("--timing-runs", type=int, default=5)
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()
    if args.timing_runs < 3:
        parser.error("--timing-runs must be at least 3")

    public_input = load(PUBLIC / "reference_input.npy")
    public_expected = load(PUBLIC / "reference_output_python.npy")
    validate_contract(public_input, public_expected)
    validate_workload("public fixture", public_input, public_expected)

    local_input = LOCAL / "reference_input.npy"
    local_expected = LOCAL / "reference_output_python.npy"
    if local_input.exists() and local_expected.exists():
        validate_workload("ignored local fixture", load(local_input), load(local_expected))
    else:
        print("ignored local fixture unavailable; public fixture still passed")

    if args.canonical_input is None:
        print("canonical input not supplied; subset/canonical checks and timing skipped")
        return
    if args.subset_expected is None or args.canonical_expected is None:
        parser.error("canonical validation requires both --subset-expected and --canonical-expected")
    canonical = load(args.canonical_input)
    subset = centered_subset(canonical)
    validate_workload("representative subset", subset, load(args.subset_expected))
    expected = load(args.canonical_expected)
    validate_workload("canonical workload", canonical, expected)
    if not args.no_timing:
        benchmark(canonical, expected, args.timing_runs)


if __name__ == "__main__":
    main()
