"""Benchmark the public 4Denoise fixed-median path on the canonical input."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "benchmark_data" / "median_filter_input.npy"
DEFAULT_FOURDENOISE_ROOT = PROJECT_ROOT.parent.parent / "4denoise_git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--fourdenoise-root",
        type=Path,
        default=DEFAULT_FOURDENOISE_ROOT,
        help="Directory containing the checked-out fourdenoise.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    fourdenoise_root = args.fourdenoise_root.resolve()
    expected_module = (fourdenoise_root / "fourdenoise.py").resolve()

    sys.path.insert(0, str(fourdenoise_root))
    import numpy as np
    import scipy
    import fourdenoise as fd

    resolved_module = Path(fd.__file__).resolve()
    if resolved_module != expected_module:
        raise RuntimeError(
            f"Imported the wrong 4Denoise module: {resolved_module}; expected {expected_module}"
        )

    load_start = time.perf_counter_ns()
    data = np.load(input_path, allow_pickle=False)
    load_end = time.perf_counter_ns()
    load_ms = (load_end - load_start) / 1.0e6

    if data.ndim != 4 or data.dtype != np.dtype("<f8") or not data.flags.c_contiguous:
        raise RuntimeError(
            "Canonical input must be a four-dimensional, C-contiguous float64 array."
        )

    dataset = fd.HyperData(data)
    coordinates = (
        (0, 0, 0, 0),
        tuple(size // 2 for size in data.shape),
        tuple(size - 1 for size in data.shape),
    )

    def denoise() -> np.ndarray:
        return dataset.denoise(
            method="median",
            domain="real",
            window_size=3,
            mode="reflect",
            cval=0.0,
            origin=0,
            return_array=True,
        )

    warmup = denoise()
    warmup_samples = tuple(float(warmup[coordinate]) for coordinate in coordinates)
    del warmup
    gc.collect()

    run_ms: list[float] = []
    run_samples: list[tuple[float, ...]] = []
    for _ in range(3):
        filter_start = time.perf_counter_ns()
        result = denoise()
        filter_end = time.perf_counter_ns()
        run_ms.append((filter_end - filter_start) / 1.0e6)
        run_samples.append(tuple(float(result[coordinate]) for coordinate in coordinates))
        del result
        gc.collect()

    reference_bits = np.asarray(warmup_samples, dtype=np.float64).view(np.uint64)
    samples_consistent = all(
        np.array_equal(np.asarray(samples, dtype=np.float64).view(np.uint64), reference_bits)
        for samples in run_samples
    )
    sorted_ms = sorted(run_ms)
    minimum_ms, median_ms, maximum_ms = sorted_ms
    median_seconds = median_ms / 1000.0
    elements_per_second = data.size / median_seconds
    output_gb_per_second = data.nbytes / median_seconds / 1.0e9

    print("Project 1 Phase A 4Denoise baseline benchmark")
    print(f"Input path: {input_path}")
    print(f"4Denoise module: {resolved_module}")
    print(f"Python version: {sys.version.replace(chr(10), ' ')}")
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {scipy.__version__}")
    print(f"Dtype: {data.dtype}")
    print(f"Shape: {data.shape}")
    print("Storage order: C-contiguous")
    print(f"Strides (bytes): {data.strides}")
    print(f"Input elements: {data.size}")
    print(f"Logical bytes: {data.nbytes}")
    print(f"NPY file bytes: {input_path.stat().st_size}")
    print(f"NPY load time_ms: {load_ms:.3f}")
    print("Warm-up runs: 1")
    for index, duration_ms in enumerate(run_ms, start=1):
        print(f"Timed run {index}_ms: {duration_ms:.3f}")
    print(f"Minimum_ms: {minimum_ms:.3f}")
    print(f"Median_ms: {median_ms:.3f}")
    print(f"Maximum_ms: {maximum_ms:.3f}")
    print(f"Output rate_Melements_per_s: {elements_per_second / 1.0e6:.3f}")
    print(f"Effective output data rate_GB_per_s: {output_gb_per_second:.3f}")
    print(
        "Timing boundary: complete HyperData.denoise call; wrapper dispatch, filtering, "
        "output allocation, and return_array=True handling included"
    )
    for coordinate, value in zip(coordinates, warmup_samples):
        print(f"Sanity {coordinate}: {value:.17g}")
    print(
        "Sanity samples bitwise-consistent across warm-up and timed runs: "
        f"{'yes' if samples_consistent else 'no'}"
    )

    return 0 if samples_consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
