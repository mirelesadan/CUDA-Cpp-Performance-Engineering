"""Paired whole-fit benchmark for the isolated serial addressing experiment.

The native executable reads the public synthetic input once, warms both paths,
then alternates their timed fit calls. Raw-file I/O and comparisons are outside
the std::chrono timing boundaries. Generated data live only in OS temp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np

from benchmark_data import (ITERATIVE_PROFILE_SPEC, WORKLOADS,
                            generate_iterative_workload, generate_workload)
from reference import KMeansResult, compare_results, fit_kmeans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workload", choices=("profiling", ITERATIVE_PROFILE_SPEC.name))
    parser.add_argument("--blocks", type=int,
                        help="override the default three primary or one iterative block")
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    generated = (generate_iterative_workload()
                 if args.workload == ITERATIVE_PROFILE_SPEC.name else
                 generate_workload(WORKLOADS["profiling"]))
    spec = generated.spec
    oracle = fit_kmeans(generated.samples, spec.k)
    expected_updates = 22 if args.workload == ITERATIVE_PROFILE_SPEC.name else 1
    if oracle.update_count != expected_updates or not oracle.converged:
        raise AssertionError("public workload iteration behavior changed")
    pairs, default_blocks = ((5, 1) if args.workload == ITERATIVE_PROFILE_SPEC.name
                             else (7, 3))
    blocks = default_blocks if args.blocks is None else args.blocks
    if not 1 <= blocks <= 10:
        parser.error("blocks must be 1..10")

    with tempfile.TemporaryDirectory(prefix="project2_address_pair_") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.f32"
        labels_path = directory / "labels.i32"
        centroids_path = directory / "centroids.f32"
        generated.samples.tofile(input_path)
        before = hashlib.sha256(input_path.read_bytes()).digest()
        command = [str(executable), str(input_path), str(spec.n), str(spec.d),
                   str(spec.k), str(labels_path), str(centroids_path),
                   "--paired-runs", str(pairs), "--paired-blocks", str(blocks)]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        metadata = json.loads(completed.stdout)
        if hashlib.sha256(input_path.read_bytes()).digest() != before:
            raise AssertionError("native CLI changed its input file")
        labels = np.empty(spec.n, dtype=np.int32)
        labels[:] = np.fromfile(labels_path, dtype=np.int32)
        centroids = np.empty((spec.k, spec.d), dtype=np.float32, order="C")
        centroids[:] = np.fromfile(centroids_path, dtype=np.float32).reshape(spec.k, spec.d)
    candidate = KMeansResult(labels, centroids, metadata["update_count"],
                             metadata["converged"])
    comparison = compare_results(generated.samples, oracle, candidate)
    if candidate.update_count != expected_updates or not candidate.converged:
        raise AssertionError("candidate update count or convergence changed")
    if metadata["warmup_each"] != 1 or metadata["paired_runs_per_block"] != pairs:
        raise AssertionError("unexpected paired benchmark warm-up/repetition protocol")
    starts = metadata["first_variant_per_block"]
    if starts != ["baseline" if index % 2 == 0 else "addressed"
                  for index in range(blocks)]:
        raise AssertionError("paired benchmark block order changed")
    baseline = metadata["baseline_times_ms"]
    addressed = metadata["addressed_times_ms"]
    if len(baseline) != blocks or len(addressed) != blocks or any(
            len(values) != pairs for group in (baseline, addressed) for values in group):
        raise AssertionError("paired benchmark omitted timed calls")
    print(f"{spec.name}: (N,D,K)=({spec.n},{spec.d},{spec.k}), seed={spec.seed}, "
          f"updates={candidate.update_count}, converged={candidate.converged}")
    print(f"oracle agreement: exact labels/count/flag, maximum centroid absolute "
          f"error={comparison.max_centroid_absolute_error:.9g}; "
          "native baseline/addressed outputs matched bitwise on every pair")
    for block in range(blocks):
        print(f"block {block + 1}, first={starts[block]}; alternating AB/BA:")
        for name, values in (("baseline", baseline[block]),
                             ("addressed", addressed[block])):
            print(f"  {name} raw ms: " + ", ".join(f"{value:.6f}" for value in values))
            print(f"  {name} min/median/max ms: {min(values):.6f} / "
                  f"{statistics.median(values):.6f} / {max(values):.6f}")
        print(f"  block speedup: "
              f"{statistics.median(baseline[block]) / statistics.median(addressed[block]):.6f}x")
    baseline_all = [value for block in baseline for value in block]
    addressed_all = [value for block in addressed for value in block]
    baseline_median = statistics.median(baseline_all)
    addressed_median = statistics.median(addressed_all)
    speedup = baseline_median / addressed_median
    print(f"all-block baseline min/median/max ms: {min(baseline_all):.6f} / "
          f"{baseline_median:.6f} / {max(baseline_all):.6f}")
    print(f"all-block addressed min/median/max ms: {min(addressed_all):.6f} / "
          f"{addressed_median:.6f} / {max(addressed_all):.6f}")
    print(f"pooled speedup={speedup:.6f}x; runtime reduction="
          f"{100.0 * (1.0 - addressed_median / baseline_median):.4f}%; "
          f"addressed throughput={spec.n * 1000.0 / addressed_median / 1e6:.6f} "
          f"Msamples/s, {spec.n * spec.k * (1 + candidate.update_count) * 1000.0 / addressed_median / 1e6:.6f} "
          "M sample-cluster distances/s")


if __name__ == "__main__":
    main()
