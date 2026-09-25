"""Generate approved public K-means workloads and time the unchanged serial fit.

This is a diagnostic driver, not a replacement for the frozen reference or
the serial API. Generated arrays and raw interchange files live in OS temp.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics
import subprocess
import tempfile

from benchmark_data import (ITERATIVE_PROFILE_SPEC, WORKLOADS,
                            generate_iterative_workload, generate_workload)
from reference import compare_results, fit_kmeans
from test_cpp_serial import run_native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workload", choices=(*WORKLOADS, ITERATIVE_PROFILE_SPEC.name))
    parser.add_argument("--sample-repeats", type=int, default=0,
                        help="use the Windows sampling executable for this many fits")
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    generated = (generate_iterative_workload()
                 if args.workload == ITERATIVE_PROFILE_SPEC.name else
                 generate_workload(WORKLOADS[args.workload]))
    spec = generated.spec
    if args.sample_repeats:
        if not 1 <= args.sample_repeats <= 10_000:
            parser.error("--sample-repeats must be between 1 and 10000")
        with tempfile.TemporaryDirectory(prefix="project2_sampling_") as temporary:
            input_path = Path(temporary) / f"{spec.name}.f32"
            generated.samples.tofile(input_path)
            completed = subprocess.run(
                [str(executable), str(input_path), str(spec.n), str(spec.d),
                 str(spec.k), str(args.sample_repeats)],
                text=True, capture_output=True, check=True)
        print(f"{spec.name}: N={spec.n}, D={spec.d}, K={spec.k}, "
              f"PCG64 seed={spec.seed}")
        print(completed.stdout, end="")
        return
    with tempfile.TemporaryDirectory(prefix="project2_profile_") as temporary:
        result, metadata = run_native(executable, generated.samples, spec.k,
                                      Path(temporary), spec.name, timed_runs=7)
    times = metadata["timings_ms"]
    if len(times) != 7 or metadata["warmup"] != 1:
        raise AssertionError("expected one warm-up and seven complete-fit timings")
    print(f"{spec.name}: N={spec.n}, D={spec.d}, K={spec.k}, "
          f"PCG64 seed={spec.seed}")
    print("whole-fit raw ms: " + ", ".join(f"{value:.6f}" for value in times))
    print(f"min/median/max ms: {min(times):.6f} / "
          f"{statistics.median(times):.6f} / {max(times):.6f}")
    print(f"update_count={result.update_count}, converged={result.converged}, "
          f"assignment_passes={1 + result.update_count}")
    if "phase_timings" in metadata:
        phases = metadata["phase_timings"]
        if len(phases) != 7:
            raise AssertionError("expected seven phase-timing records")
        if any(phase["assignment_passes"] != 1 + result.update_count or
               phase["centroid_updates"] != result.update_count for phase in phases):
            raise AssertionError("phase pass counts disagree with fit result")
        phase_names = ("validation_ms", "initialization_ms",
                       "initial_assignment_ms", "centroid_update_ms",
                       "reassignment_ms", "convergence_check_ms")
        for name in phase_names:
            print(f"phase {name} median ms: "
                  f"{statistics.median(phase[name] for phase in phases):.6f}")
        assignment = [phase["initial_assignment_ms"] + phase["reassignment_ms"]
                      for phase in phases]
        residual = [times[index] - sum(phases[index][name] for name in phase_names)
                    for index in range(7)]
        print(f"phase all_assignment_ms median: {statistics.median(assignment):.6f}; "
              f"unattributed residual median: {statistics.median(residual):.6f}")
    if args.workload == ITERATIVE_PROFILE_SPEC.name:
        oracle = fit_kmeans(generated.samples, spec.k)
        comparison = compare_results(generated.samples, oracle, result)
        digest = hashlib.sha256(generated.samples.tobytes()).hexdigest()
        print(f"iterative input SHA-256={digest}; reference update_count="
              f"{oracle.update_count}, converged={oracle.converged}; "
              f"max centroid absolute error={comparison.max_centroid_absolute_error:.9g}")


if __name__ == "__main__":
    main()
