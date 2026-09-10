"""Generate the public deterministic Phase B adaptive-median fixture."""

from pathlib import Path
import hashlib

import numpy as np


def adaptive_plane(plane: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    padded = np.pad(plane, 3, mode="constant", constant_values=np.min(plane))
    output = np.empty_like(plane)
    counts = {
        "stage_a_pass_stage_b_retain": 0,
        "stage_a_pass_stage_b_replace": 0,
        "expanded_to_5": 0,
        "expanded_to_7": 0,
        "maximum_window_fallback": 0,
    }

    for scan_y in range(plane.shape[0]):
        for scan_x in range(plane.shape[1]):
            center_y = scan_y + 3
            center_x = scan_x + 3
            center = padded[center_y, center_x]
            for window_size in (3, 5, 7):
                radius = window_size // 2
                window = padded[
                    center_y - radius : center_y + radius + 1,
                    center_x - radius : center_x + radius + 1,
                ]
                local_minimum = np.min(window)
                local_median = np.median(window)
                local_maximum = np.max(window)
                if local_minimum < local_median < local_maximum:
                    if local_minimum < center < local_maximum:
                        output[scan_y, scan_x] = center
                        counts["stage_a_pass_stage_b_retain"] += 1
                    else:
                        output[scan_y, scan_x] = local_median
                        counts["stage_a_pass_stage_b_replace"] += 1
                    break
                if window_size == 3:
                    counts["expanded_to_5"] += 1
                elif window_size == 5:
                    counts["expanded_to_7"] += 1
                else:
                    output[scan_y, scan_x] = center
                    counts["maximum_window_fallback"] += 1
    return output, counts


def build_fixture() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    base = np.arange(1, 50, dtype=np.float64).reshape(7, 7)
    data = np.empty((7, 7, 1, 4), dtype=np.float64)
    data[:, :, 0, 0] = base
    data[:, :, 0, 1] = base
    data[3, 3, 0, 1] = 999.0
    data[:, :, 0, 2] = 10.0
    data[2:5, 2:5, 0, 2] = 0.0
    data[3, 3, 0, 2] = 100.0
    data[:, :, 0, 3] = base
    data[0, 0, 0, 3] = 999.0

    output = np.empty_like(data)
    totals: dict[str, int] = {}
    for detector_x in range(data.shape[3]):
        filtered, counts = adaptive_plane(data[:, :, 0, detector_x])
        output[:, :, 0, detector_x] = filtered
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value

    # Named cases exercise the contract's principal control-flow decisions.
    assert output[3, 3, 0, 0] == 25.0  # Stage A/B pass: retain center.
    assert output[3, 3, 0, 1] == 26.0  # Stage A pass: replace impulse.
    assert output[3, 3, 0, 2] == 10.0  # Equality fails at 3x3; 5x5 succeeds.
    assert output[0, 0, 0, 3] == 999.0  # Border expands through 7x7, then falls back.
    assert data.dtype == output.dtype == np.float64
    assert data.flags.c_contiguous and output.flags.c_contiguous
    assert not np.shares_memory(data, output)
    return data, output, totals


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent
    fixture_input, fixture_output, branch_counts = build_fixture()
    input_path = destination / "reference_input.npy"
    output_path = destination / "reference_output_python.npy"
    np.save(input_path, fixture_input, allow_pickle=False)
    np.save(output_path, fixture_output, allow_pickle=False)
    print(f"shape={fixture_input.shape}, dtype={fixture_input.dtype}, values={fixture_input.size}")
    print(f"input_sha256={sha256(input_path)}")
    print(f"output_sha256={sha256(output_path)}")
    print(f"branch_counts={branch_counts}")
