from hashlib import sha256
from pathlib import Path

import numpy as np


FIXTURE_SHAPE = (7, 6, 5, 4)


def fixed_median_3x3_reference(array: np.ndarray) -> np.ndarray:
    padded = np.pad(array, ((1, 1), (1, 1), (0, 0), (0, 0)), mode="symmetric")
    output = np.empty_like(array)

    for scan_y in range(array.shape[0]):
        for scan_x in range(array.shape[1]):
            neighborhood = padded[scan_y : scan_y + 3, scan_x : scan_x + 3]
            output[scan_y, scan_x] = np.partition(
                neighborhood.reshape(9, array.shape[2], array.shape[3]),
                4,
                axis=0,
            )[4]

    return output


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    coordinates = np.indices(FIXTURE_SHAPE, dtype=np.int64)
    encoded = (
        37 * coordinates[0]
        + 19 * coordinates[1]
        + 11 * coordinates[2]
        + 5 * coordinates[3]
    ) % 53
    input_array = np.ascontiguousarray(encoded.astype(np.float64) * 0.25 - 6.5)
    output_array = np.ascontiguousarray(fixed_median_3x3_reference(input_array))

    fixture_directory = Path(__file__).resolve().parent
    input_path = fixture_directory / "reference_input.npy"
    output_path = fixture_directory / "reference_output_python.npy"
    np.save(input_path, input_array, allow_pickle=False)
    np.save(output_path, output_array, allow_pickle=False)

    print(f"shape={FIXTURE_SHAPE}, elements={input_array.size}")
    print(f"{input_path.name}: {file_sha256(input_path)}")
    print(f"{output_path.name}: {file_sha256(output_path)}")


if __name__ == "__main__":
    main()
