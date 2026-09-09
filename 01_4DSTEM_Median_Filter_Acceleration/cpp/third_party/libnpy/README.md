# libnpy dependency

- Upstream: <https://github.com/llohse/libnpy>
- Release: `v1.0.1`
- Commit: `890ea4fcda302a580e633c624c6a63e2a5d422f6`
- License: MIT; the upstream license is preserved in `LICENSE`.

Vendored upstream files:

- `include/npy.hpp` — unmodified single-header library; SHA-256 `7cd423aa9e6a4d9817e541f2da269d7dcdc965aeaf39a99ceb5b01192dd3100a`
- `LICENSE` — unmodified upstream license; SHA-256 `4e1557fab51c823b98223c1401ceaa185cd26c33d2e0b6c0ca65a309077b6837`

This dependency was selected because the project currently needs only numeric `.npy` loading. libnpy is header-only, has no runtime or link dependency, exposes dtype/shape/Fortran-order metadata, and supports the required little-endian `float64` contract. Support for `.npz`, object arrays, and general tensor abstractions is deliberately unnecessary here.
