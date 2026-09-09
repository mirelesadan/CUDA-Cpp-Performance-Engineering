# Local benchmark data

No experimental benchmark arrays are distributed with this public repository. The documented canonical measurements used a local scientifically prepared array whose redistribution status has not been established.

The native benchmark accepts a replacement NumPy `.npy` file from the command line. Its initial contract is:

- little-endian `float64` (`<f8`);
- C-contiguous storage;
- four nonempty dimensions ordered as `(scan_y, scan_x, detector_y, detector_x)`;
- finite values for the established exact median semantics.

Example after building:

```text
phase_a_baseline_benchmark <compatible-input.npy> --threads 1,2,4,8
```

The benchmark validates optimized serial and OpenMP outputs bit for bit before recording performance. Results from a replacement input are a new benchmark and should record its shape, hardware, compiler, flags, raw timings, and thread counts rather than being compared as though it were the private canonical dataset.
