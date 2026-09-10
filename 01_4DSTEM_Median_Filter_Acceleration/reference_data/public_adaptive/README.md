# Public Phase B adaptive-median fixture

This deterministic `(7, 7, 1, 4)` finite-`float64`, C-order fixture contains no research data. `generate_fixture.py` implements the established `s=3`, `sMax=7` Python contract literally and creates the committed input/reference pair.

The four detector planes provide named checks for Stage A plus Stage B retaining the center, Stage B replacing an impulse with the median, equality forcing expansion from `3 x 3` to a successful `5 x 5`, and global-minimum-padded boundary processing that expands through `7 x 7` before the maximum-window fallback. The complete fixture also exercises both Stage B outcomes and all permitted window sizes across many scan positions.

Regenerate from this directory with:

```powershell
python generate_fixture.py
```

The native `phase_b_adaptive_median_validation` target uses these arrays by default and compares all 196 values bit for bit.

- `reference_input.npy`: `5acc24be02a379b0eb70e8fab71577080870a1ee32f3165c7b7e4b031946a3d4`
- `reference_output_python.npy`: `f34de33b8b54a70a4f13daf9ee9f43e877972cd3828cd2bd795552a200aaa03b`
