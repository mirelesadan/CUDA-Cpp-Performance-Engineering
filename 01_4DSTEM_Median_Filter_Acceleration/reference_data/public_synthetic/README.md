# Public deterministic Phase A fixture

This fixture contains no experimental or research data. It is generated entirely from integer coordinates by `generate_fixture.py` and saved as a C-contiguous `float64` array with shape `(7, 6, 5, 4)`.

The expected output applies a centered `3 × 3` median over scan axes 0 and 1 only, independently at every detector coordinate, with half-sample symmetric (`reflect`) boundaries. For nine values, the result is the fifth ordered value with no averaging.

Regenerate with NumPy:

```text
python generate_fixture.py
```

The committed arrays are checked by the native validation executable. Their SHA-256 hashes are recorded below after generation.

- `reference_input.npy`: `d0e4e0e182dde03db35e77de9b83764f3b91158678bf89de7b24a85e4e456f4d`
- `reference_output_python.npy`: `a234dd793d64657c781e7dc8e738b6db0a408eaaa0ecbdcc1ebb5b7a3c1d0a21`
