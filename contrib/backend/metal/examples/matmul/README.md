# Metal matrix multiplication

[Examples](../README.md) · [Metal](../../README.md)

This program uploads `[2,3]` and `[3,2]` matrices, computes an MPS matrix product,
and explicitly downloads `[2,2]` values. A scope owns inputs and output.

```sh
mise run example:gpu:matmul
```

Expected values are `[4 5 10 11]`. macOS, CGo, and Metal hardware are required.
