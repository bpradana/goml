# Minimal backend example

This package implements an independent Go-memory device and MatMul capability using
only `tensor/backend`. It demonstrates ownership, exact transfers, typed errors,
strided inputs, explicit device selection, and reusable conformance tests.

It intentionally has no arithmetic, Fill, or convolution capability. Missing
operations return `backend.ErrUnsupported` through Tensor; execution never falls
back to another device.

Start with the [contributor walkthrough](../../../docs/extending-backends.md), then
read `device.go`, `buffer.go`, and `matmul.go`. Each file has matching tests.

```sh
mise run test:backend
```

The example is compiled and tested in the Tensor module. It is not a separate
release or a replacement for the optimized CPU backend.
