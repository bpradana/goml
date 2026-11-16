<div align="center">

<img src="../../../docs/assets/rocm.svg" width="88" height="88" alt="goml ROCm backend logo">

# rocm

**Reserved for future ROCm acceleration. GPU execution is unimplemented.**

![Status](https://img.shields.io/badge/status-unimplemented-986073)
![Go](https://img.shields.io/badge/Go-1.27.3-00ADD8)

[Scope](#scope) · [Module](#module) · [Verification](#verification) · [Project](../../../README.md)

</div>

---

This directory reserves an independent Go module for future ROCm development.
It provides package documentation and a module identity only.
Compiling this package does not provide GPU support.

## Scope

| Area | Status |
| --- | --- |
| Device discovery and initialization | Unimplemented. |
| GPU allocations and transfers | Unimplemented. |
| Numerical kernels and model execution | Unimplemented. |
| Native dependencies | None. |
| Backend registration | None. |
| Release tags | None planned for this placeholder. |

There is no Open API, working device, native runtime, or kernel implementation.
The package is excluded from the active go.work file and ordinary CI task graph.
Tensor, nn, and the Metal backend do not depend on it.

## Module

The reserved module path is `github.com/bpradana/goml/contrib/backend/rocm`.
It has its own go.mod and requires Go 1.27.3.
Do not tag or advertise this source reservation as an available backend.

## Verification

To verify the source reservation without workspace resolution:

```sh
cd contrib/backend/rocm
GOWORK=off go test ./...
```

This checks ordinary Go package compilation only.
It does not discover hardware, allocate GPU memory, execute kernels, or establish numerical correctness.

## Related guides

- [Metal backend](../metal/README.md): the active GPU implementation target.
- [GPU architecture and ownership](../../../docs/gpu.md).
- [Tensor](../../../tensor/README.md) and [nn](../../../nn/README.md).

The logo identifies this goml package; it is not an upstream vendor logo.
Licensed under [MIT](../../../LICENSE).
