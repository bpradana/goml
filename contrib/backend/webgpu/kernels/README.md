# WebGPU compute kernels

[Backend](../README.md) · [Contract](../../../../docs/backend-contract.md)

WGSL files are embedded into the Go module and compiled once per device. The
common declarations are prepended to each operation family before pipeline
creation. No shader source is loaded from the caller's filesystem at runtime.

| File | Responsibility |
| --- | --- |
| common.wgsl | Bindings, parameter layout, logical indexing, and Float32 classification. |
| arithmetic.wgsl | Typed unary, binary, and scalar numerical operations. |
| matmul.wgsl | Bounds-checked 16×16 tiled multiplication and strided matrix reads. |
| reduction.wgsl | Per-output selected-axis reductions. |
| layout.wgsl | Logical copy/scatter, fill, and finite-value status. |
| convolution.wgsl | Direct NCHW forward correlation and gathered input/weight derivatives. |
| spatial.wgsl | Pooling and nearest/bilinear/bicubic interpolation with gradients. |

The parameter storage structure is 304 bytes with four-byte elements. Its scalar
vector starts at byte 288. compute_test.go verifies the Go/WGSL layout. Shape,
stride, offset, and operation metadata are copied before asynchronous encoding.

Bindings zero through three are destination and up to three source buffers;
binding four contains parameters. Unused source bindings use an internal read-only
dummy allocation; mandatory operands still require valid buffers. Destinations
never alias sources. Most kernels use 64-thread
workgroups with a flattened two-dimensional dispatch grid; MatMul uses 16×16.

Add operation identifiers to the shared typed contract only for a real consumer.
Update the matching Go source/test pair and run hardware conformance after shader
changes. WGSL compilation errors must propagate from Open, not become successful
zero-filled results. Numerical code stays on the GPU; host transfers remain
explicit Tensor or Device calls.
