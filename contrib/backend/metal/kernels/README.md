# Metal compute shaders

[Metal backend](../README.md) · [Backend contract](../../../../docs/backend-contract.md)

These embedded Metal Shading Language sources implement device-resident
operations that use custom compute pipelines. Go owns shape and device checks;
the Objective-C bridge owns compilation, cached pipelines, command encoding,
and native resource retention.

| Source | Responsibility |
| --- | --- |
| `common.metal` | Metal standard library, shared Params layout, strided indexing, and sigmoid. |
| `layout.metal` | Fill, strided copy/scatter, and finite-value checks. |
| `arithmetic.metal` | Binary operations, scalar operations, activations, and derivatives. |
| `reduction.metal` | Sum, Mean, and Max over selected axes. |
| `spatial.metal` | Max pooling and nearest/bilinear/bicubic interpolation, including derivatives. |

`native_darwin.go` embeds the files and concatenates them in the table's order.
Open compiles the assembled library once per device and caches its pipelines.
No source files are read from disk at runtime. Operation entry points and
numerical algorithms are unchanged by the file split.

Params is a 168-byte structure shared with `kernelParams` and `GMParams` in
`metal.h`. Its scalar array starts at byte 152. The native bridge test checks
field offsets and conversion values. Shapes and strides have at most eight
dimensions and use 32-bit indices. Dispatch metadata is copied during encoding;
shaders do not retain borrowed Go slices.

MatMul uses MPSMatrixMultiplication. Convolution and its derivatives use
MPSGraph. Their native implementations remain in `metal_darwin.m`, beside
`metal.h` so CGo can compile and link the bridge in the Metal Go package.
There are no placeholder MatMul or convolution shader files.

```sh
mise run fmt:native:metal
mise run fmt:check:metal
mise run test:gpu:metal
mise run test:gpu:parity
```

Native formatting applies the backend's clang-format settings to every shader.
macOS, CGo, Xcode command-line tools, and supported Metal hardware are required
for execution. CPU-only builds keep the unsupported native stub.

Licensed under [MIT](../../../../LICENSE).
