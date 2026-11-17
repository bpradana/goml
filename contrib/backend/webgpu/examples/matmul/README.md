# WebGPU matrix multiplication

[Examples](../README.md) · [Backend](../../README.md)

The program uploads `[2,3]` and `[3,2]` Float32 matrices, executes the tiled WGSL
kernel, and explicitly reads the `[2,2]` result. It closes every tensor handle
and the execution context.

```sh
mise run example:gpu:webgpu:matmul
```

On the validated M1 Pro adapter:

```text
Apple M1 Pro (metal): shape=[2 2] values=[4 5 10 11]
```

This is native WebGPU, requiring CGo and supported hardware. It does not run a
browser, download assets, or fall back to CPU computation.

Licensed under [MIT](../../../../../LICENSE).
