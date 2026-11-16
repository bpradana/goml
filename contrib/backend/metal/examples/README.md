# Metal examples and integration tests

[Metal backend](../README.md) · [GPU contracts](../../../../docs/gpu.md)

This separate development module owns model examples and end-to-end GPU tests.
It depends on Metal, tensor, nn, and YOLO. The Metal backend itself depends only
on tensor and does not import any neural network packages.

The module is used through go.work and the opt-in Metal mise tasks. It has no
independent release target and is not required by CPU CI or backend consumers.
Native model tests require macOS, CGo, and a supported Metal device.

## Programs

| Program | Purpose |
| --- | --- |
| [matmul](matmul/README.md) | Rectangular MPS multiplication with explicit transfers. |
| [training](training/README.md) | GPU linear regression and optional portable checkpoint. |
| [yolo](yolo/README.md) | Pretrained image detection with resident model weights. |

Run test:gpu:parity for model, gradient, optimizer, training, and checkpoint checks.
Set GOML_YOLO_MODEL to enable pretrained YOLO validation with local bus.jpg and reference-320/reference-640 input files.
Run bench:gpu:models for complete optimizer steps and staged YOLO measurements.
See the [measurement report](../../../../docs/performance/metal.md) for reproduction details.
