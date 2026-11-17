# WebGPU examples and integration tests

[WebGPU backend](../README.md) · [Backend contract](../../../../docs/backend-contract.md)

This separate development module owns model examples and end-to-end WebGPU tests.
It uses WebGPU, Tensor, NeuralNet, and YOLO without adding neural-network
dependencies to the backend's runtime module. CPU CI does not require it.

The committed go.work resolves the new backend locally. Until WebGPU's first tag
is published, this examples module is workspace-only and intentionally does not
declare an unpublished WebGPU version. After publication, add the real WebGPU
requirement and generate its checksums before standalone example validation.
Do not add a committed relative replace directive or invent checksum entries.
The examples module itself has no independent release tag.

## Programs

| Program | Purpose |
| --- | --- |
| [matmul](matmul/README.md) | Rectangular WGSL multiplication and explicit host transfers. |
| [training](training/README.md) | Resident linear regression and an optional portable checkpoint. |
| [yolo](yolo/README.md) | Pretrained detection using explicit local model weights and images. |

```sh
mise run example:gpu:webgpu:matmul
mise run example:gpu:webgpu:training
mise run test:gpu:webgpu:parity
mise run test:gpu:webgpu:parity:race
```

Model tests compare forward outputs and gradients with CPU references for Dense,
Conv2D, LayerNorm, attention, transformers, RNN, and LSTM. Optimizer tests compare
SGD and Adam updates and require stable tracked GPU bytes across repeated steps.
The training test saves GPU parameters and restores the checkpoint on CPU.

## Pretrained validation

Ordinary tests do not download models or datasets. Enable the optional YOLO test
with a converted archive and its neighboring bus.jpg and reference-320/reference-640
input fixtures from the [YOLO validation workflow](../../../../docs/yolo-validation.md):

```sh
GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml" mise run test:gpu:webgpu:yolo
```

The test checks 320 and 640 raw heads, detection counts, class IDs, confidence,
boxes, and allocation cleanup. Final decoding and NMS run on the CPU after the
explicit GPU output download. See the [measurement report](../../../../docs/performance/webgpu.md)
for numerical tolerances, timings, and platform limits.

Licensed under [MIT](../../../../LICENSE).
