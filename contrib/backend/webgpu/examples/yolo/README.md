# YOLOv8n on WebGPU

[Examples](../README.md) · [Backend](../../README.md) · [YOLO](../../../../nn/yolo/README.md)

This program loads an explicitly converted local YOLOv8n archive, uploads its
weights once, and detects objects in a local JPEG or PNG. Intermediate
activations and raw heads stay on WebGPU. Final output readback, decoding,
and non-maximum suppression are explicit CPU work.

```sh
mise run example:gpu:webgpu:yolo -- \
  -model data/yolo/yolov8n.goml \
  -image data/yolo/bus.jpg \
  -size 320
```

Detections are written as JSON to standard output. The adapter, complete detection
latency, and tracked peak tensor bytes are written to standard error. No weights
or images are downloaded automatically.

Use the existing [conversion instructions](../../../../nn/yolo/README.md#weight-conversion)
for model setup. The optional pretrained test checks both 320 and 640 inputs
against CPU raw heads and detections using local reference inputs:

```sh
GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml" mise run test:gpu:webgpu:yolo
```

Direct WGSL convolution is a baseline implementation. Compare complete workloads
and transfer costs before choosing it over CPU or the specialized Metal backend.

Licensed under [MIT](../../../../../LICENSE).
