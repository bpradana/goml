# YOLO weight conversion

[YOLO module](../README.md) · [Checkpoint format](../../../../nn/checkpoint/README.md)

These offline tools convert a trusted YOLOv8n detection checkpoint and produce numerical reference outputs.
They are not part of the Go inference runtime.
Python 3.12, PyTorch 2.5.1, Ultralytics 8.3.0, NumPy, and the complete dependency graph are pinned by pyproject.toml and uv.lock.
Project commands run through the root mise configuration.

## Convert weights

From the repository root, with a local trusted checkpoint:

```sh
mise run convert:yolo -- data/yolo/yolov8n.pt data/yolo/yolov8n.goml
```

The converter validates the YOLOv8n stage layout, folds BatchNorm into convolution parameters, checks supported operators, transposes float32 kernels, and writes a goml archive.
It rejects unsupported activations, grouped or dilated kernels, nonfinite parameters, and incompatible model families.
The source file must exist; the converter does not automatically fetch a checkpoint.
Destination replacement uses a temporary file in the same directory and an atomic rename.
Repeated conversion of the same weights produces the same archive bytes.

## Compare with PyTorch

The optional Go integration test writes input.f32 to a reference directory.
Generate reference raw heads and class-aware NMS results from that exact input:

```sh
mise run reference:yolo -- data/yolo/yolov8n.pt data/yolo/reference-320 --size 320
```

The directory must already contain input.f32 with exactly `3*size*size` little-endian float32 values.
The command writes head-0.f32, head-1.f32, head-2.f32, and detections.json.
Run the [Go reference check](../README.md#numerical-validation) again to compare results.
Using identical input isolates model arithmetic from differences between Go and OpenCV image resizing.

## Check the tools

```sh
mise run fmt:converter
mise run test:converter
```

Checks include Ruff formatting and linting, checkpoint length and checksum validation, an independent OIHW-to-matrix reference, and unsupported-input tests.
No model download is needed for these unit tests.

## Licensing and trust

Only load .pt files from trusted sources because PyTorch checkpoints can contain executable serialized objects.
The converter imports Ultralytics; its software and pretrained weights retain their upstream licensing.
See the [Ultralytics license page](https://www.ultralytics.com/license) before redistributing upstream or converted model artifacts.
