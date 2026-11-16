# YOLOv8n validation

[Documentation](README.md) · [YOLO module](../contrib/nn/yolo/README.md) · [Testing](testing.md)

This report records local validation for nn v0.3.0 and contrib/nn/yolo v0.1.0 on 2026-11-16.
It distinguishes numerical parity from accuracy, portability, and release publication.
These are historical results. The [CPU performance report](performance/cpu.md)
records the tensor v0.2.0, nn v0.4.0, and YOLO v0.2.0 optimization measurements
and repeated PyTorch reference validation.

## Environment and artifacts

| Item | Value |
| --- | --- |
| Host | Apple M1 Pro, macOS arm64. |
| Go | 1.27.3, default scalar build. |
| Reference | Ultralytics 8.3.0, PyTorch 2.5.1, CPU float32, BatchNorm fused. |
| Model | Official yolov8n.pt from the Ultralytics v8.3.0 assets release. |
| Image | Ultralytics bus.jpg, 810×1080 pixels. |
| Model SHA-256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Image SHA-256 | `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| Archive | Format version 1, 51 convolution networks, 80 classes. |
| Postprocessing | Confidence 0.25, class-aware NMS IoU 0.45, maximum 300 detections. |

Weights and images are explicitly downloaded into ignored data/yolo.
They are not committed as test fixtures.
The converter and complete Python dependency graph are locked in contrib/nn/yolo/convert.
The [upstream architecture](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/v8/yolov8.yaml) defines the model topology used by the converter and loader.

## Method

Go prepares the image with its own RGB resize and letterboxing implementation and writes input.f32.
Both runtimes evaluate those exact float32 input bytes.
The reference tool saves the three raw detection heads and PyTorch's class-aware NMS output.
The subsequent Go integration test compares every raw logit, retained class ID, confidence, and box coordinate.

This isolates execution and decoding from OpenCV image-resize differences.
It also checks OIHW kernel conversion, folded BatchNorm, SiLU, C2f residual paths, SPPF, upsampling, skip routing, and all detection heads together.

| Comparison | Acceptance tolerance |
| --- | --- |
| Each raw logit | `abs(got-want) <= 0.001 + 0.0001*abs(want)`. |
| Retained detection count and class IDs | Exact match in confidence order. |
| Class probability | Absolute difference at most 0.0001. |
| NMS box coordinate | Absolute difference at most 0.05 model-input pixels. |

## Results

| Input | Stride 8 logits / maximum error | Stride 16 logits / maximum error | Stride 32 logits / maximum error | Retained detections |
| --- | --- | --- | --- | --- |
| 320×320 | 230,400 / 0.000154912472 | 57,600 / 0.0000643730164 | 14,400 / 0.0000572204590 | One bus, three people. |
| 640×640 | 921,600 / 0.0000700950623 | 230,400 / 0.0000810623169 | 57,600 / 0.000112533569 | One bus, four people. |

Every comparison passed at both resolutions.
The complete Go image-detection CLI also ran successfully and returned boxes clipped to the original image dimensions.
The input resolution changes which objects pass the confidence threshold; the two rows are separate model inputs.

Observed forward-call durations were approximately 1.47–1.48 seconds at 320 and 5.99–6.34 seconds at 640 on this host.
These timings exclude model loading, image preparation, decoding, and NMS.
They are individual local observations rather than a statistical performance benchmark or a real-time guarantee.

## Reproduce

Follow the [download and conversion steps](../contrib/nn/yolo/README.md#quick-start), then run:

```sh
export GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml"
export GOML_YOLO_IMAGE="$PWD/data/yolo/bus.jpg"
export GOML_YOLO_REFERENCE="$PWD/data/yolo/reference-320"
export GOML_YOLO_SIZE=320

mise run test:yolo:reference
mise run reference:yolo -- data/yolo/yolov8n.pt data/yolo/reference-320 --size 320
mise run test:yolo:reference
```

The first run creates input.f32 when reference heads are absent.
It reports forward results but does not establish reference parity until the Python artifacts exist.
Repeat with size 640 and a separate reference-640 directory for the default input resolution.

## Supporting checks

| Check | Local result |
| --- | --- |
| Workspace formatting, builds, vet, lint, unit tests | Passed across all four modules. |
| Race detector | Passed across all four modules. |
| govulncheck | No reachable vulnerabilities reported; one module-level finding had no called vulnerable symbols. |
| Core standalone dependency checks | Passed for simdops, tensor, and nn with GOWORK=off. |
| YOLO standalone release gate | Pending nn v0.3.0 publication and generated nn dependency checksums; the current check reports a missing go.sum entry. |
| Experimental SIMD tests | Passed across all four modules. |
| Linux amd64 and arm64, CGo disabled | Cross-compilation passed with scalar and SIMD builds. |
| Shape, view, checkpoint, architecture, and YOLO archive fuzzing | All six bounded targets passed. |
| Converter formatting, lint, and unit tests | Passed with locked dependencies. |
| Release workflow syntax | Passed pinned actionlint validation. |

Primitive tests include weighted finite-difference checks for SiLU, pooling, and all three Upsample2D modes.
Interpolation tests include explicit half-pixel references, gradient accumulation, empty dimensions, overflow rejection, and nearest-mode preservation of special values.
Checkpoint tests round-trip SiLU, pooling, nearest, bilinear, and bicubic configurations.
YOLO unit tests cover archive boundaries, convolution validation, C2f and SPPF branch order, model skip connections, image preprocessing, stable DFL decoding, and class-aware NMS.

## Limits

This is one pretrained model and one image at two input sizes.
It does not establish dataset mAP, robustness, parity for other YOLO variants, or real-time throughput.
Native execution was checked on macOS arm64; Linux amd64/arm64 cross-compilation does not substitute for native runner execution.
GitHub-hosted workflows and release publication are separate maintainer operations.

The current YOLO v0.2.0 manifests require nn v0.4.0 and tensor v0.2.0.
Use `GOWORK=off mise run deps:tidy:yolo` and `mise run test:standalone:yolo`
to verify those versioned dependencies and their generated checksums.
The workspace provides local development resolution without a committed replace directive.
