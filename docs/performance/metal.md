# Metal validation and performance

[Performance](../performance.md) · [GPU contracts](../gpu.md) · [Metal backend](../../contrib/backend/metal/README.md)

Measured locally on 2026-11-16 using Apple M1 Pro, macOS 27.0, darwin/arm64,
Go 1.27.3, and the default eight Go processors. CPU comparisons use the current
optimized scalar implementation. SIMD was not enabled for these measurements.
The native implementation uses MPS MatMul and MPSGraph convolution.

## Method and reproducibility

Matrix measurements use three repetitions with `-benchmem -benchtime=200ms`.
GPU cases warm the MPS operation before timing. Each resident iteration includes
output allocation, encoding, explicit synchronization, and output release.
Resident inputs are uploaded before timing. Transfer-inclusive cases also upload
both operands and download the output. This measures wall-clock latency, not isolated shader time.

Model measurements use three repetitions of three iterations per stage.
YOLO forward reads the exact local bus-image input.f32 reference files.
Complete detection includes image preparation, input upload when applicable,
forward execution, final downloads, decoding, NMS, and coordinate restoration.
Both CPU and GPU use the same model, image, options, and Float32 input conventions.
GPU models are warmed before forward and detection timing.

```sh
mise run bench:gpu:info
mise run bench:gpu:metal
GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml" mise run bench:gpu:models
```

The benchmark tasks print their measurements for local inspection.
Values below are medians of three samples. Raw output stays outside the repository.
These short local runs establish a baseline. They do not establish performance on
other Macs, arbitrary images, or concurrent application workloads.

## Matrix multiplication

All values are milliseconds. Dimensions are M×K×N.

| Dimensions | CPU | Metal resident | Metal including transfers |
| --- | ---: | ---: | ---: |
| 8×8×8 | 0.000354 | 0.320 | 0.344 |
| 64×64×64 | 0.0775 | 0.325 | 0.372 |
| 256×256×256 | 4.990 | 0.419 | 0.731 |
| 512×512×512 | 43.454 | 0.896 | 1.659 |
| 1024×1024×1024 | 342.441 | 2.502 | 5.958 |
| 127×65×193 | 0.474 | 0.533 | 0.559 |
| 512×64×96 | 0.881 | 0.438 | 0.587 |
| 64×512×96 | 0.865 | 0.658 | 0.952 |

Large square products benefit from MPS. Small products are slower because
submission and synchronization dominate. The irregular 127×65×193 case does not
show a resident speedup. Transfers can erase the benefit for smaller rectangular work.
No automatic device selection or CPU fallback is applied.

## Complete optimizer steps

Each step updates four 32,768-element parameters using existing gradients.
Moment initialization and the first warm-up update occur before timing.
GPU moments and parameters remain resident. Every measured GPU step includes
all parameter arithmetic, synchronization, immutable replacement, and old-state release.
CPU uses the existing fused numerical kernels.

| Optimizer | CPU milliseconds | Metal milliseconds |
| --- | ---: | ---: |
| SGD | 0.141 | 0.603 |
| Adam | 0.546 | 2.655 |

These standalone updates are slower on Metal. GPU optimizers allow training to
remain resident, but the current unfused arithmetic incurs encoding and dispatch overhead.
This result does not justify a speedup claim for GPU training in general.

The separate linear example learns `y = 2x + 1` in 100 SGD steps, ending near
`9.95e-13` MSE. A small convolution/pooling classifier trained with Adam reduces
cross-entropy from approximately 0.64618 to 0.000369 in 60 steps.
Thirty consecutive classifier runs passed the per-iteration allocation-baseline check.

## YOLOv8n latency

The model is YOLOv8n with folded BatchNorm and 80 classes. Input is the existing
810×1080 bus.jpg. Options use confidence 0.25, class-aware IoU 0.45, and at most 300 detections.
Loading the local model takes a median 254.17 ms. Uploading its weights takes 3.51 ms.
Device initialization and initial native graph compilation are excluded from these warm latency rows.
Cold graph construction was visibly slower and must be included when assessing one-shot applications.

| Stage | 320 milliseconds | 640 milliseconds |
| --- | ---: | ---: |
| Input upload | 0.255 | 0.951 |
| CPU forward | 416.345 | 1714.178 |
| Metal forward, including synchronization | 12.755 | 24.335 |
| Output download | 0.162 | 0.261 |
| CPU decoding, NMS, and coordinate restoration | 0.673 | 2.366 |
| Complete CPU detection | 428.598 | 1728.983 |
| Complete Metal detection | 28.654 | 98.374 |

Forward, download, postprocessing, and complete-detection rows are independent
measurements. Their medians must not be added to reconstruct a complete latency.
Complete detection includes CPU image preparation and allocation costs.

Metal forward samples range from 12.43–19.68 ms at 320 and 21.07–50.43 ms at 640.
Complete Metal detection samples range from 27.38–32.02 ms and 86.45–101.09 ms.
Those ranges show meaningful local variation. More samples and controlled hardware
conditions are needed before setting a regression threshold or latency guarantee.

## Numerical evidence

The same converted model and exact input.f32 files are evaluated on CPU and Metal.
Every raw head element uses `abs(got-want) <= 0.001 + 0.0001*abs(want)`.
Detection counts and class IDs match exactly. Confidence tolerance is 0.0001.
Final box tolerance is 0.1 original-image pixels.

| Input | Maximum observed absolute CPU/Metal head error | Detections |
| --- | ---: | --- |
| 320×320 | Below 0.00008 | One bus and three people. |
| 640×640 | Below 0.00008 | One bus and four people. |

CPU is also checked against the existing pinned PyTorch reference workflow.
This connects GPU execution to the independently generated reference without
changing the input bytes or preprocessing implementation.
These comparisons establish numerical parity for the recorded examples, not dataset accuracy.

Other real-GPU tests cover Dense, RNN, LSTM, causal attention, and transformer forward/backward, native
convolution derivatives, overlapping pooling gradients, nearest/bilinear/bicubic
interpolation gradients, SGD and Adam references, leaf-gradient accumulation,
portable checkpoint round trips, foreign-device errors, malformed views,
concurrent scopes, and shutdown with pending commands.

## Memory and lifecycle

YOLO weights occupy 12,607,552 tracked native bytes across 126 buffers.
The recorded parity run observed tracked peaks of approximately 58.8 MB at 320
and 200.9 MB at 640. Input and head ownership can change these high-water marks.
GPU intermediates remain on-device. Output download occurs only at explicit host access or postprocessing.

Stats counts owned native buffers and buffers retained by queued commands.
It excludes MPS-private workspaces, shader/compiler caches, and process-wide memory.
Go B/op and allocs/op count Go allocations only. These measurements are not a total
GPU-memory peak or a guarantee that the driver releases physical pages immediately.
The repeated training test checks a stable 584-byte resident baseline after
closing each scope, clearing gradients, and synchronizing. That tiny fixture is a
lifecycle regression test, not a representative training-memory budget.

## Local artifacts

| Artifact | SHA-256 |
| --- | --- |
| Converted yolov8n.goml | `10ae8563016dd27ed7f49483626fdc8e3b36c4dcf3af37096ba937df7b93eab2` |
| Original yolov8n.pt | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| bus.jpg | `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |

Weights, images, and reference tensors stay in ignored data/yolo and are not committed.
Ultralytics/PyTorch conversion and reference versions are recorded in the
[existing validation report](../yolo-validation.md).

## Remaining release and hardware limits

Native validation covers this M1 Pro. Other Apple GPUs and native amd64 execution
remain separate validation work. Cross-compilation and unsupported-platform tests
establish build behavior only.

New downstream standalone tests require published tensor v0.3.0, nn v0.5.0, and YOLO v0.3.0.
Their checksums must be generated after upstream publication. No tags are pushed by this work.
The Metal CI workflow requires a configured self-hosted Mac GPU runner and has not
been executed on GitHub merely by adding the workflow file.
CUDA and ROCm remain explicitly unimplemented.
