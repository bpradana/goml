# WebGPU validation and performance

[WebGPU backend](../../contrib/backend/webgpu/README.md) · [Backend contract](../backend-contract.md)

Validated locally on 2026-11-17 with Go 1.27.3, macOS 27.0, Apple M1 Pro,
16 GiB unified memory, and eight Go processors. The selected adapter reports
`Apple M1 Pro (metal)` through wgpu-native. The Go binding is pinned at v1.38.1;
its platform-library versions and checksums are recorded in the WebGPU module.
No raw benchmark output, profiles, model weights, or datasets are committed.

## Correctness and ownership

The common backendtest suites passed on real hardware for Device, Unary, Binary,
Scalar, MatMul, Reduction, Fill, Copy, Concat, Finite, Convolution/Backward,
Pool/Backward, Resize/Backward, and Tensor integration. There is no CPU fallback.
Each implemented source file has its corresponding test file.

The checks cover rectangular and irregular matrices, empty dimensions, strided
inputs, scatter destinations, broadcasting, invalid layouts, foreign buffers,
closed contexts, overlapping writes, concurrent independent allocations, scopes,
shared tensor leases, and source buffers closed before pending copies complete.
Shader compilation failures return errors and leave a valid context usable.
Injected callback failures are isolated to one context and still permit cleanup;
they do not simulate every possible physical GPU loss.

Existing CPU CI passed formatting, build, vet, lint, unit tests, race checks, and
govulncheck. The CPU vulnerability scan reported no reachable vulnerabilities and
one uncalled module advisory. Linux amd64/arm64 CPU cross-builds passed with and
without SIMD and CGo disabled. Native WebGPU and model race checks passed;
unsupported WebGPU stubs compiled for Linux, Windows, and js/wasm.

Additional cases compare nonconstant nearest, bilinear, and bicubic values and
gradients with CPU across singleton/non-square images and scales one through three.
Arithmetic checks include NaNs, infinities, division by zero, and signed zero.
Transfers and strided copies preserve tested Float32 bit patterns, including NaN
payloads. This does not promise identical NaN payloads after arithmetic.

## Neural networks and checkpoints

Forward outputs and gradients matched CPU for Dense, Conv2D, LayerNorm, attention,
transformers, RNN, and LSTM. Model comparisons use `0.001 + 0.001*abs(reference)`
for outputs and `0.003 + 0.003*abs(reference)` for gradients. The shared numerical
suites use their tighter operation-specific tolerances.

The seeded linear training example reduced MSE from `6.5598497` to
`9.947598e-13` in 100 SGD updates. It returned to zero tracked tensor allocations
after cleanup. GPU parameters were saved through explicit host reads and restored
on CPU using the existing checkpoint format. The restored network retained its
predictions without requiring a WebGPU context.

SGD and Adam matched CPU over repeated updates, preserving parameter identity and
device placement. Tracked resident bytes returned to a stable baseline after each
step, including Adam's persistent device moments.

## Pretrained YOLOv8n

Local converted weights, bus.jpg, and the existing reference inputs were used.
No assets were downloaded by the tests. WebGPU was compared directly with CPU
heads and detections; this is fixture parity, not a new dataset accuracy study.

| Input | Maximum raw-head absolute difference | Retained detections |
| --- | ---: | ---: |
| 320×320 | 0.0000522137 | 4 |
| 640×640 | 0.0000553131 | 5 |

The comparison checks every raw logit with `0.001 + 0.0001*abs(reference)`,
confidence within `1e-4`, class IDs exactly, and bounding-box coordinates within
0.1 input-image pixels. Both repeated forwards produced the same recorded maximum
differences. Final model cleanup returned tracked native tensor bytes to zero.

The 126 weights occupied 12,607,552 tracked bytes. Observed peak tensor-buffer
storage was 52,991,552 bytes at 320 and 174,143,552 bytes at 640. These totals
include closed allocations retained by pending commands, but exclude WebGPU
parameter/readback buffers and driver-private allocations.

The following are observations from the validation run, not repeated benchmarks:

| Stage | 320×320 | 640×640 |
| --- | ---: | ---: |
| CPU forward | 420.13 ms | 1683.00 ms |
| First GPU encode plus synchronization | 86.77 ms | 211.49 ms |
| Repeated GPU encode plus synchronization | 55.60 ms | 185.49 ms |
| Repeated output download and numerical comparison | 7.71 ms | 20.87 ms |
| Complete GPU detection, including CPU preprocessing/postprocessing | 91.09 ms | 271.22 ms |

Weight upload took 13.93 ms in this run. Model loading, image preparation, GPU
encoding, synchronization, output readback, and CPU decoding/NMS are distinct
costs; a resident forward is not complete detection latency.

## Repeated benchmarks

The table reports medians of three 200 ms Go benchmark samples. Inputs are
initialized before resident timing. Every resident iteration includes output
allocation, encoding, synchronization, and output release. The transfer-inclusive
case also uploads both operands and downloads the result. Go allocation counts
exclude native memory.

| MatMul dimensions M×K×N | CPU | WebGPU resident | WebGPU with transfers |
| --- | ---: | ---: | ---: |
| 8×8×8 | 0.000614 ms | 1.500 ms | 1.695 ms |
| 31×17×9 | 0.002963 ms | 1.508 ms | 1.828 ms |
| 128×128×128 | 0.605 ms | 1.510 ms | 1.900 ms |
| 256×64×128 | 0.611 ms | 1.561 ms | 2.000 ms |
| 512×512×512 | 43.148 ms | 5.902 ms | 7.289 ms |

CPU used seven Go allocations per product. Resident WebGPU used 19; the
transfer-inclusive case used 43–45. The 512-square resident case allocated about
936 Go bytes per iteration versus 1,048,848 CPU bytes; native GPU outputs are
additional allocations and are not represented by those Go-byte totals.

Context startup, including pipeline construction and shutdown, measured 6.826 ms
and 86 Go allocations. A 4096-element upload/download round trip measured
1.450 ms and five Go allocations. The driver may reuse compiled shader caches;
these are not cold-machine or first-process startup measurements.

CPU is faster for the small matrices measured here. Submission and synchronization
overhead dominate those workloads. The tiled WGSL path benefits the tested large
matrix, but this is not a universal speedup claim. The direct convolution and
serial-per-output reduction kernels are initial implementations, and the
specialized Metal/MPS backend remains a separate performance choice.

## Reproduce

```sh
mise run test:gpu:webgpu
mise run test:gpu:webgpu:standalone
mise run test:gpu:webgpu:race
mise run test:gpu:webgpu:parity
mise run test:gpu:webgpu:parity:race
mise run test:gpu:webgpu:portable
mise run bench:gpu:webgpu
GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml" mise run test:gpu:webgpu:yolo
```

Standalone backend tests use published Tensor and binding versions with GOWORK
disabled. The separate examples module remains workspace-only until WebGPU's
initial tag is published and its real versioned requirement can be added.

Linux/Windows GPU execution, browser execution, native amd64 hardware, physical
GPU-loss recovery, and cold driver startup were not validated here. CGo-free
cross-builds establish stub compatibility, not GPU execution on those platforms.
The opt-in hardware workflow requires a configured self-hosted Mac runner;
configured workflow files are not evidence that hosted CI has run.
