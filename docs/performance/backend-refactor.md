# Backend refactor validation

[Backend contract](../backend-contract.md) · [Contributor tutorial](../extending-backends.md) · [API migration](../backend-migration.md)

Validated locally on 2026-11-17 with Go 1.27.3, macOS 27.0, Apple M1 Pro,
darwin/arm64, and eight Go processors. The comparison source is `1202d83`, before
the backend refactor. Raw measurements, profiles, and the original source archive
stay outside the repository. No published module tag was changed.

## Execution audit and implementation evidence

The original execution path selected CPU through a nil device and direct Go-slice
storage. Accelerated operations used Tensor-owned interfaces and string operation
names. NeuralNet and checkpoint callers depended on CPU-only Data/Contiguous
methods and GPU-only lifetime rules. Those paths now share device-owned buffers,
optional capabilities, and explicit handles.

| Phase | Changed source | Verification and result |
| --- | --- | --- |
| Audit and baseline | Original Tensor, nn, checkpoint, YOLO, and Metal callers; existing benchmark harnesses | Recorded scalar kernels, convolution forward/backward, inference, resident/transfer MatMul, optimizers, and YOLO stages before edits. |
| Contract | `tensor/backend` | Contracts compile without Tensor, NeuralNet, or concrete backend imports. Typed operations, optional capabilities, physical views, and errors have direct tests. |
| CPU migration | `tensor/internal/cpu`, `tensor/internal/shape` | Preserved SIMD arithmetic, blocked MatMul, tiled convolution, traversal, and fused updates. CPU passes every capability suite it claims. |
| Storage and dispatch | Tensor storage, transfer, view, arithmetic, reduction, convolution, and spatial files | Default CPU is explicit. Views and scopes retain independent handles. Closed resources, unsupported capabilities, overlapping writes, and allocation byte overflow are tested. |
| Metal migration | Metal device/buffer, arithmetic, MatMul, reduction, layout, convolution, spatial, and validation files | Real hardware passes shared conformance. Production imports use `tensor/backend`; native Objective-C and shader algorithms are unchanged. |
| Conformance | `tensor/backendtest` | Independent references cover arithmetic, reductions, matrices, convolution gradients, pooling, interpolation, patches, concatenation, host access, fused updates, ownership, and concurrency. |
| NeuralNet integration | nn device/parameter/graph, activation, layer, loss, optimizer, checkpoint, and YOLO callers | CPU and Metal gradients, optimizers, inference, checkpoints, and training examples pass. Saved parameter generations survive updates; active graphs prevent cross-device parameter moves. |
| Contributor experience | Backend contract/tutorial/migration guides, module READMEs, `tensor/examples/backend` | A MatMul-only Go-memory backend compiles and passes Device/MatMul suites using only public contracts. Source files have matching numerical tests. |
| Stabilization | mise backend tasks, regression/race/fuzz/SIMD/portable checks | Local quality gates pass. CPU builds remain free of CGo. CUDA and ROCm remain documentation-only modules outside the workspace. Downstream standalone checks remain publication-gated. |

## Numerical and lifetime results

`mise run ci` passed formatting, build, vet, lint, unit tests, CPU race tests, and
vulnerability analysis. Govulncheck reported no reachable vulnerabilities; it also
reported an uncalled advisory in the required module graph. Separate native Metal
lint, conformance, model parity, and race checks passed on real hardware.

Scalar and experimental SIMD tests passed. Linux amd64/arm64 scalar and SIMD
cross-builds passed without CGo. Unsupported-platform Metal builds and tests also
passed with CGo disabled. Bounded fuzzing passed for shapes, broadcasting, backend
views, Tensor views, checkpoints, architecture restoration, and YOLO archives.
The backend-view fuzz run alone executed 266,907 cases without a failure.

A resident CNN trained with Adam from approximately 0.646178 cross-entropy to
0.000369189, with 584 stable tracked resident bytes between iterations. Shared
views, scoped temporaries, pending commands, foreign noncomparable buffers,
idempotent shutdown, and saved parameter generations passed lifetime tests.
MPS implementation-private workspaces are not included in tracked buffer totals.

## Pretrained YOLOv8n

The existing converted archive and bus-image fixtures were used without network
downloads. CPU inference matched saved PyTorch heads and detections at both sizes.
Metal matched CPU heads, object counts, class IDs, confidence values, and boxes.
Final model shutdown returned tracked native allocations to zero. The 126 model
weights occupied 12,607,552 tracked bytes; observed peak tracked storage was
56,767,552 bytes at 320 and 201,023,552 bytes at 640. These totals exclude
MPS implementation-private allocations.

| Input | Maximum CPU/PyTorch logit difference | Maximum Metal/CPU logit difference | Detections |
| --- | ---: | ---: | ---: |
| 320×320 | 0.000154913 | 0.0000572205 | 4 |
| 640×640 | 0.000112534 | 0.0000782013 | 5 |

Comparisons use the existing Float32 absolute-plus-relative logit tolerances,
confidence tolerance `1e-4`, and bounding-box tolerances in the integration tests.
This is fixture parity, not an accuracy evaluation over a detection dataset.

The first GPU forward included native graph construction: encode plus synchronize
was about 420 ms at 320 and 295 ms at 640. Repeated forwards were about 31 ms and
43 ms in that validation run. Startup and warmed benchmark measurements are
therefore reported separately. These short observations are sensitive to driver
caching and other machine activity.

## Measurement method

CPU kernel and inference rows use medians of three 100 ms samples. Existing Metal
MatMul comparisons were repeated with five 500 ms samples after shorter runs
showed substantial timing variation. Model stages use three samples of three
iterations. Inputs and model weights are resident before warmed GPU timings.
Resident rows include output allocation, encoding, explicit synchronization, and
release. Transfer-inclusive and complete detection rows include their stated host
work. Go allocation counts exclude native memory.

New lifecycle, isolated convolution, and complete Dense backward harnesses were
run against both revisions with identical public-API benchmark bodies. Context
startup includes shader-library compilation and pipeline construction for each new
device; the Metal driver may reuse its own cache. It is not a cold-machine boot
measurement. Convolution graphs are warmed before resident timing. The Dense
backward harness excludes device and weight initialization; native lazy startup
is amortized across its timed iterations.

## CPU kernels and ownership cost

The numerical algorithms remain unchanged. Unified buffer ownership adds two Go
objects to an ordinary CPU result and introduces checked operation-boundary
dispatch. Profiling exposed unnecessary cleanup registration and repeated layout
validation; these were removed. CPU allocations identify fully garbage-collected
storage through the optional GCManagedBuffer capability, while native allocations
retain their cleanup safety net.

| CPU operation | Before | After | Go allocations before/after |
| --- | ---: | ---: | ---: |
| MatMul 4×4×4 | 0.144 µs | 0.380 µs | 5 / 7 |
| MatMul 128×128×128 | 0.607 ms | 0.604 ms | 5 / 7 |
| MatMul 512×512×512 | 41.943 ms | 41.819 ms | 5 / 7 |
| Add contiguous 64×64 | 4.066 µs | 4.399 µs | 5 / 7 |
| Add broadcast 64×64 | 4.622 µs | 4.894 µs | 5 / 7 |
| Sum selected axis | 12.010 µs | 12.378 µs | 6 / 8 |
| YOLO640-shaped Conv2D forward | 30.326 ms | 30.976 ms | 11 / 15 |
| YOLO640-shaped Conv2D backward | 84.739 ms | 85.848 ms | 148 / 189 |

| CPU inference | Before | After | Go allocations before/after |
| --- | ---: | ---: | ---: |
| Dense Sequential | 1.588 µs | 2.349 µs | 19 / 25 |
| CNN | 14.678 µs | 15.556 µs | 11 / 15 |
| RNN | 22.743 µs | 37.533 µs | 441 / 542 |
| LSTM | 72.888 µs | 107.274 µs | 1023 / 1254 |
| Transformer | 69.796 µs | 89.776 µs | 1007 / 1162 |

A second paired inference run showed substantial timing variation: RNN measured
58.290 µs before and 34.418 µs after, while LSTM measured 89.485 µs before and
110.519 µs after. Allocation increases are repeatable even when short timing
samples vary. The added handles, buffer references, validation, and graph lifetime
bookkeeping are a remaining cost of this architecture, especially when individual
kernels are tiny.

Small CPU models retain measurable metadata and lifetime overhead. This report
does not claim identical tiny-model latency. Large kernels retain the earlier
blocking, packing, SIMD, and bounded-patch improvements. Short-run differences
are observations, not statistically established regressions or speedups across
machines. Further work should target measured storage/validation overhead rather
than change unrelated numerical algorithms to compensate.

## Metal lifecycle and kernels

| Metal workload | Before | After | Go allocations before/after |
| --- | ---: | ---: | ---: |
| Context startup, shader compilation, and shutdown | 1.142 ms | 1.163 ms | 8 / 8 |
| Allocate/release 4096 Float32 elements | 8.338 µs | 7.992 µs | 2 / 2 |
| Upload 16 KiB, idle queue | 0.556 µs | 0.707 µs | 1 / 1 |
| Download 16 KiB, idle queue | 0.595 µs | 0.736 µs | 1 / 1 |
| Resident MatMul 256×256×256 | 0.270 ms | 0.268 ms | 12 / 11 |
| Resident MatMul 1024×1024×1024 | 1.094 ms | 1.103 ms | 12 / 11 |
| Resident Conv2D forward | 0.897 ms | 0.557 ms | 16 / 15 |
| Resident Conv2D backward | 0.899 ms | 0.711 ms | 24 / 22 |
| Dense forward, loss, backward, and gradient clearing | 2.044 ms | 1.855 ms | 254 / 211 |

The MatMul rows are medians of the longer five-sample runs; both are within 1% of
the original implementation. The convolution harness uses NCHW `[1,8,32,32]`,
16 output channels, a 3×3 kernel, stride one, and padding one. Dense uses batch 32
and a 64→32 layer. Both include explicit synchronization and temporary cleanup.
Transfer rows measure idle shared-memory copies and validation, not end-to-end
transfers with queued model execution. These samples do not establish universal
speedups for convolution or backward execution.

## Complete resident workloads

| Workload | Before | After |
| --- | ---: | ---: |
| YOLO model load | 252.34 ms | 253.33 ms |
| YOLO weight upload | 3.54 ms | 3.58 ms |
| SGD, four 32,768-element parameters, Metal | 0.688 ms | 0.631 ms |
| Adam, four 32,768-element parameters, Metal | 2.705 ms | 2.567 ms |
| YOLO320 CPU forward | 416.62 ms | 424.95 ms |
| YOLO320 Metal forward | 14.04 ms | 13.53 ms |
| YOLO320 Metal complete detection | 29.38 ms | 26.39 ms |
| YOLO640 CPU forward | 1671.10 ms | 1721.71 ms |
| YOLO640 Metal forward | 23.27 ms | 22.35 ms |
| YOLO640 Metal complete detection | 96.73 ms | 93.73 ms |

Metal forward Go allocations fell from 4,400 to 2,962 per YOLO pass. Adam-step Go
allocations fell from 883 to 623. The native mathematical kernels are unchanged;
these rows describe dispatch, ownership, and ordinary timing variation, not new
GPU numerical algorithms.

## Reproduce and release

```sh
mise run test:backend
mise run test:backend:metal
mise run test:gpu:parity
mise run test:gpu:race
mise run bench:backend:cpu
mise run bench:backend:metal
```

Enable pretrained model measurements explicitly with GOML_YOLO_MODEL as described
in the [Metal performance guide](metal.md). Raw output stays outside the repository.

Standalone simdops and Tensor tests passed with GOWORK disabled, as did the CUDA
and ROCm placeholder packages. No compatible tensor/v0.3.0, nn/v0.5.0, or YOLO/v0.3.0
tag was present on the remote during validation. The aggregate standalone task
passed simdops and Tensor, then stopped at nn with missing go.sum entries for
Tensor and its backend package. Publish upstream versions in
order, generate real downstream checksums, and rerun standalone nn/YOLO/Metal
checks before releasing those modules. Workspace tests cannot establish public
dependency resolution. No relative replacements or fabricated sums were added.

Native amd64 execution, other Metal GPUs, fully cold driver startup, and hosted CI
execution were not measured here. CUDA and ROCm have no GPU execution support and
are not release targets.
