# CPU performance measurements

Measured on 2026-11-16 using Go 1.27.3 through mise, darwin/arm64, Apple M1 Pro,
with the default eight Go processors. The baseline is commit
`caa070660ae783f97f188a400ee1b65fb0a15449`. The optimized results describe the
uncommitted source snapshot retained by `mise run bench:record`.

## Method

Inputs use PCG seeds `(1,2)` unless a correctness test specifies another seed.
Numbers below are medians of three runs, with `-benchmem -benchtime=100ms` and
`-p=1` to serialize benchmark packages. Each operation itself is single-threaded.
Build flags are the default, with no required CGo or SIMD experiment. Both scalar
and `GOEXPERIMENT=simd` results were recorded separately. GFLOPS counts `2*M*N*K`
operations divided by elapsed time.

The existing benchmark tasks were run before implementation changes. Larger
matrix, convolution, copying, optimizer, and 320/640 YOLO workloads were then
added before recording an expanded baseline. A second baseline run used an
archive of the original commit with the same expanded benchmark definitions and
serialized package execution. Original results were retained rather than
overwritten. CPU and allocation profiles were collected separately from the
final benchmark run.

Local pretrained artifacts are `data/yolo/yolov8n.goml`, `bus.jpg`, and the
existing `reference-320` and `reference-640` directories. These large artifacts
are not included in source control. Forward benchmarks use an all-ones NCHW
input. Detection benchmarks include preparation, inference, decoding, and NMS
for a deterministic blank 640x480 NRGBA image. Numerical validation uses the bus
image and its existing PyTorch references. Latencies therefore describe these
specific workloads, rather than arbitrary detection images.

## Results

Scalar results; allocation counts include output storage and metadata. Layer
and training improvements include the shared numerical kernel changes.

| Workload | Before | After | Speedup | Before / after B/op | Before / after allocs/op |
| --- | ---: | ---: | ---: | ---: | ---: |
| MatMul 4x4x4 | 225.6 ns | 120.7 ns | 1.87x | 208 / 208 | 5 / 5 |
| MatMul 32x32x32 | 46.84 us | 8.46 us | 5.54x | 4,240 / 4,240 | 5 / 5 |
| MatMul 128x128x128 | 2.962 ms | 0.502 ms | 5.90x | 65,680 / 65,680 | 5 / 5 |
| MatMul 512x512x512 | 182.99 ms | 32.73 ms | 5.59x | 1,048,720 / 1,048,757 | 5 / 5 |
| MatMul 1024x64x128 | 12.98 ms | 1.927 ms | 6.74x | 524,446 / 524,440 | 5 / 5 |
| MatMul transposed RHS, 129x129 | 3.193 ms | 0.503 ms | 6.35x | 73,872 / 147,600 | 5 / 6 |
| Add row broadcast, 64x64 + 64 | 30.95 us | 3.89 us | 7.96x | 16,528 / 16,528 | 5 / 5 |
| Add column broadcast, 64x64 + 64x1 | 31.15 us | 7.29 us | 4.27x | 16,528 / 16,528 | 5 / 5 |
| Add transposed 64x64 | 32.62 us | 7.25 us | 4.50x | 16,528 / 16,528 | 5 / 5 |
| Data contiguous 256x256 | 256.81 us | 18.15 us | 14.15x | 262,144 / 262,144 | 1 / 1 |
| Data transposed 256x256 | 285.35 us | 71.50 us | 3.99x | 262,144 / 262,146 | 1 / 1 |
| Sum axis 0, 64x64 | 41.43 us | 9.29 us | 4.46x | 392 / 392 | 6 / 6 |
| Conv2D CNN inference | 35.68 ms | 5.92 ms | 6.03x | 8,914,560 / 573,792 | 42 / 11 |
| Conv2D CNN forward + backward | 110.37 ms | 18.81 ms | 5.87x | 16,277,992 / 8,169,181 | 164 / 148 |
| Conv2D YOLO640 first layer inference | 109.26 ms | 24.40 ms | 4.48x | 57,755,376 / 6,564,960 | 43 / 11 |
| Autograd square + mean, 32x32 | 38.19 us | 10.06 us | 3.80x | 39,254 / 30,704 | 74 / 64 |
| SGD, 65,536 parameters | 248.16 us | 63.34 us | 3.92x | 524,541 / 262,273 | 11 / 6 |
| Adam, 65,536 parameters | 524.81 us | 163.69 us | 3.21x | 788,894 / 263,170 | 9 / 6 |
| Dense sequential inference | 2.88 us | 1.33 us | 2.16x | 1,648 / 1,648 | 19 / 19 |
| RNN inference | 37.85 us | 17.50 us | 2.16x | 37,120 / 32,192 | 473 / 441 |
| LSTM inference | 145.08 us | 59.09 us | 2.46x | 109,264 / 101,776 | 1,119 / 1,023 |
| Transformer encoder inference | 110.59 us | 54.80 us | 2.02x | 95,224 / 87,240 | 1,113 / 1,002 |

The small differences in measured bytes for large allocations reflect runtime
sampling/amortization. The ordinary MatMul path still allocates only its result
and existing metadata. The 512-square throughput increased from 1.47 to 8.20
GFLOPS. These are short repeated local measurements, not statistical confidence
intervals or promises for other CPUs.

CNN convolution uses `[4,16,32,32]` input, 32 output channels, a 3x3 kernel,
stride 1, padding 1. The YOLO640 first-layer workload uses `[1,3,640,640]`, 16
output channels, a 3x3 kernel, stride 2, padding 1. Dense uses `[8,16]` input and
eight outputs followed by tanh. RNN/LSTM use `[4,8,8]` input and hidden width 16.
Transformer encoder uses `[2,8,16]`, four heads, feed-forward width 32.

### YOLOv8n

These timings come from a separate paired run of the preserved baseline and
optimized test binaries, with the same `-benchtime=100ms -count=3` settings and
no active profiler. The earlier profiled baseline was slower; it is retained
locally but is not used for the speedups below. Complete detection includes
preprocessing and postprocessing; forward timing excludes them.

| Workload | Before | After | Speedup | Before / after B/op | Before / after allocs/op |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward 320x320 | 1.493 s | 0.318 s | 4.70x | 339,107,584 / 66,432,064 | 3,554 / 1,461 |
| Forward 640x640 | 5.980 s | 1.277 s | 4.68x | 1,350,055,136 / 234,815,760 | 3,554 / 1,464 |
| Detection 320x320 | 1.474 s | 0.325 s | 4.54x | 344,023,168 / 71,347,888 | 310,766 / 308,676 |
| Detection 640x640 | 5.880 s | 1.302 s | 4.52x | 1,369,650,688 / 254,410,976 | 1,232,366 / 1,230,273 |

Forward allocated bytes fell by 80.4% at 320 and 82.6% at 640. Detection's high
remaining allocation count comes mainly from the existing per-pixel image
conversion path; it was not a material CPU hotspot relative to convolution.

Peak RSS was measured separately using macOS `/usr/bin/time -l`, each in a
fresh test process with one benchmark iteration. It includes model loading,
input creation, Go runtime, and the operation. It is a process high-water mark,
not the sum of allocated bytes or an isolated activation-memory measurement.

| Fresh-process workload | Before peak RSS | After peak RSS |
| --- | ---: | ---: |
| YOLO forward 320x320 | 80.94 MiB | 50.92 MiB |
| YOLO forward 640x640 | 196.30 MiB | 108.44 MiB |
| CNN convolution inference | 16.52 MiB | 8.31 MiB |
| CNN convolution forward + backward | 20.06 MiB | 15.56 MiB |

Both pretrained comparisons pass with the existing tolerances and the same
maximum logit differences as the baseline: `0.000154912472` at 320 and
`0.000112533569` at 640. All head values and final detections are checked by the
existing PyTorch-reference test. This does not claim bitwise agreement with
PyTorch or cross-platform benchmark parity.

`BenchmarkModelStages` captures actual upstream activations once and measures
each backbone/neck stage and all six head branches independently. Both image
sizes were recorded with three 50ms samples. The largest optimized 640 workloads
are shown below. These warm-input stage timings exclude routing and setup and
must not be summed to estimate full-model latency.

| YOLO640 stage | Before | After | Before / after B/op |
| --- | ---: | ---: | ---: |
| Classification head, stride 8 | 877.87 ms | 177.79 ms | 138,040,704 / 14,755,360 |
| Regression head, stride 8 | 637.32 ms | 133.63 ms | 119,608,704 / 11,830,816 |
| Backbone stage 04 | 449.40 ms | 105.00 ms | 155,661,536 / 24,990,832 |
| Backbone stage 02 | 344.30 ms | 91.61 ms | 190,063,648 / 36,175,488 |
| Backbone stage 06 | 403.67 ms | 86.12 ms | 77,837,424 / 13,112,432 |
| Neck stage 15 | 339.94 ms | 82.13 ms | 107,324,192 / 17,476,992 |

The updated CPU profile still concentrates work in matrix multiplication and
patch extraction. No pooling, upsampling, recurrent, or attention rewrite was
justified by these profiles.

## Changes and evidence

### Matrix multiplication

The original CPU profile spent 98% of the isolated workload in MatMul, with
repeated stride arithmetic inside its innermost loop. The new kernel blocks
32 rows, 128 columns, and 64 features, reusing each RHS row across four output
rows. A direct strided dot-product path handles small transposed products.
Larger transposed RHS operands are packed once when `M>=8`, `K>=16`, `N>=8`.
Packing costs one operand-sized allocation and improves the measured 129-square
transposed product by 6.35x. There is no shared cache or parallel scheduling.

Every output retains increasing-K Float32 accumulation. The original Float32
loop remains in tests alongside an independent Float64 reference. Tests cover
tile boundaries, offset slices, both operand transposes, empty dimensions,
nonfinite products, and exact Float32 parity for finite reference cases.
No existing correctness tolerance was relaxed.

### Convolution and copies

The original allocation profile attributed 36% of sampled allocation bytes to
`Tensor.Data` and another 22% to float-slice cloning. Training now populates one
owned patch tensor through `Im2Col`; `Col2Im` scatters gradients directly into
their final owned storage. Input strides are consumed directly. The old
`Zeros -> Data -> From` patch copy chain is gone.

Inference uses 64-position patch tiles and a reusable tile product buffer, then
writes NCHW output and bias directly. Scratch storage is at most
`64*(inputChannels*kernel*kernel+outputChannels)*4` bytes, independent of spatial
area and batch size. For the CNN workload this is 45,056 bytes instead of a
2,359,296-byte full patch matrix (previously copied into multiple buffers).
Training still retains a full patch matrix because weight differentiation needs
the forward patches; it is not a bounded-memory training implementation.

The numerical kernels live in `tensor` so backing arrays remain private.
`nn/layer/conv2d.go` retains parameter ownership, configuration, graph wiring,
and the existing serialized weight shape. No mutable tensor storage is exported.
Finite differences, overlapping gradients, stride/padding, empty batches, and
tiled inference versus training on offset-strided inputs are tested.

Three alternatives are retained as benchmarks, not an algorithm framework:

| YOLO320 first convolution, no bias | Latency | B/op |
| --- | ---: | ---: |
| Direct reference kernel | 40.60 ms | 1,638,656 |
| Full im2col | 6.28 ms | 6,046,592 |
| Tiled im2col | 6.45 ms | 1,649,664 |

This comparison favors tiled inference for its memory reduction at similar
latency. The direct reference was slower on the tested workloads, and was not
added to production dispatch.

`Data()` now checks contiguity once and uses `copy` for contiguous storage.
Strided copies advance coordinates incrementally. `Contiguous()` still returns
independent storage, and views still share immutable backing storage. Tensor
concatenation now copies directly into final output storage; neural-network
concatenation delegates to it instead of copying every input and the output.

### Arithmetic, reductions, and training

Broadcast arithmetic uses effective strides and an incremental outer iterator.
Contiguous inner rows use the existing scalar/SIMD kernels, including ordinary
row broadcasting. Scalars and column broadcasts use zero effective strides.
Reductions use contiguous group loops when reduced axes form a suffix and
incremental source/destination coordinates otherwise. Both retain logical
element order and the existing NaN, infinity, empty-group, and axis semantics.

Autograd reuses its per-pass adjoint map to stage accumulated gradients before
committing, rather than allocating another map. Contiguous gradient reshapes
share immutable storage instead of copying values. MatMul backward computes
only gradients needed by tracked parents. A late-rule-error test verifies that
existing gradients survive unchanged and the graph can be retried successfully.

SGD writes one result through `SubScaled`, explicitly rounding the scaled
Float32 product before subtraction. Adam reads tensors directly, writes one
result, and updates only caller-owned moment slices. Bias corrections and the
original moment expressions are retained. Missing-gradient steps do not create
or advance moment state; parameter identity and previous tensors are preserved.

Inference slices skip unused backward metadata. Softmax reuses an axis-width
Float64 exponential buffer instead of evaluating each exponential twice; its
Float64 normalization and nonfinite behavior are unchanged. The measured
recurrent and attention gains primarily come from shared kernels and copies;
there is no replacement recurrent or attention execution engine.

### SIMD experiment

Existing scalar and experimental SIMD execution paths remain available.
Row-wise `simdops.Axpy` was benchmarked as an alternative MatMul inner kernel.
For 512-square products it took approximately 86 ms in scalar mode and 80 ms
with SIMD, versus 33–42 ms for the blocked four-row kernel. It was rejected for
production dispatch. The test-only experiment remains in `benchmark_test.go`.
Contiguous arithmetic continues to use SIMD when built with the experiment.

## Reproduction and artifacts

```sh
mise run bench:matmul
mise run bench:conv2d
mise run bench:tensor
mise run bench:autograd
mise run bench:optimizer
mise run bench:attention
mise run bench:inference
GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml" mise run bench:yolo
GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml" mise exec -- go test \
  -run '^$' -bench='^BenchmarkModelStages$' -benchmem -benchtime=50ms -count=3 ./contrib/nn/yolo
GOML_BENCH_DIR=benchmark-results/my-run \
  GOML_YOLO_MODEL="$PWD/data/yolo/yolov8n.goml" mise run bench:record
```

`bench:record` creates a timestamped directory by default. It records hardware,
toolchain, source commit/dirty status, flags, model path, a source archive that
includes untracked kernels, and separate repeated scalar/SIMD results. Existing
recordings are retained. Keep the machine otherwise idle and compare like build
modes, input data, and benchmark settings.

To reproduce the expanded baseline, archive the original commit into a temporary
directory, copy the current layer/optimizer/YOLO benchmark files, and copy the
tensor benchmarks preceding `BenchmarkMatMulAxpy` (omitting the unused simdops
import). The remaining tensor benchmark functions exercise newly added kernels
and cannot compile against the original commit. Run Go from the repository's
mise environment with `go -C <baseline-directory> test -p=1 ...`.

Profiles can be collected without changing source:

```sh
mise exec -- go test -run '^$' -bench='BenchmarkMatMul/\[512' \
  -benchtime=1s -cpuprofile=/tmp/matmul.prof -memprofile=/tmp/matmul-mem.prof ./tensor
mise exec -- go tool pprof -top /tmp/matmul.prof
mise exec -- go tool pprof -top -sample_index=alloc_space /tmp/matmul-mem.prof
```

Raw measurements, profiles, source archives, and reference-test logs remain
under ignored `benchmark-results/`, including `cpu-before/`, `cpu-after/`, and
`cpu-final/`. They are local evidence and are not intended for commits. This
report retains summarized measurements and reproduction instructions only.

## Release dependencies

The release set is tensor v0.2.0, nn v0.4.0, and contrib/nn/yolo v0.2.0.
nn requires tensor v0.2.0; YOLO requires nn v0.4.0 and tensor v0.2.0. simdops
remains at v0.1.0. The workspace consumes the current source of all four modules.
Local candidate modules can be tested with `GOWORK=off` using temporary modfiles
that replace dependencies with these source directories; that verifies module
isolation. Use the versioned standalone tasks to validate published dependencies.
The [release guide](../releasing.md) documents checksum generation and the
tensor → nn → YOLO dependency order. No tags or releases were created by this work.

## Validation

- `mise run ci` passed locally: formatting, builds, vet, lint, all workspace
  tests, race tests, and vulnerability analysis. Govulncheck found no reachable
  vulnerable symbols; it reported one advisory in required modules that is not
  called by this code.
- `mise run test:simd`, `mise run test:fuzz`, and `mise run build:portable`
  passed. Portable builds covered Linux amd64 and arm64, with CGo disabled,
  in both default and experimental SIMD modes.
- All four current module candidates passed standalone tests with `GOWORK=off`
  and temporary modfiles replacing upstream dependencies with local source.
  This check used temporary files rather than adding repository replacements.
- Before the release dependency update, `mise run test:standalone` passed
  simdops and tensor but failed to build nn against the previous tensor v0.1.0
  API. The manifests now require tensor v0.2.0 and nn v0.4.0 instead.
- Both local pretrained PyTorch reference comparisons passed at 320 and 640.
  Existing layer finite differences, serialization, gradient accumulation,
  inference-mode, ownership, and graph-lifetime tests remain passing.

Remote GitHub Actions and native amd64 performance were not measured here.
