# Changelog

## tensor v0.3.0, nn v0.5.0, contrib/nn/yolo v0.3.0, and Metal v0.1.0

Publish compatible tensor, nn, and YOLO versions before the independent Metal backend.
simdops remains at v0.1.0. CUDA, ROCm, and the nested Metal examples module have no release tags.
Downstream checksums and standalone checks require the upstream tags to resolve publicly.

- Composable contracts in `tensor/backend`, typed operations, standardized errors, and explicit CPU/Metal device identities.
- Unified backend-owned storage, safe shared views, retained handles, and scopes on CPU and Metal, preserving optimized numerical kernels.
- Breaking Tensor migration: use `CPU()` instead of nil, `HostData()` instead of `Data()`, and `MakeContiguous()` instead of `Contiguous()`; same-device `To` copies while `Retain` shares.
- Reusable backend conformance suites, source-paired numerical tests, a compiled MatMul-only contributor backend, and backend development tasks.
- Saved parameter generations survive updates until graph consumption; cross-device transfers reject active parameter graphs.
- Independent Apple Metal backend with cached embedded shaders, persistent buffers, MPS MatMul, and native MPSGraph convolution gradients.
- Device-resident arithmetic, broadcasting, reductions, layout operations, activations, pooling, and all three spatial interpolation modes.
- Existing nn models, eager autograd, gradient accumulation, SGD, and Adam on GPU without intermediate array downloads.
- Explicit gradient and optimizer-state cleanup, safe views, pending-command resource retention, and concurrent cleanup synchronization.
- Portable checkpoints with explicit GPU transfers and staged restoration preserving parameter identity.
- YOLOv8n weight transfer, GPU Forward, staged Postprocess, and complete detection at 320 and 640.
- Minimal, clearly unimplemented CUDA and ROCm module reservations outside the active workspace.
- Package logos, GPU migration and performance guides, opt-in Mac hardware CI, Metal release-note automation, and runnable examples.
- Actual M1 Pro parity, convergence, allocation-lifetime, and repeated benchmark evidence in the [Metal report](docs/performance/metal.md) and [backend refactor report](docs/performance/backend-refactor.md).

## tensor v0.2.0, nn v0.4.0, and contrib/nn/yolo v0.2.0

Publish `tensor/v0.2.0`, then `nn/v0.4.0`, then `contrib/nn/yolo/v0.2.0`.
nn requires tensor v0.2.0; YOLO requires nn v0.4.0 and tensor v0.2.0.
simdops remains at v0.1.0. Generate downstream checksums after each upstream tag
is available. Workspace validation and measurements are documented in
[the CPU performance report](docs/performance/cpu.md).

- Blocked Float32 matrix multiplication with small strided execution and selective RHS packing.
- Single-allocation training patches and bounded tiled Conv2D inference without mutable tensor access.
- Incremental broadcasting, reductions, and strided copies, plus direct tensor concatenation.
- Reduced gradient copying and staging allocations, and skipped untracked MatMul derivatives.
- Fused immutable SGD and Adam parameter results with preserved moment and rounding semantics.
- Reused Softmax exponentials and omitted unused inference slice metadata.
- Focused mise benchmarks and source-backed scalar/SIMD recordings for CPU and YOLO workloads.
- On the measured Apple M1 Pro, 512-square MatMul improved from 183 ms to 32.7 ms;
  YOLO640 forward improved from 5.98 s to 1.28 s with 82.6% fewer allocated bytes.

## nn v0.3.0 and contrib/nn/yolo v0.1.0

Tensor and simdops remain at v0.1.0. Release nn/v0.3.0 before contrib/nn/yolo/v0.1.0.

- SiLU activation with stable evaluation and eager differentiation.
- NCHW MaxPool2D with negative-infinity padding and overlapping-window gradients.
- Upsample2D with nearest, bilinear, and bicubic modes, half-pixel interpolation, and input gradients.
- Checkpoint round trips for all new built-in spatial layers and SiLU.
- Independent YOLOv8n detection module with immutable model loading, RGB preprocessing, DFL decoding, and class-aware NMS.
- Locked offline conversion from trusted pretrained PyTorch checkpoints with BatchNorm folding and exact float32 kernel layout conversion.
- A pure-Go image-detection command and optional numerical comparison against pinned PyTorch outputs.
- Nested-module GitHub release automation, workspace tasks, and dedicated standalone release validation.
- Package and example READMEs throughout nn, plus conversion, inference, validation, and release guides.

## nn v0.2.0

Tensor and simdops remain at v0.1.0. Their APIs and nn's published upstream requirements are unchanged.

- Conv2D with NCHW layout, square kernels, stride, symmetric zero padding, and optional bias.
- Batch-first RNN and LSTM layers with explicit caller-owned recurrent states.
- Multi-head self-attention, causal masking, LayerNorm, sinusoidal positional encoding, and a transformer encoder block.
- Flatten and LastStep adapters for image and sequence classifiers.
- Stable axis-wise Softmax in nn/activation.
- Differentiable Slice, Concat, Contiguous, Scale, and a validated multi-input custom operation boundary.
- Seeded CNN, RNN, LSTM, and transformer training examples with convergence checks and mise tasks.
- A CNN option in the MNIST example, sharing local IDX loading and batching with the default dense model.
- Versioned, checksummed float32 weight files and complete built-in Sequential network serialization in nn/checkpoint.
- Architecture reconstruction preserving nested and shared layers without random weight initialization.
- MNIST checkpoint saving, loading, warm starts, and inference-only evaluation.
- Checkpoint corruption, metadata, round-trip, and state-preservation tests, plus file and architecture fuzz targets.
- Independent scalar references, weighted finite-difference layer checks, and advanced-layer benchmarks.
- Module-specific GitHub release-note automation for pushed tags and manual backfills.
- Standalone module checks in regular CI and updated published-module installation instructions.

## v0.1.0

Published as `simdops/v0.1.0`, `tensor/v0.1.0`, and `nn/v0.1.0`.

- Three independent modules managed through a Go workspace and mise.
- Float32 scalar kernels and optional portable experimental SIMD arithmetic.
- Immutable N-dimensional tensors, strided views, broadcasting, reductions,
  mathematical functions, and 2D matrix multiplication.
- Eager reverse-mode differentiation with consuming backward passes,
  gradient accumulation, parameter identity, and explicit inference mode.
- Dense and Sequential in `nn/layer`, parameterless layers in `nn/activation`,
  and SGD and Adam in `nn/optimizer`.
- Stable classification cross-entropy and mean squared error.
- Loss implementations in `nn/loss` with NewMSE, NewCrossEntropy, and a typed,
  single-method Loss interface.
- A validated single-input operation constructor for custom backward rules.
- Tensor basics, linear regression, XOR, and local-data MNIST examples.
- Paired, behavior-named tests, finite-difference checks, fuzz targets,
  benchmarks, CI tasks, and ownership documentation.
- Verified README examples, Mermaid diagrams, module guides,
  troubleshooting, and an ordered release procedure.

Each module retains its own release tags and versioned dependencies.
