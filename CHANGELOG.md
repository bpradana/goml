# Changelog

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
