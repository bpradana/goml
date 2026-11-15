# Changelog

## Unreleased — v0.1.0

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

The module tags and external dependency checks are release gates.
This checkout does not imply that v0.1.0 has been published.
