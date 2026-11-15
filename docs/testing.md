# Testing and validation

[Documentation](README.md) · [Development](development.md)

goml combines exact contract tests, independent numerical references, gradient checks, and fuzzing.
This guide describes what each check establishes and where evidence is still incomplete.

## Run the checks

| Purpose | Command |
| --- | --- |
| Ordinary tests in all modules | `mise run test` |
| Focused module tests | `mise run test:nn` |
| Formatting, build, vet, lint, tests | `mise run check` |
| Full local CI gate | `mise run ci` |
| Race detector | `mise run test:race` |
| Experimental SIMD path | `mise run test:simd` |
| Package statement coverage | `mise run test:coverage` |
| Bounded shape and view fuzzing | `mise run test:fuzz` |
| Linux amd64/arm64 cross-compilation | `mise run build:portable` |
| Published module dependency resolution | `mise run test:standalone` |

Ordinary tests use local inputs and generated fixtures.
Initial tool installation and vulnerability analysis can require network access.
Standalone downstream tests require published upstream module versions.

## Slice and tensor references

Scalar kernel tests use explicit known results.
Public slice API tests cover validation before writes, exact aliasing, empty inputs, and allocation behavior.
SIMD tests compare vector kernels and scalar tails with their scalar references.

Tensor tests cover construction copies, metadata copies, strided views, reshape constraints, and broadcasting.
Reduction tests include empty groups and large empty shapes.
Matrix tests check both values and output shapes.

MatMul reference calculations read logical values through At and accumulate independently in float64.
The cases include transposed operands, irregular dimensions, and zero dimensions.
These small reference cases use an absolute tolerance of `1e-5`.

Special-value tests include NaN, infinity, signed zero, and invalid numerical domains.
Coverage percentages complement these checks but do not establish numerical correctness by themselves.

## Gradient checks

Centered finite differences compare analytical gradients with perturbed forward evaluations:

```text
reference[i] = (loss(x + epsilon*e[i]) - loss(x - epsilon*e[i])) / (2*epsilon)
```

The checks use `epsilon = 1e-3`.
The comparison tolerance is `0.003 + 0.015*abs(reference)`.
This accounts for float32 forward rounding in the tested cases.
Additional layer tests use weighted output seeds and tolerance `0.003 + 0.02*abs(reference)`.
They perturb every input element and every trainable parameter element.

| Area | Evidence |
| --- | --- |
| Binary arithmetic | Both operands, including broadcast gradients. |
| Matrix multiplication | Left and right derivatives against finite differences. |
| Reductions | Selected axes and multiple reduced axes. |
| Views and sequence assembly | Reshape, inverse transpose, slice scattering, concatenation, and contiguous materialization. |
| Activations | ReLU away from zero, Sigmoid, Tanh, and Softmax along different axes. |
| Losses | MSE and stable cross-entropy in nn/loss. |
| Custom operations | Output seeds, callback errors, exact input-gradient shapes, and inference behavior. |
| Conv2D | Input and kernel derivatives across batches, channels, stride, and zero padding. |
| RNN and LSTM | Recurrent weights, packed gates, initial states, and time-step dependencies. |
| Normalization and attention | LayerNorm affine parameters and all attention projections. |
| TransformerEncoder | Both residual paths and every trainable parameter. |

Separate exact tests cover shared dependencies, accumulation across passes,
consumed graphs, explicit seed shapes, and multiple singleton broadcast axes.
ReLU's derivative at zero is tested separately.

## Optimizer and training checks

SGD is compared with explicit expected updates.
Multiple Adam steps are compared with an independent float64 recurrence and bias correction.
Tests also cover missing gradients, parameter identity, and retained gradients.

The seeded XOR test exercises complete Dense/Tanh/Dense/Sigmoid training with Adam.
Its final predictions must be within 0.1 of the targets.
The linear example must match its target line within 0.001.

The additional CNN, LSTM, and transformer examples must classify every generated training sample correctly.
Their final mean cross-entropy must be below `0.03`.
The RNN example must approximate each sequence sum within `0.03`.
These tests use fixed seeds and do not download datasets.

The loss interface tests preserve distinct target types.
They also verify target gradients for MSE and saved class-label semantics for cross-entropy.

## Fuzzing

The bounded fuzz task runs three targets:

| Target | Properties |
| --- | --- |
| FuzzShapes | Negative dimensions and overflow, checked against arbitrary-precision multiplication. |
| FuzzBroadcast | Valid broadcast dimensions and incompatible shape pairs. |
| FuzzViews | Transpose round trips, contiguous copies, reshape preservation, indexing, and slice bounds. |

Each target runs for three seconds with two workers.
Input sizes bound tensor allocations.
Go can preserve interesting inputs in its fuzz cache.

A recorded pass executed 248,283 layout cases, 238,722 broadcast cases, and 189,860 view cases.
Short fuzz runs provide regression evidence.
They do not prove that every shape or layout is correct.

## Recorded verification

Local checks use Go 1.27.3 on macOS arm64 with an Apple M1 Pro.

| Check | Recorded result |
| --- | --- |
| Format, build, vet, lint, unit tests | Passed. |
| Race detector | Passed. |
| Experimental SIMD suite | Passed across all modules. |
| Shape and view fuzzing | Passed the bounded targets. |
| Linux amd64/arm64, CGo disabled | Cross-compiled with and without SIMD. |
| Standalone modules | All three pass with workspace mode disabled and published upstream v0.1.0 dependencies. |
| Linear regression and XOR | Passed their convergence checks. |
| MNIST fixtures | Loader checks, dense and CNN training, partial batches, parameter gradients, and seeded inference passed. |
| Additional model examples | CNN, RNN, LSTM, and transformer pass their convergence checks. |

Cross-compilation checks build compatibility.
It does not execute a binary on the target CPU.

govulncheck 1.9.0 reported zero reachable vulnerabilities.
The recorded report also identified one module-level vulnerability that the code did not call.
That report is a point-in-time database result.
Nancy was not run because its external service access is optional.

## Remaining release evidence

| Item | Status |
| --- | --- |
| Native amd64 tests and benchmarks | Requires a native runner or target machine. |
| Hosted GitHub workflows | Configured locally, not executed on GitHub yet. |
| Future nn release publication | Additional layers are tested locally and remain unreleased. |
| Real MNIST dataset training | Not measured in this checkout. |
| Held-out MNIST accuracy | Not measured. |

Update this evidence when validating a release candidate.
Do not describe configured workflows or synthetic fixtures as completed deployment or dataset validation.
