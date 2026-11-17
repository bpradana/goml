# Contributing

[Project overview](README.md) · [Development guide](docs/development.md)

Contributions should preserve the small dependency chain and explicit ownership rules.
Numerical behavior and maintainable Go APIs take priority over speculative abstractions.

## Prepare the environment

Install mise, then run these commands from the repository root:

```sh
mise install --locked
mise run test
mise run check
```

If personal global tools interfere, use the
[project-only installation command](docs/development.md#install-the-project-tools).

## Make a focused change

1. Read the relevant package comments and ownership contract.
2. Add or update behavior-specific tests.
3. Implement the smallest change that satisfies the contract.
4. Run the module formatter, tests, and lint.
5. Run the broader checks required by the changed behavior.
6. Update affected API comments and guides.

For example:

```sh
mise run fmt:tensor
mise run test:tensor
mise run lint:tensor
mise run check
```

Use race checks for shared state and graph-lifecycle changes.
Use SIMD checks and benchmarks for kernel changes.
Use fuzz checks for shapes, indexing, slicing, reshape, and broadcasting.

## Keep module boundaries

| Module | Responsibility |
| --- | --- |
| simdops | Validated slice APIs and numerical kernels. |
| tensor | Immutable arrays, layouts, views, and mathematics. |
| nn | Differentiation, parameters, model composition, losses, and updates. |

nn does not import simdops directly.
Tensor does not import nn.
The root workspace supplies local module resolution.

Do not add a root go.mod or Makefile. Keep CPU modules pure Go.
Metal implements tensor-owned device contracts in its independent contrib module.
Keep CGo and Apple frameworks inside that module. CUDA and ROCm remain unimplemented.
Subpackages remain part of their parent module.
Do not add relative replace directives to released manifests.

## Write clear Go

Use concrete types and direct control flow.
Introduce an interface when a current composition or substitution need justifies it.

Use the formatter for indentation and whitespace.
Separate validation, computation, and result handling.
Return errors for invalid public inputs.
Do not silently substitute zero values for failed operations.

Keep storage, graph internals, optimized kernels, and optimizer state private.
Document any new ownership or lifetime boundary.
Use caller-supplied random generators for reproducible initialization.

## Pair implementation and tests

Pair files such as `adam.go` and `adam_test.go`.
Documentation-only and interface-only files do not need artificial tests.
Shared test helpers can stay in test files.

Name tests by operation and behavior:

| Clear name | What it identifies |
| --- | --- |
| `TestDot_EmptyInput` | The operation and empty-input contract. |
| `TestMax_PropagatesNaN` | The special-value rule. |
| `TestBackward_AccumulatesSharedDependencies` | The graph behavior. |
| `TestAdam_MultipleStepsMatchIndependentReference` | The optimizer calculation. |

Use descriptive table subtests for meaningful input classes.
Do not combine unrelated behaviors under a name such as TestEverything.

Numerical tests need independent references where practical.
Gradient tests use finite differences.
Cover invalid dimensions, empty arrays, views, broadcasting, and relevant special values.
See [validation](docs/testing.md) for existing methods and tolerances.

## Benchmark before optimizing

Keep a scalar reference for numerical kernels.
Measure representative sizes and irregular tails.
Report the CPU, Go version, build mode, sample count, time, and allocations.

Do not infer a regression from one noisy sample.
Do not add pooling, goroutines, packing, or a backend interface without a measured need.
See the [performance guide](docs/performance.md).

## Update documentation

Exported comments should explain calculation, shapes, errors, allocation, and sharing.
Document numerical edge cases when they affect callers.

README examples should use current APIs and handle errors.
Link substantial programs to compiled example sources.
Keep measured results distinct from expectations and future work.

Use Mermaid for every diagram.
Use tables for source-file catalogs and numerical comparisons.
Do not add ASCII diagrams or claim hosted CI or releases that do not exist.

## Commit and review

Use Conventional Commit subjects with a focused scope, for example:

```text
feat(tensor): add a numerical operation
fix(autograd): preserve shared gradient contributions
docs: clarify tensor ownership
```

Keep related implementation, tests, and API comments together.
Separate unrelated functionality into distinct commits.
A review should explain the resulting behavior, verification, and any remaining limitations.

The normal local gate is `mise run check`.
The CI gate adds race testing and vulnerability analysis.
Standalone dependency checks become mandatory once upstream releases exist.

Publication is a separate maintainer operation.
Follow the [release guide](docs/releasing.md) in dependency order.

## Backend contributions

Start with the [backend contract](docs/backend-contract.md) and
[implementation tutorial](docs/extending-backends.md). Implement Device and only
needed capabilities; do not add a registry or change unrelated Tensor or NeuralNet
code. A complete MatMul-only example lives in `tensor/examples/backend`.

Keep tests paired with their source: `array.go` and `array_test.go`, `matmul.go` and
`matmul_test.go`. Use names that identify behavior and detailed subtests. Run the
shared Device suite and every capability suite your backend claims. Mock-generated
test doubles may exercise focused failure paths; numerical conformance executes
real implementations and cannot be replaced by mocks.

Use `mise run test:backend` for portable contracts, CPU, and contributor examples.
Use `mise run test:backend:metal` and the GPU parity/race tasks on supported Apple
hardware. CUDA and ROCm remain documentation-only source reservations outside the
active workspace. Raw benchmark outputs stay outside the repository.

## Dataset contributions

Dataset contributors can start with the [data contracts and extension guide](docs/datasets.md).
The [custom dataset example](data/examples/custom/README.md) uses only public interfaces;
run `mise run test:data:race` and `mise run test:standalone:data` for scoped validation.

## Metrics and training changes

Metrics is standard-library-only. New accumulators implement the generic
Update/Reset boundary and may expose their own typed Result; test streaming
partitions, invalid-update atomicity, and independent result snapshots. The
[custom example](metrics/examples/custom/README.md) shows an external implementation.

Trainer composes public data, nn, and Tensor APIs. Keep concrete backend imports
in optional applications/tests, close each yielded batch and per-batch scope,
and preserve caller ownership of models, optimizers, devices, and metric instances.
Run `mise run test:trainer:race` and `mise run test:standalone:trainer`; optional
hardware tasks validate real Metal/WebGPU execution. Each implementation file
has a matching test file. See the [training guide](docs/training.md).
