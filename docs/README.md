# Documentation

goml provides three independently usable Go modules.
The [project README](../README.md) introduces the APIs and provides the local quick start.

## Choose a guide

| Task | Guide |
| --- | --- |
| Use allocation-free slice operations | [simdops](../simdops/README.md) |
| Work with numerical arrays and views | [tensor](../tensor/README.md) |
| Build and train a dense network | [nn](../nn/README.md) |
| Understand module and package boundaries | [Architecture](architecture.md) |
| Check shape, memory, or numerical behavior | [Tensor semantics](tensor-semantics.md) |
| Understand backward, accumulation, or inference | [Autograd](autograd.md) |
| Install tools or choose a development command | [Development](development.md) |
| Add a change and its tests | [Contributing](../CONTRIBUTING.md) |
| Check correctness evidence and its limits | [Validation](testing.md) |
| Reproduce or interpret a benchmark | [Performance](performance.md) |
| Prepare independent module releases | [Releasing](releasing.md) |
| Train with local MNIST files | [MNIST example](../nn/examples/mnist/README.md) |

## Read the current API

Run this command from the repository root:

```sh
mise run docs:api
```

It prints GoDoc for the three modules and nn's activation, layer, loss, and optimizer packages.
Exported function comments define input constraints, output shapes, errors, and ownership.

The complete README programs were compiled and executed during local validation.

## Status and evidence

v0.1.0 is under development. Workspace tests resolve local modules through
`go.work`. Published dependency resolution requires the upstream module tags.

The validation and performance guides distinguish completed local checks from
native CI runs, external installation, and real-dataset measurements.
No hosted CI status or published release is claimed by these documents.
