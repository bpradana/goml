# Documentation

goml provides independently versioned numerical, neural-network, dataset, and inference modules.
The [project README](../README.md) introduces the APIs and provides the local quick start.

## Choose a guide

| Task | Guide |
| --- | --- |
| Use allocation-free slice operations | [simdops](../simdops/README.md) |
| Load datasets, shuffle, collate, and preprocess samples | [data](../data/README.md) and [contributor guide](datasets.md) |
| Work with numerical arrays and views | [tensor](../tensor/README.md) |
| Run tensor operations and models on Apple Silicon | [Metal](../contrib/backend/metal/README.md) and [GPU contracts](gpu.md) |
| Run WGSL kernels and models through native WebGPU | [WebGPU](../contrib/backend/webgpu/README.md) and [validation report](performance/webgpu.md) |
| Implement an independent execution backend | [Backend contract](backend-contract.md) and [contributor tutorial](extending-backends.md) |
| Migrate development device APIs | [Backend migration](backend-migration.md) |
| Review backend migration measurements | [Backend refactor validation](performance/backend-refactor.md) |
| Inspect GPU parity and measured workloads | [Metal performance](performance/metal.md) |
| Run pretrained YOLOv8n detection | [YOLO](../contrib/nn/yolo/README.md) |
| Inspect pretrained numerical comparisons | [YOLO validation](yolo-validation.md) |
| Build and train a neural network | [nn](../nn/README.md) |
| Use convolution, recurrent state, or transformer blocks | [Layers](layers.md) |
| Run generated-data training programs | [Neural network examples](../nn/examples/README.md) |
| Save weights or restore a network | [Checkpoints](../nn/checkpoint/README.md) |
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

It prints GoDoc for the library modules and nn's activation, checkpoint, layer, loss, and optimizer packages.
Exported function comments define input constraints, output shapes, errors, and ownership.

The complete README programs were compiled and executed during local validation.

## Status and evidence

The release set is simdops v0.1.0, tensor v0.3.0, nn v0.5.0, and contrib/nn/yolo v0.3.0.
nn resolves tensor v0.3.0; YOLO resolves nn v0.5.0 and tensor v0.3.0.
Workspace tests resolve local changes through `go.work`.

The validation and performance guides distinguish completed local checks from
native CI runs, external installation, and real-dataset measurements.
Local checks do not establish hosted workflow execution or real-dataset accuracy.
