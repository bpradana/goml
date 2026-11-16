# Documentation

goml provides three core Go modules and an independent YOLO inference module.
The [project README](../README.md) introduces the APIs and provides the local quick start.

## Choose a guide

| Task | Guide |
| --- | --- |
| Use allocation-free slice operations | [simdops](../simdops/README.md) |
| Work with numerical arrays and views | [tensor](../tensor/README.md) |
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

It prints GoDoc for the four modules and nn's activation, checkpoint, layer, loss, and optimizer packages.
Exported function comments define input constraints, output shapes, errors, and ownership.

The complete README programs were compiled and executed during local validation.

## Status and evidence

The release set is simdops v0.1.0, tensor v0.1.0, nn v0.3.0, and contrib/nn/yolo v0.1.0.
The three core modules resolve published upstream dependencies.
YOLO standalone checks require publication of nn v0.3.0 and subsequent checksum generation.
Workspace tests resolve local changes through `go.work`.

The validation and performance guides distinguish completed local checks from
native CI runs, external installation, and real-dataset measurements.
Local checks do not establish hosted workflow execution or real-dataset accuracy.
