# Classification training

[Trainer](../../README.md) · [Source](main.go)

```sh
mise run example:trainer:classification
```

Four generated points train a seeded Dense classifier through data.Loader and
Trainer. Batch size three exercises the partial final batch. Twenty epochs use
SGD, validation, and an application-owned progress callback. The program checks
convergence; no data downloads or native device dependencies are required.
Training and validation share this tiny dataset solely for an API demonstration.
Use an independent validation split for real experiments.

For real local data, see the [MNIST application](../../../nn/examples/mnist/README.md).

Licensed under [MIT](../../../LICENSE).
