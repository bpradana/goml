# Custom typed training steps

[Trainer](../../README.md) · [Source](main.go)

```sh
mise run example:trainer:custom
```

A data.Loader collates two-dimensional points into Tensor batches. Generic
trainer.Fit receives typed StepFunc functions for regularized reconstruction:
MSE plus 0.001 times the mean squared output. The example owns its custom loss,
backward pass, optimizer update, gradients, Tensor scope, and batch cleanup.
Trainer supplies epoch iteration, validation statistics, and history.

This path supports application-specific tasks without changing Dataset,
DataLoader, Tensor, NeuralNet, or a global task registry. Custom evaluation metrics
can remain in application-owned typed accumulators and be read in OnEpochEnd.


If your task only changes a loss, prefer the
[configured objective example](../objective/README.md). It reuses the built-in
batch lifecycle; generic Fit remains for tasks that need arbitrary batches or
custom forwards/gradients. Generic best-checkpoint saving uses SaveBest alongside
BestCheckpointPath and a validation source.

Licensed under [MIT](../../../LICENSE).
