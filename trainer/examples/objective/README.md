# Denoising with a configured objective

[Trainer](../../README.md) · [Source](main.go) · [Objective](objective.go)

```sh
mise run example:trainer:objective
mise run example:trainer:objective -- -best-weights archive/denoising.goml
```

A seeded two-output Dense model maps perturbed points to clean targets. Its loss
combines mean reconstruction error and 0.001 times mean squared output energy.
NewRegressorWithConfig owns the ordinary batching, transfers, backward pass,
optimizer update, gradient clearing, and per-batch cleanup. Both terms contribute
to gradients; the paired objective test checks their independent scalar reference.

The four generated points demonstrate an API, not generalization to real images.
The same tiny set is used for training and validation. Validation.MeanLoss is the
configured composite objective; Regression.MSE remains reconstruction error.

The optional explicit output path configures SaveBest to use the existing
weights-only checkpoint format, overriding Sequential's architecture serializer.
The callback owns serialization; no output is written unless requested.

## Why no new lifecycle helper

The older [generic reconstruction example](../custom/README.md) accepts a custom
Tensor batch and owns a complete StepFunc. This version uses the existing
RegressionBatch and replaces only the loss. It needs one forward pass, two loss
terms, and ordinary gradients. Trainer already owns all repeated resource work,
so another scope/step interface would add no benefit here.

Keep generic Fit for custom batches, paired/multiple forwards, or specialized
gradient behavior. Such steps continue to own their own batch and scope. Model
still has Forward, Predict, and Parameters: this example works with Sequential
without changing or splitting that interface.

Licensed under [MIT](../../../LICENSE).
