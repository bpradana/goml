# Recurrent sequence regression

[All examples](../README.md) · [nn](../../README.md) · [Source](main.go)

Learn the sum of each three-number sequence from its final hidden state.
The example uses a fixed random seed, local data, and the CPU-only nn runtime.
It does not download a dataset or require CGo, Python, or a GPU.

## Run

From the repository root, after `mise install --locked`:

```sh
mise run example:rnn
```

The program prints its training result and returns a nonzero exit status if the convergence check fails.
Predictions must be within 0.03 of each target sum.

## Model and training

The model path is: RNN → LastStep → Dense → mean squared error → Adam.
Each iteration performs a fresh forward pass, computes a scalar loss, calls Backward, steps the optimizer, and clears gradients.
Final predictions use Sequential.Predict to disable gradient recording.
The source includes complete error handling and the chosen initialization and optimizer settings.

This generated dataset demonstrates API behavior and training convergence.
It does not establish held-out accuracy or production performance.

## Verify

```sh
mise run test:nn
```

The paired main_test.go executes this example's convergence check during normal module tests.
See [checkpoints](../../checkpoint/README.md) to save a model's weights or complete built-in architecture.
