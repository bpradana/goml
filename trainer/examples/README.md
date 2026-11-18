# Trainer examples

[Trainer](../README.md) · [Training guide](../../docs/training.md)

| Program | Demonstrates |
| --- | --- |
| [Classification](classification/README.md) | Seeded toy classifier, validation metrics, and callbacks. |
| [Regression](regression/README.md) | Adam, linear regression, and a pluggable custom metric. |
| [Objective](objective/README.md) | Replace a built-in loss with a composite denoising objective. |
| [Custom](custom/README.md) | A typed reconstruction step with a custom objective. |
| [MNIST](../../nn/examples/mnist/README.md) | Dense/CNN training and checkpoint workflows with explicit CPU or GPU selection. |

Run the generated-data programs with `mise run example:trainer`. Their paired
main_test.go files execute the same public API paths without external datasets.
MNIST uses local files only and has optional hardware-specific build tags.

Licensed under [MIT](../../LICENSE).
