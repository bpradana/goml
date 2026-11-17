# Metrics examples

[Metrics](../README.md)

| Program | Purpose |
| --- | --- |
| [classification](classification/README.md) | Stream class IDs/logits into accuracy, precision/recall/F1, and confusion counts. |
| [custom](custom/README.md) | Implement a typed metric and plug the same package into Trainer. |
| [regression](regression/README.md) | Stream numerical observations into MAE/MSE/RMSE/R². |

Run `mise run example:metrics`. All programs use generated slices and standard
library dependencies only. Each has a paired main_test.go.

Licensed under [MIT](../../LICENSE).
