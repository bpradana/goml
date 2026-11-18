# Streaming regression metrics

[Examples](../README.md) · [Metrics](../../README.md)

Run `mise run example:metrics:regression` from the repository root.
The program accumulates two batches, then prints an independent numerical report.
It needs no datasets, Tensor library, GPU, or external service.


The program also serializes a single-observation result through encoding/json.
Undefined R² appears as null with R2Valid false; its Go numerical value stays NaN.

Licensed under [MIT](../../../LICENSE).
