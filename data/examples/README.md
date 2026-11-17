# Data examples

[Data module](../README.md) · [MNIST application](../../nn/examples/mnist/README.md)

These programs use only data and Tensor. Model training belongs to the separate
MNIST application module, which can combine data with nn and opt-in Metal.

| Program | Purpose |
| --- | --- |
| [classification](classification/README.md) | Tensor snapshots, shuffled batches, partial final batch, and cleanup. |
| [custom](custom/README.md) | Application-owned sample types, variable token lengths, and a custom collator. |

```sh
mise run example:data
mise run test:data
mise run example:data:mnist -- -data archive
```

No datasets are downloaded. The first two examples use generated in-memory data.
Each program has its own paired main_test.go and compiles with the data module.

Licensed under [MIT](../../LICENSE).
