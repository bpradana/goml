# Classification batches

[Examples](../README.md) · [Data](../../README.md)

```sh
mise run example:data:classification
```

The program copies a `[4,2]` Tensor and integer labels into a dataset, then
iterates one seeded shuffled epoch with batch size three. It prints `[3,2]` and
`[1,2]` input shapes and closes each batch. The source Tensor has an independent
lifetime. No model, accelerator, file, or network dependency is needed.

Licensed under [MIT](../../../LICENSE).
