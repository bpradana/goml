# Custom samples and collation

[Examples](../README.md) · [Contributor guide](../../../docs/datasets.md)

```sh
mise run example:data:custom
```

The program defines its own corpus Dataset, sentence sample, and batch type.
Each sentence has text and variable-length token IDs. Its collator copies the
retained token slices without padding, reflection, or a global registration step.
The batch contains ordinary Go slices and does not need a Close method.

For file-backed custom datasets, check context during I/O and return useful
index/read errors. Keep Len stable during an iteration. Custom samples must
remain valid while the collator reads them; do not reuse one mutable buffer for
all Get results in a batch. If a custom batch owns resources, release partial
allocations on collation errors and close successful batches in the application.

Licensed under [MIT](../../../LICENSE).
