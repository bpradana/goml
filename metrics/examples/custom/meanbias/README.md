# Mean-bias example implementation

[Custom metric example](../README.md) · [Metrics contracts](../../../../docs/metrics.md)

Metric implements `metrics.Metric[float32,float32,Result]` using a running
Float64 mean of prediction minus target. It validates entire batches before
mutation, rejects nonfinite values, retains no input slices, and has constant
memory. Results are plain typed values. Reset permits reuse.

This package illustrates the extension contract rather than adding another
built-in metric. The Trainer regression example imports it directly and passes
its instance to NewRegressor. No global registry, Tensor dependency, or adapter
specific to Trainer is required.

Licensed under [MIT](../../../../LICENSE).
