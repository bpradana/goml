# Metrics contracts

[Metrics module](../metrics/README.md)

metrics is standard-library-only. It consumes ordinary slices and holds only
integer counts or Float64 running statistics. Accumulators do not own tensors,
models, files, datasets, or device contexts.

## Extend with a typed metric

```go
type Accumulator[P, T any] interface {
	Update(predictions []P, targets []T) error
	Reset()
}

type Metric[P, T, R any] interface {
	Accumulator[P, T]
	Result() (R, error)
}
```

P and T are slice element types; R is an application-defined report. Built-ins
satisfy `Metric[int,int,ClassificationResult]` and
`Metric[float32,float32,RegressionResult]`. There is no registration, result map,
reflection, or dependency on Trainer. The update/reset boundary lets Trainer
consume metrics without erasing the concrete Result type owned by the application.

Classification.Logits() adapts the same classification state to
`Metric[float32,int,ClassificationResult]`. Its Update accepts flattened
`[batch,classes]` raw scores and delegates to UpdateLogits. Result and Reset
still apply to the original accumulator. This is the classification metric
adapter accepted by Trainer. Raw-score custom metrics can implement the same
interface directly for tasks such as confidence summaries or Top-K evaluation.

A custom Update should validate lengths and its entire batch before committing
state, avoid mutating/retaining borrowed slices, and document undefined results.
Reset clears observations without invalidating independent reports. The
[external implementation test](../metrics/metric_test.go) demonstrates a custom
mean-bias metric with its own concrete result struct.

## Classification conventions

Confusion entries use `Confusion[actual][predicted]`. Per-class support is actual
count. Precision is TP/predicted support, recall TP/actual support, and F1 the
harmonic mean. Undefined class precision/recall/F1 are zero. Macro averages
include every configured class; weighted averages use actual support. In this
single-label multiclass setting, micro precision/recall/F1 equal accuracy.

UpdateLogits requires finite Float32 values and exact shape. ArgMax ties select
the lowest index. No softmax is computed. Invalid updates change no counters.
Confusion counters are uint64 with overflow rejection; floating ratios may round
counts larger than Float64's exact integer range. Memory is O(classes²).

## Regression conventions

Each flattened scalar observation has equal weight. Absolute and squared errors
use compensated Float64 sums; target variance uses Welford updates. Prediction
minus target is calculated after conversion to Float64, avoiding Float32 error
squaring overflow and small-error underflow.

R² is `1 - squared_error / target_m2`. With fewer than two observations or zero
target variance, R2 is NaN and R2Valid is false. Negative R² is valid. Empty
Result returns an error for both families. No arbitrary successful zero report
is substituted for an undefined dataset evaluation.

## Lifecycle and concurrency

Result does not modify state and copies mutable classification report slices.
Reset leaves prior results independent. Updates retain no sample arrays and
allocate no per-observation storage. Callers synchronize concurrent mutation;
the initial implementation intentionally has no internal locks or goroutines.

See [measured performance](performance/metrics.md) for workloads and limitations.
