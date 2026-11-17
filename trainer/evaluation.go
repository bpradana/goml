package trainer

import (
	"context"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/metrics"
	"github.com/bpradana/goml/tensor/backend"
)

// EvaluateClassification evaluates logits without constructing an optimizer or
// calling Fit. It borrows the model/device and closes every yielded batch.
func EvaluateClassification(ctx context.Context, model Model, device backend.Device, classes int, source BatchSource[*data.ClassificationBatch], extraMetrics ...metrics.Accumulator[float32, int]) (Evaluation, error) {
	if err := validateModel(model, device); err != nil {
		return Evaluation{}, err
	}

	runner := Classifier{model: model, device: device, classes: classes, metrics: extraMetrics}

	return runner.evaluate(ctx, source, 0)
}

// EvaluateRegression evaluates exact-shape outputs without an optimizer or Fit.
// It returns an error for an empty source and independent streaming metric state.
func EvaluateRegression(ctx context.Context, model Model, device backend.Device, source BatchSource[*data.RegressionBatch], extraMetrics ...metrics.Accumulator[float32, float32]) (Evaluation, error) {
	if err := validateModel(model, device); err != nil {
		return Evaluation{}, err
	}

	runner := Regressor{model: model, device: device, metrics: extraMetrics}

	return runner.evaluate(ctx, source, 0)
}
