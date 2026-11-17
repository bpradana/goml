package trainer_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/metrics"
	"github.com/bpradana/goml/metrics/examples/custom/meanbias"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

type identityModel struct{}

func (identityModel) Forward(v *nn.Value) (*nn.Value, error) { return v, nil }

func (identityModel) Predict(t *tensor.Tensor) (*tensor.Tensor, error) { return t.Retain() }

func (identityModel) Parameters() []*nn.Parameter { return nil }

func TestEvaluation_WeightedPartialBatchesAndIndependentMetrics(t *testing.T) {
	x, _ := tensor.From([]float32{1, 0, 1, 0, 0, 3}, 3, 2)
	defer func() { _ = x.Close() }()

	ds, _ := data.NewClassificationDataset(x, []int{0, 1, 0})
	loader, _ := data.NewLoader(ds, data.CollateClassification, data.LoaderOptions{BatchSize: 2})
	report, err := trainer.EvaluateClassification(context.Background(), identityModel{}, tensor.CPU(), 2, loader)

	want := (math.Log1p(math.Exp(-1)) + math.Log1p(math.Exp(1)) + math.Log1p(math.Exp(3))) / 3
	if err != nil || report.Samples != 3 || report.Batches != 2 || report.Classification.Accuracy != 1.0/3 || math.Abs(report.MeanLoss-want) > 2e-7 {
		t.Fatal(report, err, want)
	}

	inputs, _ := tensor.From([]float32{1, 2, 5}, 3, 1)
	defer func() { _ = inputs.Close() }()

	targets, _ := tensor.From([]float32{0, 0, 1}, 3, 1)
	defer func() { _ = targets.Close() }()

	reg, _ := data.NewRegressionDataset(inputs, targets)
	regLoader, _ := data.NewLoader(reg, data.CollateRegression, data.LoaderOptions{BatchSize: 2})

	result, err := trainer.EvaluateRegression(context.Background(), identityModel{}, tensor.CPU(), regLoader)
	if err != nil || result.MeanLoss != 7 || result.Regression.MSE != 7 || math.Abs(result.Regression.R2+30.5) > 1e-12 {
		t.Fatal(result, err)
	}
}

func TestEvaluation_CustomTypedRegressionMetricResetsAndUsesEveryBatch(t *testing.T) {
	input, err := tensor.From([]float32{2, 4, 2.5}, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = input.Close() }()

	target, err := tensor.From([]float32{1, 3, 3}, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = target.Close() }()

	dataset, err := data.NewRegressionDataset(input, target)
	if err != nil {
		t.Fatal(err)
	}

	loader, err := data.NewLoader(dataset, data.CollateRegression, data.LoaderOptions{BatchSize: 2})
	if err != nil {
		t.Fatal(err)
	}

	bias := meanbias.New()
	for range 2 {
		if _, err := trainer.EvaluateRegression(context.Background(), identityModel{}, tensor.CPU(), loader, bias); err != nil {
			t.Fatal(err)
		}

		result, err := bias.Result()
		if err != nil || result.Count != 3 || result.MeanBias != 0.5 {
			t.Fatalf("custom result: %+v, %v", result, err)
		}
	}
}

type failingClassificationMetric struct {
	failure error
	resets  int
}

func (m *failingClassificationMetric) Update([]float32, []int) error {
	return m.failure
}

func (m *failingClassificationMetric) Reset() {
	m.resets++
}

func TestEvaluation_CustomMetricFailureClosesBatch(t *testing.T) {
	wanted := errors.New("custom metric failed")
	metric := &failingClassificationMetric{failure: wanted}
	source := &observedClasses{source: classLoader(t, 3)}
	model := denseModel(t, 1, 2)

	_, err := trainer.EvaluateClassification(context.Background(), model, tensor.CPU(), 2, source, metric)
	if !errors.Is(err, wanted) || metric.resets != 1 {
		t.Fatal(err, metric.resets)
	}

	if len(source.batches) != 1 || !source.batches[0].Inputs.IsClosed() {
		t.Fatal("custom metric failure left the batch open")
	}
}

func TestEvaluation_EmptyValidationIsAnError(t *testing.T) {
	x, _ := tensor.Zeros(0, 2)
	defer func() { _ = x.Close() }()

	ds, _ := data.NewClassificationDataset(x, nil)

	loader, _ := data.NewLoader(ds, data.CollateClassification, data.LoaderOptions{BatchSize: 1})
	if _, err := trainer.EvaluateClassification(context.Background(), identityModel{}, tensor.CPU(), 2, loader); err == nil {
		t.Fatal("empty evaluation succeeded")
	}
}

func TestEvaluation_PluggableMetricsShareScoresAndResetPerEvaluation(t *testing.T) {
	input, _ := tensor.From([]float32{1, 0, 0, 2}, 2, 2)
	defer func() { _ = input.Close() }()

	dataset, _ := data.NewClassificationDataset(input, []int{0, 1})
	loader, _ := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: 1})

	extra, _ := metrics.NewClassification(2)
	for range 2 {
		report, err := trainer.EvaluateClassification(context.Background(), identityModel{}, tensor.CPU(), 2, loader, extra.Logits())
		if err != nil {
			t.Fatal(err)
		}

		custom, err := extra.Result()
		if err != nil || custom.Count != 2 || custom.Accuracy != report.Classification.Accuracy {
			t.Fatal(custom, err)
		}
	}
}
