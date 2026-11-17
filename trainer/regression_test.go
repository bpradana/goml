package trainer_test

import (
	"context"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

func regressionLoader(t testing.TB) *data.Loader[data.RegressionSample, *data.RegressionBatch] {
	t.Helper()

	inputs, err := tensor.From([]float32{-1, 0, 1, 2}, 4, 1)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = inputs.Close() }()

	targets, err := tensor.From([]float32{-1, 1, 3, 5}, 4, 1)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = targets.Close() }()

	dataset, err := data.NewRegressionDataset(inputs, targets)
	if err != nil {
		t.Fatal(err)
	}

	loader, err := data.NewLoader(dataset, data.CollateRegression, data.LoaderOptions{BatchSize: 3})
	if err != nil {
		t.Fatal(err)
	}

	return loader
}

func TestRegressor_LinearTrainingAndStandaloneEvaluation(t *testing.T) {
	model := denseModel(t, 1, 1)

	opt, err := optimizer.NewAdam(model.Parameters(), 0.05, 0.9, 0.999, 1e-8)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = opt.Close() }()

	runner, err := trainer.NewRegressor(model, opt, tensor.CPU())
	if err != nil {
		t.Fatal(err)
	}

	loader := regressionLoader(t)

	before, err := runner.Evaluate(context.Background(), loader)
	if err != nil {
		t.Fatal(err)
	}

	history, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 100})
	if err != nil {
		t.Fatal(err)
	}

	last := history.Epochs[99]
	if last.Validation.MeanLoss >= before.MeanLoss*0.001 || last.Regression.MSE > 0.001 || !last.Regression.R2Valid || last.Regression.R2 < 0.999 {
		t.Fatalf("failed regression convergence: %+v %+v", last, last.Regression)
	}

	evaluation, err := trainer.EvaluateRegression(context.Background(), model, tensor.CPU(), loader)
	if err != nil || evaluation.Samples != 4 || evaluation.Batches != 2 || evaluation.Regression.Count != 4 {
		t.Fatal(evaluation, err)
	}
}
