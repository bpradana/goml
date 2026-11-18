package trainer_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/loss"
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

type regressionLossFunc func(*nn.Value, *nn.Value) (*nn.Value, error)

func (f regressionLossFunc) Forward(prediction, target *nn.Value) (*nn.Value, error) {
	return f(prediction, target)
}

func TestRegressor_ConfiguredLossAndDefaultCompatibility(t *testing.T) {
	model := denseModel(t, 1, 1)
	opt, _ := optimizer.NewSGD(model.Parameters(), 0.01)
	loader := regressionLoader(t)
	standard, _ := trainer.NewRegressor(model, opt, tensor.CPU())

	defaults, err := trainer.NewRegressorWithConfig(model, opt, tensor.CPU(), trainer.RegressorConfig{})
	if err != nil {
		t.Fatal(err)
	}

	before, err := standard.Evaluate(context.Background(), loader)
	if err != nil {
		t.Fatal(err)
	}

	same, err := defaults.Evaluate(context.Background(), loader)
	if err != nil || before.MeanLoss != same.MeanLoss {
		t.Fatal(before, same, err)
	}

	calls, count := 0, 0
	custom := regressionLossFunc(func(prediction, target *nn.Value) (*nn.Value, error) {
		calls++

		count += target.Data().Shape()[0]
		if !slices.Equal(prediction.Data().Shape(), target.Data().Shape()) {
			t.Fatal("incorrect target shape")
		}

		value, err := loss.NewMSE().Forward(prediction, target)
		if err != nil {
			return nil, err
		}

		return nn.Scale(value, 0.5)
	})

	runner, err := trainer.NewRegressorWithConfig(model, opt, tensor.CPU(), trainer.RegressorConfig{Loss: custom})
	if err != nil {
		t.Fatal(err)
	}

	history, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 1})
	if err != nil || calls != 4 || count != 8 {
		t.Fatal(history, err, calls, count)
	}

	validation := history.Epochs[0]
	if math.Abs(validation.Validation.MeanLoss-0.5*validation.Regression.MSE) > 1e-5 {
		t.Fatal("validation substituted MSE", validation)
	}
}

func TestRegressor_CustomLossFailureClosesBatchAndTargets(t *testing.T) {
	model := denseModel(t, 1, 1)
	sgd, _ := optimizer.NewSGD(model.Parameters(), 0.01)
	opt := &countingOptimizer{Optimizer: sgd}
	wanted := errors.New("regression loss")

	var input, target *tensor.Tensor

	custom := regressionLossFunc(func(prediction, expected *nn.Value) (*nn.Value, error) {
		input, target = prediction.Data(), expected.Data()
		return nil, wanted
	})

	runner, err := trainer.NewRegressorWithConfig(model, opt, tensor.CPU(), trainer.RegressorConfig{Loss: custom})
	if err != nil {
		t.Fatal(err)
	}

	_, err = runner.Fit(context.Background(), regressionLoader(t), nil, trainer.FitOptions{Epochs: 1})
	if !errors.Is(err, wanted) || opt.steps != 0 || input == nil || target == nil || !input.IsClosed() || !target.IsClosed() {
		t.Fatal(err, opt.steps)
	}
}
