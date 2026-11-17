package trainer_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

type scriptedOptimizer struct {
	model *layer.Sequential
	step  int
}

func (o *scriptedOptimizer) ZeroGrad() {
	for _, p := range o.model.Parameters() {
		p.ZeroGrad()
	}
}

func (o *scriptedOptimizer) Step() error {
	value := []float32{1, 3, 2}[o.step]

	o.step++
	for index, p := range o.model.Parameters() {
		if index != 0 {
			value = 0
		}

		next, err := tensor.FromLike(p.Value().Data(), []float32{value}, p.Value().Data().Shape()...)
		if err != nil {
			return err
		}

		err = p.Update(next)
		_ = next.Close()

		if err != nil {
			return err
		}
	}

	return nil
}

func TestCheckpoint_BestValidationModelIsSavedNotFinalModel(t *testing.T) {
	model := denseModel(t, 1, 1)

	value, _ := tensor.Ones(1, 1)
	defer func() { _ = value.Close() }()

	ds, _ := data.NewRegressionDataset(value, value)
	loader, _ := data.NewLoader(ds, data.CollateRegression, data.LoaderOptions{BatchSize: 1})
	runner, _ := trainer.NewRegressor(model, &scriptedOptimizer{model: model}, tensor.CPU())
	path := filepath.Join(t.TempDir(), "best.goml")

	history, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 3, BestCheckpointPath: path})
	if err != nil || history.Epochs[0].Validation.MeanLoss != 0 || history.Epochs[2].Validation.MeanLoss != 1 {
		t.Fatal(history, err)
	}

	saved, err := checkpoint.Load(path)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		for _, p := range saved.Parameters() {
			_ = p.Value().Data().Close()
		}
	}()

	prediction, err := saved.Predict(value)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = prediction.Close() }()

	got, _ := prediction.HostData()
	if got[0] != 1 {
		t.Fatal("saved final rather than best model", got)
	}
}

type customDense struct{ layer.Layer }

func TestCheckpoint_UnsupportedArchitecturePreservesExistingFile(t *testing.T) {
	base := denseModel(t, 1, 1)

	model, err := layer.NewSequential(customDense{Layer: base})
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "preserved.goml")

	previous := []byte("previous checkpoint")
	if err := os.WriteFile(path, previous, 0o600); err != nil {
		t.Fatal(err)
	}

	runner, _ := trainer.NewRegressor(model, &scriptedOptimizer{model: base}, tensor.CPU())

	loader := regressionLoader(t)
	if _, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 1, BestCheckpointPath: path}); err == nil {
		t.Fatal("unsupported architecture saved")
	}

	after, err := os.ReadFile(path)
	if err != nil || !slices.Equal(previous, after) {
		t.Fatal("existing checkpoint changed", err)
	}
}

var _ trainer.Model = (*layer.Sequential)(nil)
