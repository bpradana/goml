package trainer_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/optimizer"
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

func TestCheckpoint_CustomSaverAndGenericFitSelectOnlyStrictImprovements(t *testing.T) {
	losses := []float64{1, 1, 0.5, 0.75}
	epoch, saves, callbacks := 0, 0, 0

	var events []string

	train := func(context.Context, int) (trainer.StepResult, error) {
		return trainer.StepResult{MeanLoss: 1, Samples: 1}, nil
	}
	evaluate := func(context.Context, int) (trainer.StepResult, error) {
		value := losses[epoch]
		epoch++

		return trainer.StepResult{MeanLoss: value, Samples: 1}, nil
	}
	options := trainer.FitOptions{
		Epochs:             4,
		BestCheckpointPath: "application-owned.bin",
		SaveBest: func(path string) error {
			if path != "application-owned.bin" {
				t.Fatal(path)
			}

			saves++

			events = append(events, "save")

			return nil
		},
		OnEpochEnd: func(trainer.EpochReport) error {
			callbacks++

			events = append(events, "report")

			return nil
		},
	}

	history, err := trainer.Fit(context.Background(), numberSource{1}, numberSource{1}, train, evaluate, options)
	if err != nil || len(history.Epochs) != 4 || saves != 2 || callbacks != 4 || !slices.Equal(events, []string{"save", "report", "report", "save", "report", "report"}) {
		t.Fatal(history, err, events)
	}
}

func TestCheckpoint_CustomSaverOverridesSequentialCodecAndPropagatesFailure(t *testing.T) {
	base := denseModel(t, 1, 1)

	model, err := layer.NewSequential(customDense{Layer: base})
	if err != nil {
		t.Fatal(err)
	}

	opt, err := optimizer.NewSGD(base.Parameters(), 0.01)
	if err != nil {
		t.Fatal(err)
	}

	runner, err := trainer.NewRegressor(model, opt, tensor.CPU())
	if err != nil {
		t.Fatal(err)
	}

	wanted := errors.New("custom save failed")
	saves, callbacks := 0, 0
	options := trainer.FitOptions{
		Epochs:             1,
		BestCheckpointPath: "custom.bin",
		SaveBest:           func(string) error { saves++; return wanted },
		OnEpochEnd:         func(trainer.EpochReport) error { callbacks++; return nil },
	}
	loader := regressionLoader(t)

	history, err := runner.Fit(context.Background(), loader, loader, options)
	if !errors.Is(err, wanted) || saves != 1 || callbacks != 0 || len(history.Epochs) != 1 {
		t.Fatal(history, err, saves, callbacks)
	}

	options.SaveBest = func(string) error { saves++; return nil }
	if _, err := runner.Fit(context.Background(), loader, loader, options); err != nil || saves != 2 || callbacks != 1 {
		t.Fatal(err, saves, callbacks)
	}

	for _, parameter := range model.Parameters() {
		if parameter.Value().Data().IsClosed() {
			t.Fatal("custom checkpoint closed model")
		}
	}
}

func TestCheckpoint_InvalidSaveBestConfiguration(t *testing.T) {
	step := func(context.Context, int) (trainer.StepResult, error) {
		return trainer.StepResult{MeanLoss: 1, Samples: 1}, nil
	}
	for _, options := range []trainer.FitOptions{
		{Epochs: 1, SaveBest: func(string) error { return nil }},
		{Epochs: 1, BestCheckpointPath: "missing-saver"},
	} {
		if _, err := trainer.Fit(context.Background(), numberSource{1}, numberSource{1}, step, step, options); err == nil {
			t.Fatal("invalid saver configuration accepted")
		}
	}

	if _, err := trainer.Fit(context.Background(), numberSource{1}, nil, step, nil, trainer.FitOptions{Epochs: 1, BestCheckpointPath: "path", SaveBest: func(string) error { return nil }}); err == nil {
		t.Fatal("checkpoint without validation accepted")
	}
}

type weightsOnlyModel struct{ trainer.Model }

func TestCheckpoint_NonSequentialModelStillSavesWeights(t *testing.T) {
	model := denseModel(t, 1, 1)

	runner, err := trainer.NewRegressor(weightsOnlyModel{model}, &scriptedOptimizer{model: model}, tensor.CPU())
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "weights.goml")

	loader := regressionLoader(t)
	if _, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 1, BestCheckpointPath: path}); err != nil {
		t.Fatal(err)
	}

	if err := checkpoint.LoadWeights(path, model.Parameters()); err != nil {
		t.Fatal(err)
	}
}
