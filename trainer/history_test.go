package trainer_test

import (
	"context"
	"math"
	"testing"

	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

func TestHistory_ClassificationCallbacksOwnIndependentReports(t *testing.T) {
	model := denseModel(t, 1, 2)
	opt, _ := optimizer.NewSGD(model.Parameters(), 0.1)
	runner, _ := trainer.NewClassifier(model, opt, tensor.CPU(), 2)
	loader := classLoader(t, 3)

	history, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 2, OnEpochEnd: func(report trainer.EpochReport) error {
		report.Classification.Confusion[0][0] = 99
		report.Classification.PerClass[0].Support = 99
		report.Classification.Accuracy = 99

		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}

	for i, report := range history.Epochs {
		if report.Epoch != i+1 || report.Classification.Accuracy > 1 || report.Classification.PerClass[0].Support != 2 || report.Classification.Confusion[0][0] == 99 {
			t.Fatal(report)
		}
	}
}

func TestHistory_InvalidStepCountsAndNonfiniteLosses(t *testing.T) {
	for _, result := range []trainer.StepResult{{Samples: 0}, {Samples: 1, MeanLoss: math.NaN()}, {Samples: 1, MeanLoss: math.Inf(1)}} {
		step := func(context.Context, int) (trainer.StepResult, error) { return result, nil }
		if _, err := trainer.Fit(context.Background(), numberSource{1}, nil, step, nil, trainer.FitOptions{Epochs: 1}); err == nil {
			t.Fatal("invalid step report accepted")
		}
	}
}
