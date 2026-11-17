package trainer_test

import (
	"context"
	"testing"

	"github.com/bpradana/goml/trainer"
)

func TestEarlyStopping_PatienceAndStrictMinDelta(t *testing.T) {
	train := func(context.Context, int) (trainer.StepResult, error) {
		return trainer.StepResult{MeanLoss: 1, Samples: 1}, nil
	}

	for _, tc := range []struct {
		losses   []float64
		patience int
		delta    float64
		epochs   int
		stopped  bool
	}{
		{[]float64{1, 1, 1, 1}, 2, 0, 3, true},
		{[]float64{1, 0.875, 0.75, 0.5}, 3, 0.25, 4, false},
		{[]float64{1, 1}, 1, 0, 2, false},
	} {
		index := 0
		evaluate := func(context.Context, int) (trainer.StepResult, error) {
			loss := tc.losses[index]
			index++

			return trainer.StepResult{MeanLoss: loss, Samples: 1}, nil
		}

		history, err := trainer.Fit(context.Background(), numberSource{1}, numberSource{1}, train, evaluate, trainer.FitOptions{Epochs: len(tc.losses), EarlyStopping: &trainer.EarlyStopping{Patience: tc.patience, MinDelta: tc.delta}})
		if err != nil || len(history.Epochs) != tc.epochs || history.StoppedEarly != tc.stopped {
			t.Fatal(history, err)
		}
	}
}
