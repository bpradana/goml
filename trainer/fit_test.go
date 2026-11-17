package trainer_test

import (
	"context"
	"errors"
	"iter"
	"math"
	"testing"

	"github.com/bpradana/goml/trainer"
)

type numberSource []int

func (s numberSource) Batches(ctx context.Context, epoch int) iter.Seq2[int, error] {
	return func(yield func(int, error) bool) {
		for _, value := range s {
			if !yield(value, nil) {
				return
			}
		}
	}
}

func TestFit_CustomWeightedStepsHistoryAndCallbackIsolation(t *testing.T) {
	calls := 0
	step := func(ctx context.Context, batch int) (trainer.StepResult, error) {
		calls++
		return trainer.StepResult{MeanLoss: float64(batch), Samples: uint64(batch)}, ctx.Err()
	}

	history, err := trainer.Fit(context.Background(), numberSource{1, 3}, numberSource{1, 3}, step, step, trainer.FitOptions{Epochs: 2, OnEpochEnd: func(report trainer.EpochReport) error {
		report.Validation.MeanLoss = 99
		return nil
	}})
	if err != nil || calls != 8 || len(history.Epochs) != 2 || history.Epochs[0].Train.MeanLoss != 2.5 || history.Epochs[0].Validation.MeanLoss != 2.5 || history.StopReason != "completed" {
		t.Fatal(history, err, calls)
	}
}

func TestFit_CancellationCallbackAndInvalidOptionsPreserveCompletedReports(t *testing.T) {
	step := func(ctx context.Context, batch int) (trainer.StepResult, error) {
		return trainer.StepResult{MeanLoss: 1, Samples: 1}, ctx.Err()
	}
	wanted := errors.New("callback")

	history, err := trainer.Fit(context.Background(), numberSource{1}, nil, step, nil, trainer.FitOptions{Epochs: 3, OnEpochEnd: func(trainer.EpochReport) error { return wanted }})
	if !errors.Is(err, wanted) || len(history.Epochs) != 1 || history.StopReason != "" {
		t.Fatal(history, err)
	}

	for _, options := range []trainer.FitOptions{{}, {Epochs: -1}, {Epochs: 1, EarlyStopping: &trainer.EarlyStopping{Patience: 1}}, {Epochs: 1, BestCheckpointPath: "unused"}, {Epochs: 1, EarlyStopping: &trainer.EarlyStopping{Patience: 0, MinDelta: math.NaN()}}} {
		if _, err := trainer.Fit(context.Background(), numberSource{1}, nil, step, nil, options); err == nil {
			t.Fatal("invalid options accepted")
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := trainer.Fit(ctx, numberSource{1}, nil, step, nil, trainer.FitOptions{Epochs: 1}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}

	if _, err := trainer.Fit(context.Background(), numberSource{}, nil, step, nil, trainer.FitOptions{Epochs: 1}); err == nil {
		t.Fatal("empty training succeeded")
	}
}
