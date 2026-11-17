package trainer

import (
	"context"
	"fmt"
	"math"
)

// StepResult describes a successful batch's sample-mean loss. For multi-output
// tasks compute the per-sample output mean, then return the number of samples.
type StepResult struct {
	MeanLoss float64
	Samples  uint64
}

// StepFunc handles one owned batch. It must close owned resources on every path,
// including canceled contexts, and clear task-specific gradients on failure.
// This typed extension supports arbitrary data.Loader samples/batches and losses.
type StepFunc[B any] func(context.Context, B) (StepResult, error)

// FitOptions controls one run. Callbacks run synchronously after a completed
// epoch/evaluation and successful best-checkpoint save. Their reports are isolated
// copies. BestCheckpointPath requires validation and uses the strict lowest
// validation loss, independent of early stopping's MinDelta. No optimizer state
// is serialized. Options and callback captures must not be mutated concurrently.
type FitOptions struct {
	Epochs             int
	OnEpochEnd         func(EpochReport) error
	EarlyStopping      *EarlyStopping
	BestCheckpointPath string
}

// Fit orchestrates arbitrary typed steps without a model/task hierarchy. Steps
// own batch cleanup and all numerical behavior. Evaluation uses evaluateStep with
// its own application-owned typed accumulator if desired. Built-in classifier /
// regressor methods additionally attach metric snapshots and model checkpoints.
// Generic Fit rejects BestCheckpointPath: custom applications save in callbacks.
func Fit[B any](ctx context.Context, train, validation BatchSource[B], trainStep, evaluateStep StepFunc[B], options FitOptions) (History, error) {
	if train == nil || trainStep == nil || (validation != nil && evaluateStep == nil) {
		return History{}, fmt.Errorf("trainer: batch source and matching step functions required")
	}

	if options.BestCheckpointPath != "" {
		return History{}, fmt.Errorf("trainer: custom Fit: save custom checkpoints in OnEpochEnd")
	}

	var evaluate func(int) (Evaluation, error)
	if validation != nil {
		evaluate = func(epoch int) (Evaluation, error) {
			stats, err := runEpoch(ctx, validation, epoch, evaluateStep)
			return Evaluation{EpochStats: stats}, err
		}
	}

	return fit(ctx, options, func(epoch int) (EpochStats, error) { return runEpoch(ctx, train, epoch, trainStep) }, evaluate, nil)
}

func runEpoch[B any](ctx context.Context, source BatchSource[B], epoch int, step StepFunc[B]) (EpochStats, error) {
	var stats EpochStats
	if err := ctx.Err(); err != nil {
		return stats, err
	}

	for batch, err := range source.Batches(ctx, epoch) {
		if err != nil {
			return stats, err
		}
		// The step always receives successful batches, even if cancellation
		// races a yield, so its ownership/cleanup contract is never bypassed.
		result, err := step(ctx, batch)
		if err != nil {
			return stats, err
		}

		if err := ctx.Err(); err != nil {
			return stats, err
		}

		if err := stats.add(result); err != nil {
			return stats, err
		}
	}

	if err := ctx.Err(); err != nil {
		return stats, err
	}

	if stats.Samples == 0 {
		return stats, fmt.Errorf("trainer: empty epoch")
	}

	return stats, nil
}

func fit(ctx context.Context, options FitOptions, train func(int) (EpochStats, error), evaluate func(int) (Evaluation, error), save func(string) error) (History, error) {
	var history History
	if options.Epochs <= 0 {
		return history, fmt.Errorf("trainer: epochs must be positive")
	}

	if (options.EarlyStopping != nil || options.BestCheckpointPath != "") && evaluate == nil {
		return history, fmt.Errorf("trainer: early stopping and best checkpoint require validation")
	}

	if options.EarlyStopping != nil {
		if err := options.EarlyStopping.validate(); err != nil {
			return history, err
		}
	}

	stopping := stoppingState{best: math.Inf(1)}
	checkpointBest := math.Inf(1)

	for epoch := range options.Epochs {
		if err := ctx.Err(); err != nil {
			return history, err
		}

		stats, err := train(epoch)
		if err != nil {
			return history, fmt.Errorf("trainer: epoch %d training: %w", epoch+1, err)
		}

		report := EpochReport{Epoch: epoch + 1, Train: stats}
		if evaluate != nil {
			result, err := evaluate(epoch)
			if err != nil {
				return history, fmt.Errorf("trainer: epoch %d validation: %w", epoch+1, err)
			}

			report.Validation = &result.EpochStats
			report.Classification = result.Classification
			report.Regression = result.Regression
		}

		history.Epochs = append(history.Epochs, report)
		if err := ctx.Err(); err != nil {
			return history, err
		}

		if options.BestCheckpointPath != "" && report.Validation.MeanLoss < checkpointBest {
			if err := save(options.BestCheckpointPath); err != nil {
				return history, fmt.Errorf("trainer: best checkpoint: %w", err)
			}

			checkpointBest = report.Validation.MeanLoss
		}

		if options.OnEpochEnd != nil {
			if err := options.OnEpochEnd(cloneReport(report)); err != nil {
				return history, fmt.Errorf("trainer: epoch %d callback: %w", epoch+1, err)
			}
		}

		if err := ctx.Err(); err != nil {
			return history, err
		}

		if options.EarlyStopping != nil && stopping.update(report.Validation.MeanLoss, *options.EarlyStopping) && epoch+1 < options.Epochs {
			history.StoppedEarly = true
			history.StopReason = "early_stopping"

			return history, nil
		}
	}

	history.StopReason = "completed"

	return history, nil
}
