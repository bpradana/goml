package trainer

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/metrics"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

// Classifier trains with stable mean cross-entropy from logits and evaluates
// streaming classification metrics. Dependencies are borrowed, never closed or
// moved. Place parameters before constructing the optimizer and this runner.
type Classifier struct {
	model     Model
	optimizer optimizer.Optimizer
	device    backend.Device
	classes   int
	metrics   []metrics.Accumulator[float32, int]
}

// NewClassifier validates initialized dependencies, current parameter device
// identity, and a positive nonoverflowing class count. It never migrates weights.
// Extra metrics receive raw [samples,classes] logits during evaluation and are
// reset once per evaluation. Read their typed Result in OnEpochEnd or afterward.
func NewClassifier(model Model, opt optimizer.Optimizer, device backend.Device, classes int, extraMetrics ...metrics.Accumulator[float32, int]) (*Classifier, error) {
	if err := validateModel(model, device); err != nil {
		return nil, err
	}

	if opt == nil || classes <= 0 || classes > int(^uint(0)>>1)/8/classes {
		return nil, fmt.Errorf("trainer: classifier: optimizer and valid class count required")
	}

	for _, metric := range extraMetrics {
		if metric == nil {
			return nil, fmt.Errorf("trainer: nil custom classification metric")
		}
	}

	return &Classifier{model: model, optimizer: opt, device: device, classes: classes, metrics: slices.Clone(extraMetrics)}, nil
}

// Fit runs zero-based loader epochs and returns one-based completed reports.
// Every yielded batch is closed. Errors preserve completed reports and clear
// step gradients, but cannot roll back optimizer updates already performed.
func (r *Classifier) Fit(ctx context.Context, train, validation BatchSource[*data.ClassificationBatch], options FitOptions) (History, error) {
	if train == nil {
		return History{}, fmt.Errorf("trainer: training source required")
	}

	if err := validateModel(r.model, r.device); err != nil {
		return History{}, err
	}

	var evaluate func(int) (Evaluation, error)
	if validation != nil {
		evaluate = func(epoch int) (Evaluation, error) { return r.evaluate(ctx, validation, epoch) }
	}

	return fit(ctx, options, func(epoch int) (EpochStats, error) {
		return runEpoch(ctx, train, epoch, func(ctx context.Context, batch *data.ClassificationBatch) (StepResult, error) {
			return r.batch(ctx, batch, true, nil)
		})
	}, evaluate, func(path string) error { return saveModel(path, r.model) })
}

// Evaluate performs inference without optimizer calls, retaining no graph or
// prediction arrays across batches. The source receives epoch zero.
func (r *Classifier) Evaluate(ctx context.Context, source BatchSource[*data.ClassificationBatch]) (Evaluation, error) {
	if err := validateModel(r.model, r.device); err != nil {
		return Evaluation{}, err
	}

	return r.evaluate(ctx, source, 0)
}

func (r *Classifier) evaluate(ctx context.Context, source BatchSource[*data.ClassificationBatch], epoch int) (Evaluation, error) {
	if source == nil {
		return Evaluation{}, fmt.Errorf("trainer: evaluation source required")
	}

	metric, err := metrics.NewClassification(r.classes)
	if err != nil {
		return Evaluation{}, err
	}

	for _, extra := range r.metrics {
		if extra == nil {
			return Evaluation{}, fmt.Errorf("trainer: nil custom classification metric")
		}

		extra.Reset()
	}

	stats, err := runEpoch(ctx, source, epoch, func(ctx context.Context, batch *data.ClassificationBatch) (StepResult, error) {
		return r.batch(ctx, batch, false, metric)
	})
	if err != nil {
		return Evaluation{}, err
	}

	result, err := metric.Result()

	return Evaluation{EpochStats: stats, Classification: &result}, err
}

func (r *Classifier) batch(ctx context.Context, batch *data.ClassificationBatch, training bool, metric *metrics.Classification) (result StepResult, err error) {
	if batch == nil {
		return result, fmt.Errorf("trainer: nil classification batch")
	}
	defer func() { err = errors.Join(err, batch.Close()) }()

	if training {
		r.optimizer.ZeroGrad()
		defer r.optimizer.ZeroGrad()
	}

	if err := ctx.Err(); err != nil {
		return result, err
	}

	if batch.Inputs == nil || batch.Inputs.IsClosed() || batch.Inputs.Rank() == 0 || len(batch.Labels) == 0 || batch.Inputs.Shape()[0] != len(batch.Labels) {
		return result, fmt.Errorf("trainer: classification batch requires [samples,...] inputs and matching nonempty labels")
	}

	scope, err := tensor.NewScope(r.device)
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, scope.Close()) }()

	input, err := scope.To(batch.Inputs)
	if err != nil {
		return result, err
	}

	prediction, err := forward(r.model, input, training)
	if err != nil {
		return result, err
	}

	if prediction == nil {
		return result, fmt.Errorf("trainer: nil prediction")
	}

	if prediction.Data() != nil {
		defer func() { err = errors.Join(err, prediction.Data().Close()) }()
	}

	if err := validateOutput(prediction.Data(), r.device); err != nil {
		return result, err
	}

	shape := prediction.Data().Shape()
	if len(shape) != 2 || shape[0] != len(batch.Labels) || shape[1] != r.classes {
		return result, fmt.Errorf("trainer: expected logits [%d,%d], got %v", len(batch.Labels), r.classes, shape)
	}

	objective, err := loss.NewCrossEntropy().Forward(prediction, batch.Labels)
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, objective.Data().Close()) }()

	mean, err := scalarLoss(objective)
	if err != nil {
		return result, err
	}

	if metric != nil {
		logits, err := prediction.Data().HostData()
		if err != nil {
			return result, err
		}

		if err := metric.UpdateLogits(logits, batch.Labels); err != nil {
			return result, err
		}

		for index, extra := range r.metrics {
			if err := extra.Update(logits, batch.Labels); err != nil {
				return result, fmt.Errorf("trainer: custom classification metric %d: %w", index, err)
			}
		}
	}

	if err := ctx.Err(); err != nil {
		return result, err
	}

	if training {
		if err := objective.Backward(); err != nil {
			return result, err
		}

		if err := ctx.Err(); err != nil {
			return result, err
		}

		if err := r.optimizer.Step(); err != nil {
			return result, err
		}
	}

	return StepResult{MeanLoss: mean, Samples: uint64(len(batch.Labels))}, nil
}
