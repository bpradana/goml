package trainer

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/metrics"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

// Regressor trains fixed-shape numerical outputs with mean squared error.
// Epoch loss weights samples; metrics flatten output scalars equally. Model,
// optimizer, and device are borrowed. Outputs and targets must match exactly.
type Regressor struct {
	model     Model
	optimizer optimizer.Optimizer
	device    backend.Device
	metrics   []metrics.Accumulator[float32, float32]
}

// NewRegressor validates initialized dependencies and model placement. Move the
// model before optimizer construction; the trainer never moves parameters.
// Extra metrics receive flattened predictions/targets during evaluation. They
// reset per evaluation; their concrete typed results remain application-owned.
func NewRegressor(model Model, opt optimizer.Optimizer, device backend.Device, extraMetrics ...metrics.Accumulator[float32, float32]) (*Regressor, error) {
	if err := validateModel(model, device); err != nil {
		return nil, err
	}

	if opt == nil {
		return nil, fmt.Errorf("trainer: regressor: optimizer required")
	}

	for _, metric := range extraMetrics {
		if metric == nil {
			return nil, fmt.Errorf("trainer: nil custom regression metric")
		}
	}

	return &Regressor{model: model, optimizer: opt, device: device, metrics: slices.Clone(extraMetrics)}, nil
}

// Fit owns only iteration temporaries and yielded batches. Completed reports
// survive failures. Optimizers and their persistent state remain caller-owned.
func (r *Regressor) Fit(ctx context.Context, train, validation BatchSource[*data.RegressionBatch], options FitOptions) (History, error) {
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
		return runEpoch(ctx, train, epoch, func(ctx context.Context, batch *data.RegressionBatch) (StepResult, error) {
			return r.batch(ctx, batch, true, nil)
		})
	}, evaluate, func(path string) error { return saveModel(path, r.model) })
}

// Evaluate runs inference and streaming scalar regression metrics. It neither
// steps the optimizer nor clears existing model gradients. Epoch is zero.
func (r *Regressor) Evaluate(ctx context.Context, source BatchSource[*data.RegressionBatch]) (Evaluation, error) {
	if err := validateModel(r.model, r.device); err != nil {
		return Evaluation{}, err
	}

	return r.evaluate(ctx, source, 0)
}

func (r *Regressor) evaluate(ctx context.Context, source BatchSource[*data.RegressionBatch], epoch int) (Evaluation, error) {
	if source == nil {
		return Evaluation{}, fmt.Errorf("trainer: evaluation source required")
	}

	metric := metrics.NewRegression()

	for _, extra := range r.metrics {
		if extra == nil {
			return Evaluation{}, fmt.Errorf("trainer: nil custom regression metric")
		}

		extra.Reset()
	}

	stats, err := runEpoch(ctx, source, epoch, func(ctx context.Context, batch *data.RegressionBatch) (StepResult, error) {
		return r.batch(ctx, batch, false, metric)
	})
	if err != nil {
		return Evaluation{}, err
	}

	result, err := metric.Result()

	return Evaluation{EpochStats: stats, Regression: &result}, err
}

func (r *Regressor) batch(ctx context.Context, batch *data.RegressionBatch, training bool, metric *metrics.Regression) (result StepResult, err error) {
	if batch == nil {
		return result, fmt.Errorf("trainer: nil regression batch")
	}
	defer func() { err = errors.Join(err, batch.Close()) }()

	if training {
		r.optimizer.ZeroGrad()
		defer r.optimizer.ZeroGrad()
	}

	if err := ctx.Err(); err != nil {
		return result, err
	}

	if batch.Inputs == nil || batch.Targets == nil || batch.Inputs.IsClosed() || batch.Targets.IsClosed() || batch.Inputs.Rank() == 0 || batch.Targets.Rank() == 0 || batch.Inputs.Shape()[0] <= 0 || batch.Inputs.Shape()[0] != batch.Targets.Shape()[0] {
		return result, fmt.Errorf("trainer: regression inputs and targets require matching positive sample dimensions")
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

	targets, err := scope.To(batch.Targets)
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

	if !slices.Equal(prediction.Data().Shape(), targets.Shape()) {
		return result, fmt.Errorf("trainer: regression prediction %v differs from target %v", prediction.Data().Shape(), targets.Shape())
	}

	var targetValue *nn.Value
	if training {
		targetValue, err = nn.NewValue(targets, false)
	} else {
		targetValue, err = nn.Inference(targets)
	}

	if err != nil {
		return result, err
	}

	objective, err := loss.NewMSE().Forward(prediction, targetValue)
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, objective.Data().Close()) }()

	mean, err := scalarLoss(objective)
	if err != nil {
		return result, err
	}

	if metric != nil {
		values, err := prediction.Data().HostData()
		if err != nil {
			return result, err
		}

		expected, err := batch.Targets.HostData()
		if err != nil {
			return result, err
		}

		if err := metric.Update(values, expected); err != nil {
			return result, err
		}

		for _, extra := range r.metrics {
			if err := extra.Update(values, expected); err != nil {
				return result, fmt.Errorf("trainer: custom regression metric: %w", err)
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

	return StepResult{MeanLoss: mean, Samples: uint64(batch.Inputs.Shape()[0])}, nil
}
