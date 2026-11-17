// Package trainer orchestrates typed training and evaluation using data loaders,
// existing nn operations, and explicitly selected Tensor devices. Models,
// optimizers, and devices remain caller-owned. Trainers are not concurrency-safe.
package trainer

import (
	"context"
	"fmt"
	"iter"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

// Model is satisfied by layer.Sequential. Predict must disable graph recording.
// Results are batch-owned handles: return a new tensor/view or Retain a borrowed
// parameter, never its original handle. Custom operations should propagate scopes
// using Tensor operations/FromLike and clean their own resources on failure.
type Model interface {
	Forward(*nn.Value) (*nn.Value, error)
	Predict(*tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*nn.Parameter
}

// BatchSource is satisfied by data.Loader[S,B] regardless of sample type S.
// Successful batches transfer ownership to the step. Error yields must not
// transfer resources. Sources must respect early termination and cancellation.
type BatchSource[B any] interface {
	Batches(context.Context, int) iter.Seq2[B, error]
}

func validateModel(model Model, device backend.Device) error {
	if model == nil || device == nil || device.ID() == "" {
		return fmt.Errorf("trainer: initialized model and device required")
	}

	for index, parameter := range model.Parameters() {
		if parameter == nil || parameter.Value() == nil || parameter.Value().Data() == nil || parameter.Value().Data().IsClosed() {
			return fmt.Errorf("trainer: invalid or closed parameter %d", index)
		}

		if parameter.Value().Data().Device().ID() != device.ID() {
			return fmt.Errorf("trainer: parameter %d: %w: place model before constructing optimizer/trainer", index, backend.ErrDeviceMismatch)
		}
	}

	return nil
}

func scalarLoss(value *nn.Value) (float64, error) {
	if value == nil || value.Data() == nil || value.Data().Rank() != 0 {
		return 0, fmt.Errorf("trainer: loss must be a scalar")
	}

	values, err := value.Data().HostData()
	if err != nil {
		return 0, err
	}

	result := float64(values[0])
	if math.IsNaN(result) || math.IsInf(result, 0) {
		return 0, fmt.Errorf("trainer: loss is nonfinite")
	}

	return result, nil
}

func validateOutput(output *tensor.Tensor, device backend.Device) error {
	if output == nil || output.IsClosed() || output.Device() == nil {
		return fmt.Errorf("trainer: model returned nil or closed output")
	}

	if output.Device().ID() != device.ID() {
		return fmt.Errorf("trainer: output: %w", backend.ErrDeviceMismatch)
	}

	return nil
}

func forward(model Model, input *tensor.Tensor, training bool) (*nn.Value, error) {
	if training {
		value, err := nn.NewValue(input, false)
		if err != nil {
			return nil, err
		}

		return model.Forward(value)
	}

	output, err := model.Predict(input)
	if err != nil {
		return nil, err
	}

	return nn.Inference(output)
}
