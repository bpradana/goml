package loss

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/nn"
)

// MSE computes mean squared error without trainable parameters or mutable state.
type MSE struct{}

// NewMSE constructs a mean squared error loss.
func NewMSE() *MSE {
	return &MSE{}
}

// Forward returns a scalar averaging all squared prediction-target errors.
// Inputs must be initialized, nonempty, and have identical shapes.
// Broadcasting is rejected. Gradients are recorded for either tracked input.
// Invalid values, shapes, and consumed graphs return errors.
// Inference mode suppresses graph recording.
func (*MSE) Forward(prediction, target *nn.Value) (*nn.Value, error) {
	if prediction == nil || target == nil || prediction.Data() == nil || target.Data() == nil {
		return nil, fmt.Errorf("nn/loss: mse: nil or uninitialized value")
	}

	predictionData, targetData := prediction.Data(), target.Data()
	if predictionData.Numel() == 0 || !slices.Equal(predictionData.Shape(), targetData.Shape()) {
		return nil, fmt.Errorf("nn/loss: mse: nonempty equal shapes required, got %v and %v", predictionData.Shape(), targetData.Shape())
	}

	difference, err := nn.Sub(prediction, target)
	if err != nil {
		return nil, err
	}

	squared, err := nn.Mul(difference, difference)
	if err != nil {
		return nil, err
	}

	return nn.Mean(squared)
}
