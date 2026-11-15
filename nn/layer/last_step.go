package layer

import (
	"fmt"

	"github.com/bpradana/goml/nn"
)

// LastStep selects the final time step of a batch-first sequence.
type LastStep struct{}

// NewLastStep constructs a stateless sequence-to-vector layer.
func NewLastStep() *LastStep {
	return &LastStep{}
}

// Forward maps [batch,time,features] to [batch,features]. Time must be positive.
// It materializes the selected values and routes gradients only to the last step.
// Invalid inputs, ranks, and consumed graphs return errors.
func (*LastStep) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: last step: uninitialized input")
	}

	shape := input.Data().Shape()
	if len(shape) != 3 || shape[1] <= 0 {
		return nil, fmt.Errorf("nn: last step: expected [batch,time>0,features], got %v", shape)
	}

	last, err := nn.Slice(input, 1, shape[1]-1, shape[1])
	if err != nil {
		return nil, err
	}

	last, err = nn.Contiguous(last)
	if err != nil {
		return nil, err
	}

	return nn.Reshape(last, shape[0], shape[2])
}

// Parameters returns nil because LastStep has no trainable parameters.
func (*LastStep) Parameters() []*nn.Parameter {
	return nil
}
