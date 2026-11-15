package activation

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// ReLU is a stateless rectified linear layer.
type ReLU struct{}

// NewReLU constructs a rectified linear activation layer.
func NewReLU() *ReLU {
	return &ReLU{}
}

// Forward allocates the element-wise rectified linear with the input's shape.
// NaNs propagate. The derivative at zero is defined as zero.
// Invalid inputs and consumed graphs return errors. Inference disables tracking.
func (*ReLU) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn/activation: relu: nil or uninitialized value")
	}

	source := input.Data()
	shape := source.Shape()
	values := source.Data()

	for i, x := range values {
		values[i] = float32(math.Max(0, float64(x)))
	}

	data, err := tensor.From(values, shape...)
	if err != nil {
		return nil, err
	}

	if !input.RequiresGrad() {
		return nn.NewOperation(data, input, nil)
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		gradient := g.Data()

		for i, value := range values {
			if value <= 0 {
				gradient[i] = 0
			} else if math.IsNaN(float64(value)) {
				gradient[i] = float32(math.NaN())
			}
		}

		return tensor.From(gradient, shape...)
	})
}

// Parameters returns nil because ReLU has no trainable parameters.
func (*ReLU) Parameters() []*nn.Parameter {
	return nil
}
