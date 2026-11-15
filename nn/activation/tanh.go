package activation

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// Tanh is a stateless hyperbolic tangent layer.
type Tanh struct{}

// NewTanh constructs a hyperbolic tangent activation layer.
func NewTanh() *Tanh {
	return &Tanh{}
}

// Forward allocates the element-wise hyperbolic tangent with the input's shape.
// NaNs propagate. Large finite inputs saturate toward negative or positive one.
// Invalid inputs and consumed graphs return errors. Inference disables tracking.
func (*Tanh) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn/activation: tanh: nil or uninitialized value")
	}

	source := input.Data()
	shape := source.Shape()
	values := source.Data()

	for i, x := range values {
		values[i] = float32(math.Tanh(float64(x)))
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
			gradient[i] *= 1 - value*value
		}

		return tensor.From(gradient, shape...)
	})
}

// Parameters returns nil because Tanh has no trainable parameters.
func (*Tanh) Parameters() []*nn.Parameter {
	return nil
}
