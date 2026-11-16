package activation

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// Sigmoid is a stateless logistic sigmoid layer.
type Sigmoid struct{}

// NewSigmoid constructs a logistic sigmoid activation layer.
func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

// Forward allocates the element-wise logistic sigmoid with the input's shape.
// A stable evaluation avoids exponential overflow for large negative inputs.
// Invalid inputs and consumed graphs return errors. Inference disables tracking.
func (*Sigmoid) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn/activation: sigmoid: nil or uninitialized value")
	}

	if !input.Data().IsCPU() {
		return deviceActivation(input, "sigmoid")
	}

	source := input.Data()
	shape := source.Shape()
	values := source.Data()

	for i, x := range values {
		if x >= 0 {
			values[i] = float32(1 / (1 + math.Exp(-float64(x))))
		} else {
			exponential := math.Exp(float64(x))
			values[i] = float32(exponential / (1 + exponential))
		}
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
			gradient[i] *= value * (1 - value)
		}

		return tensor.From(gradient, shape...)
	})
}

// Parameters returns nil because Sigmoid has no trainable parameters.
func (*Sigmoid) Parameters() []*nn.Parameter {
	return nil
}
