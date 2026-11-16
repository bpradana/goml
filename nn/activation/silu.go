package activation

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// SiLU is the stateless sigmoid-weighted linear activation x*sigmoid(x).
type SiLU struct{}

// NewSiLU constructs a sigmoid-weighted linear activation layer.
func NewSiLU() *SiLU {
	return &SiLU{}
}

// Forward allocates an element-wise SiLU result with the input shape. Invalid
// inputs and consumed graphs return errors. Negative infinity maps to negative
// zero, positive infinity to itself, and NaN propagates. Inference skips tracking.
func (*SiLU) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn/activation: silu: nil or uninitialized value")
	}

	shape := input.Data().Shape()
	values := input.Data().Data()

	var sigmoid []float32
	if input.RequiresGrad() {
		sigmoid = make([]float32, len(values))
	}

	for i, x := range values {
		var probability float32
		if x >= 0 {
			probability = float32(1 / (1 + math.Exp(-float64(x))))
		} else {
			exponential := math.Exp(float64(x))
			probability = float32(exponential / (1 + exponential))
		}

		values[i] = x * probability
		if math.IsInf(float64(x), -1) {
			values[i] = float32(math.Copysign(0, -1))
		}

		if sigmoid != nil {
			sigmoid[i] = probability
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

		for i, probability := range sigmoid {
			derivative := probability + values[i]*(1-probability)
			if math.IsInf(float64(values[i]), 1) {
				derivative = 1
			}

			gradient[i] *= derivative
		}

		return tensor.From(gradient, shape...)
	})
}

// Parameters returns nil because SiLU has no trainable parameters.
func (*SiLU) Parameters() []*nn.Parameter {
	return nil
}
