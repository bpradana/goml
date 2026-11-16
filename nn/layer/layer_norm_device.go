package layer

import (
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func (l *LayerNorm) deviceForward(input *nn.Value) (*nn.Value, error) {
	shape := input.Data().Shape()
	axis := len(shape) - 1
	keep := input.Data().Shape()
	keep[axis] = 1

	mean, err := nn.Mean(input, axis)
	if err != nil {
		return nil, err
	}

	mean, err = nn.Reshape(mean, keep...)
	if err != nil {
		return nil, err
	}

	centered, err := nn.Sub(input, mean)
	if err != nil {
		return nil, err
	}

	squares, err := nn.Mul(centered, centered)
	if err != nil {
		return nil, err
	}

	variance, err := nn.Mean(squares, axis)
	if err != nil {
		return nil, err
	}

	stabilized, err := tensor.AddScalar(variance.Data(), float32(l.epsilon))
	if err != nil {
		return nil, err
	}

	adjusted, err := nn.NewOperation(stabilized, variance, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return g, nil
	})
	if err != nil {
		return nil, err
	}

	deviation, err := nn.Sqrt(adjusted)
	if err != nil {
		return nil, err
	}

	deviation, err = nn.Reshape(deviation, keep...)
	if err != nil {
		return nil, err
	}

	normalized, err := nn.Div(centered, deviation)
	if err != nil {
		return nil, err
	}

	scaled, err := nn.Mul(normalized, l.scale.Value())
	if err != nil {
		return nil, err
	}

	return nn.Add(scaled, l.bias.Value())
}
