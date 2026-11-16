package activation

import (
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func deviceSoftmax(input *nn.Value, axis int) (*nn.Value, error) {
	shape := input.Data().Shape()
	keep := input.Data().Shape()
	keep[axis] = 1

	maximum, err := tensor.Max(input.Data(), axis)
	if err != nil {
		return nil, err
	}

	maximum, err = maximum.Reshape(keep...)
	if err != nil {
		return nil, err
	}

	shifted, err := tensor.Sub(input.Data(), maximum)
	if err != nil {
		return nil, err
	}

	exponentials, err := tensor.Exp(shifted)
	if err != nil {
		return nil, err
	}

	total, err := tensor.Sum(exponentials, axis)
	if err != nil {
		return nil, err
	}

	total, err = total.Reshape(keep...)
	if err != nil {
		return nil, err
	}

	probabilities, err := tensor.Div(exponentials, total)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(probabilities, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		weighted, err := tensor.Mul(g, probabilities)
		if err != nil {
			return nil, err
		}

		sum, err := tensor.Sum(weighted, axis)
		if err != nil {
			return nil, err
		}

		sum, err = sum.Reshape(keep...)
		if err != nil {
			return nil, err
		}

		centered, err := tensor.Sub(g, sum)
		if err != nil {
			return nil, err
		}

		gradient, err := tensor.Mul(centered, probabilities)
		if err != nil {
			return nil, err
		}

		return gradient.Reshape(shape...)
	})
}
