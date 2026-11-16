package loss

import (
	"fmt"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func deviceCrossEntropy(prediction *nn.Value, targets []int) (*nn.Value, error) {
	shape := prediction.Data().Shape()
	labels := make([]float32, prediction.Data().Numel())

	for row, class := range targets {
		if class < 0 || class >= shape[1] {
			return nil, fmt.Errorf("nn/loss: cross-entropy: invalid class %d at row %d", class, row)
		}

		labels[row*shape[1]+class] = 1
	}

	if err := tensor.CheckFinite(prediction.Data()); err != nil {
		return nil, err
	}

	target, err := tensor.FromLike(prediction.Data(), labels, shape...)
	if err != nil {
		return nil, err
	}

	maximum, err := tensor.Max(prediction.Data(), 1)
	if err != nil {
		return nil, err
	}

	maximumView, err := maximum.Reshape(shape[0], 1)
	if err != nil {
		return nil, err
	}

	shifted, err := tensor.Sub(prediction.Data(), maximumView)
	if err != nil {
		return nil, err
	}

	exponentials, err := tensor.Exp(shifted)
	if err != nil {
		return nil, err
	}

	total, err := tensor.Sum(exponentials, 1)
	if err != nil {
		return nil, err
	}

	logTotal, err := tensor.Log(total)
	if err != nil {
		return nil, err
	}

	selected, err := tensor.Mul(prediction.Data(), target)
	if err != nil {
		return nil, err
	}

	selected, err = tensor.Sum(selected, 1)
	if err != nil {
		return nil, err
	}

	offset, err := tensor.Sub(maximum, selected)
	if err != nil {
		return nil, err
	}

	losses, err := tensor.Add(logTotal, offset)
	if err != nil {
		return nil, err
	}

	result, err := tensor.Mean(losses)
	if err != nil {
		return nil, err
	}

	if !prediction.RequiresGrad() {
		return nn.NewOperation(result, prediction, nil)
	}

	totalView, err := total.Reshape(shape[0], 1)
	if err != nil {
		return nil, err
	}

	probabilities, err := tensor.Div(exponentials, totalView)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(result, prediction, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		difference, err := tensor.Sub(probabilities, target)
		if err != nil {
			return nil, err
		}

		difference, err = tensor.Scale(difference, 1/float32(shape[0]))
		if err != nil {
			return nil, err
		}

		return tensor.Mul(difference, g)
	})
}
