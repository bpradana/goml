package nn

import (
	"fmt"

	"github.com/bpradana/goml/tensor"
)

// Concat allocates values joined along axis. Inputs must have equal ranks and
// equal dimensions outside axis. Empty axis dimensions are allowed. No inputs,
// invalid axes, incompatible shapes, and size overflow return errors.
func Concat(axis int, inputs ...*Value) (*Value, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("nn: concat: at least one input required")
	}

	if err := checkValues("concat", inputs...); err != nil {
		return nil, err
	}

	tensors := make([]*tensor.Tensor, len(inputs))
	shapes := make([][]int, len(inputs))

	for i, input := range inputs {
		tensors[i] = input.data
		shapes[i] = input.data.Shape()
	}

	data, err := tensor.Concat(axis, tensors...)
	if err != nil {
		return nil, err
	}

	return NewMultiOperation(data, inputs, func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		grads := make([]*tensor.Tensor, len(shapes))

		start := 0
		for i, current := range shapes {
			part, err := g.Slice(axis, start, start+current[axis])
			if err != nil {
				return nil, err
			}

			grads[i], err = part.MakeContiguous()
			if err != nil {
				return nil, err
			}

			start += current[axis]
		}

		return grads, nil
	})
}
