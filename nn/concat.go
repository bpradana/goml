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

	shape := inputs[0].data.Shape()
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("nn: concat: axis %d outside shape %v", axis, shape)
	}

	shape[axis] = 0
	count := 0
	maxInt := int(^uint(0) >> 1)

	for _, input := range inputs {
		current := input.data.Shape()
		if len(current) != len(shape) {
			return nil, fmt.Errorf("nn: concat: incompatible ranks")
		}

		for i, dim := range current {
			if i != axis && dim != shape[i] {
				return nil, fmt.Errorf("nn: concat: incompatible shapes %v and %v", shape, current)
			}
		}

		if current[axis] > maxInt-shape[axis] || input.data.Numel() > maxInt-count {
			return nil, fmt.Errorf("nn: concat: size overflow")
		}

		shape[axis] += current[axis]
		count += input.data.Numel()
	}

	values := make([]float32, count)
	inner, outer := 1, 0

	if count > 0 {
		for _, dim := range shape[axis+1:] {
			inner *= dim
		}

		outer = count / (shape[axis] * inner)
	}

	shapes := make([][]int, len(inputs))
	offset := 0

	for i, input := range inputs {
		shapes[i] = input.data.Shape()
		width := shapes[i][axis] * inner

		source := input.data.Data()
		for row := range outer {
			copy(values[row*shape[axis]*inner+offset:], source[row*width:(row+1)*width])
		}

		offset += width
	}

	data, err := tensor.From(values, shape...)
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

			grads[i] = part.Contiguous()
			start += current[axis]
		}

		return grads, nil
	})
}
