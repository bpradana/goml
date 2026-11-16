package tensor

import (
	"fmt"
	"slices"
)

// Concat allocates tensors joined along axis. Inputs must have equal ranks and
// equal dimensions outside axis; empty dimensions and strided views are supported.
// No inputs, invalid tensors or axes, incompatible shapes, and overflow fail.
func Concat(axis int, inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("tensor: concat: at least one input required")
	}

	if err := valid("concat", inputs...); err != nil {
		return nil, err
	}

	shape := slices.Clone(inputs[0].shape)
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("tensor: concat: axis %d outside shape %v", axis, shape)
	}

	shape[axis] = 0
	for _, input := range inputs {
		if input.Rank() != len(shape) {
			return nil, fmt.Errorf("tensor: concat: incompatible ranks")
		}

		for i, dim := range input.shape {
			if i != axis && dim != shape[i] {
				return nil, fmt.Errorf("tensor: concat: incompatible shapes %v and %v", shape, input.shape)
			}
		}

		if input.shape[axis] > int(^uint(0)>>1)-shape[axis] {
			return nil, fmt.Errorf("tensor: concat: size overflow")
		}

		shape[axis] += input.shape[axis]
	}

	out, err := allocated(shape)
	if err != nil || out.size == 0 {
		return out, err
	}

	inner := out.strides[axis]
	outer := out.size / (shape[axis] * inner)
	offset := 0

	for _, input := range inputs {
		width := input.shape[axis] * inner
		contiguous := input.IsContiguous()

		for row := range outer {
			target := out.data[row*shape[axis]*inner+offset:][:width]
			if contiguous {
				start := input.offset + row*width
				copy(target, input.data[start:start+width])
			} else {
				for j := range target {
					target[j] = input.data[input.index(row*width+j)]
				}
			}
		}

		offset += width
	}

	return out, nil
}
