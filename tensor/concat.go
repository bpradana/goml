package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
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

	if err := compatible("concat", inputs...); err != nil {
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

	defer runtime.KeepAlive(inputs)

	views := make([]backend.View, len(inputs))
	for i, input := range inputs {
		views[i] = input.deviceView()
	}

	compute, hasConcat := inputs[0].device.(backend.ConcatBackend)

	copier, hasCopy := inputs[0].device.(backend.CopyBackend)
	if !hasConcat && !hasCopy {
		return nil, unsupported("concat", inputs[0].device)
	}

	out, err := allocatedOn(shape, inputs...)
	if err != nil {
		return nil, err
	}

	if hasConcat {
		err = compute.Concat(out.deviceView(), axis, views)
	} else {
		offset := 0

		for _, source := range views {
			target := out.deviceView()
			target.Shape = slices.Clone(target.Shape)
			target.Shape[axis] = source.Shape[axis]
			target.Offset = offset * out.strides[axis]

			err = copier.Copy(target, source)
			if err != nil {
				break
			}

			offset += source.Shape[axis]
		}
	}

	if err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}
