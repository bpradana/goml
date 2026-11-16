package nn

import "github.com/bpradana/goml/tensor"

// Slice returns a differentiable storage-sharing view selecting [start,end) on
// axis. Bounds follow tensor.Slice. Backward scatters into a new input gradient.
func Slice(input *Value, axis, start, end int) (*Value, error) {
	if err := checkValues("slice", input); err != nil {
		return nil, err
	}

	data, err := input.data.Slice(axis, start, end)
	if err != nil {
		return nil, err
	}

	if !input.RequiresGrad() {
		return NewOperation(data, input, nil)
	}

	shape := input.data.Shape()

	return NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.ScatterSlice(g, shape, axis, start)
	})
}
