package nn

import "github.com/bpradana/goml/tensor"

// Contiguous allocates independent row-major storage, preserving the shape and
// gradient relationship. Invalid values and consumed graphs return an error.
func Contiguous(input *Value) (*Value, error) {
	if err := checkValues("contiguous", input); err != nil {
		return nil, err
	}

	data, err := input.data.MakeContiguous()
	if err != nil {
		return nil, err
	}

	return NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return g, nil
	})
}
