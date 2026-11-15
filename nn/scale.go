package nn

import "github.com/bpradana/goml/tensor"

// Scale allocates input*factor with the same shape and records its derivative.
// NaNs and infinities follow IEEE 754. Invalid values return an error.
func Scale(input *Value, factor float32) (*Value, error) {
	if err := checkValues("scale", input); err != nil {
		return nil, err
	}

	data, err := tensor.Scale(input.data, factor)
	if err != nil {
		return nil, err
	}

	return NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Scale(g, factor)
	})
}
