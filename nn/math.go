package nn

import "github.com/bpradana/goml/tensor"

// Exp allocates the element-wise exponential on the input device and records its
// derivative. IEEE overflow produces infinity; invalid inputs return errors.
func Exp(input *Value) (*Value, error) {
	if err := checkValues("exp", input); err != nil {
		return nil, err
	}

	result, err := tensor.Exp(input.data)
	if err != nil {
		return nil, err
	}

	return NewOperation(result, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Mul(g, result)
	})
}

// Log allocates the element-wise natural logarithm on the input device and
// records its derivative. Zero and negative inputs follow tensor.Log semantics.
func Log(input *Value) (*Value, error) {
	if err := checkValues("log", input); err != nil {
		return nil, err
	}

	saved := input.data

	result, err := tensor.Log(saved)
	if err != nil {
		return nil, err
	}

	return NewOperation(result, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Div(g, saved)
	})
}

// Sqrt allocates element-wise square roots on the input device and records the
// derivative. Negative inputs produce NaNs; the derivative at zero is singular.
func Sqrt(input *Value) (*Value, error) {
	if err := checkValues("sqrt", input); err != nil {
		return nil, err
	}

	result, err := tensor.Sqrt(input.data)
	if err != nil {
		return nil, err
	}

	return NewOperation(result, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		denominator, err := tensor.Scale(result, 2)
		if err != nil {
			return nil, err
		}

		return tensor.Div(g, denominator)
	})
}
