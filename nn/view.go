package nn

import "github.com/bpradana/goml/tensor"

// Reshape creates a differentiable view. It requires contiguous input storage
// and equal element counts, following tensor.Reshape.
func Reshape(a *Value, shape ...int) (*Value, error) {
	if err := checkValues("reshape", a); err != nil {
		return nil, err
	}

	data, err := a.data.Reshape(shape...)
	if err != nil {
		return nil, err
	}

	out := record(data, a)
	if out.requires {
		original := a.data.Shape()
		out.backward = func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
			grad, err := g.Contiguous().Reshape(original...)
			return []*tensor.Tensor{grad}, err
		}
	}

	return out, nil
}

// Transpose creates a differentiable axis-permuted view.
// No axes reverses dimensions; invalid permutations return an error.
func Transpose(a *Value, axes ...int) (*Value, error) {
	if err := checkValues("transpose", a); err != nil {
		return nil, err
	}

	data, err := a.data.Transpose(axes...)
	if err != nil {
		return nil, err
	}

	out := record(data, a)
	if out.requires {
		inverse := make([]int, a.data.Rank())

		if len(axes) == 0 {
			for i := range inverse {
				inverse[i] = len(inverse) - 1 - i
			}
		} else {
			for i, axis := range axes {
				inverse[axis] = i
			}
		}

		out.backward = func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
			grad, err := g.Transpose(inverse...)
			return []*tensor.Tensor{grad}, err
		}
	}

	return out, nil
}
