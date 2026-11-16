package nn

import "github.com/bpradana/goml/tensor"

// Sum allocates a sum over axes and records the broadcast backward operation.
// No axes reduces all dimensions. Invalid or duplicate axes return an error.
func Sum(a *Value, axes ...int) (*Value, error) {
	return reduce(a, false, axes)
}

// Mean allocates a mean over axes and records its derivative.
// No axes reduces all dimensions. Empty groups and invalid axes return errors.
func Mean(a *Value, axes ...int) (*Value, error) {
	return reduce(a, true, axes)
}

func reduce(a *Value, mean bool, axes []int) (*Value, error) {
	if err := checkValues("reduction", a); err != nil {
		return nil, err
	}

	var (
		data *tensor.Tensor
		err  error
	)

	if mean {
		data, err = tensor.Mean(a.data, axes...)
	} else {
		data, err = tensor.Sum(a.data, axes...)
	}

	if err != nil {
		return nil, err
	}

	out := record(data, a)
	if !out.requires {
		return out, nil
	}

	shape := a.data.Shape()
	reduced := make([]bool, len(shape))

	if len(axes) == 0 {
		for i := range reduced {
			reduced[i] = true
		}
	} else {
		for _, axis := range axes {
			reduced[axis] = true
		}
	}

	keep := make([]int, len(shape))
	count := 1

	if data.Numel() > 0 {
		count = a.data.Numel() / data.Numel()
	}

	for i, d := range shape {
		keep[i] = d
		if reduced[i] {
			keep[i] = 1
		}
	}

	out.backward = func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		if !g.IsContiguous() {
			g = g.Contiguous()
		}

		grad, err := g.Reshape(keep...)
		if err != nil {
			return nil, err
		}

		ones, err := tensor.Ones(shape...)
		if err != nil {
			return nil, err
		}

		grad, err = tensor.Mul(grad, ones)
		if err != nil {
			return nil, err
		}

		if mean {
			grad, err = tensor.Scale(grad, 1/float32(count))
		}

		return []*tensor.Tensor{grad}, err
	}

	return out, nil
}
