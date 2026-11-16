package tensor

import (
	"fmt"
	"math"
	"slices"
)

// SubScaled allocates a-factor*b for equally shaped tensors, including views.
// The scaled product is rounded to Float32 before subtraction, as with Scale
// followed by Sub. Invalid inputs or unequal shapes return errors.
func SubScaled(a, b *Tensor, factor float32) (*Tensor, error) {
	if err := valid("subscaled", a, b); err != nil {
		return nil, err
	}

	if !slices.Equal(a.shape, b.shape) {
		return nil, fmt.Errorf("tensor: subscaled: shapes %v and %v differ", a.shape, b.shape)
	}

	if err := compatible("subscaled", a, b); err != nil {
		return nil, err
	}

	if a.gpu != nil {
		scaled, err := Scale(b, factor)
		if err != nil {
			return nil, err
		}
		defer func() { _ = scaled.Close() }()

		return Sub(a, scaled)
	}

	out, err := allocated(a.shape)
	if err != nil {
		return nil, err
	}

	if a.IsContiguous() && b.IsContiguous() {
		av, bv := a.data[a.offset:a.offset+a.size], b.data[b.offset:b.offset+b.size]
		for i := range out.data {
			out.data[i] = av[i] - float32(factor*bv[i])
		}
	} else {
		for i := range out.data {
			out.data[i] = a.data[a.index(i)] - float32(factor*b.data[b.index(i)])
		}
	}

	return out, nil
}

// AdamUpdate allocates a bias-corrected Adam parameter update, without changing
// values or gradient. First and second are caller-owned, independent moment
// buffers in logical element order; only those buffers are updated in place.
// Shapes and moment lengths must match. Step must be positive, rate and epsilon
// finite and positive, and betas in [0,1). Validation errors leave moments intact.
func AdamUpdate(values, gradient *Tensor, first, second []float32, rate, beta1, beta2, epsilon float32, step uint64) (*Tensor, error) {
	if err := valid("adam", values, gradient); err != nil {
		return nil, err
	}

	if values.gpu != nil || gradient.gpu != nil {
		return nil, fmt.Errorf("tensor: adam: CPU moment slices cannot update GPU values; use tensor operations with GPU moments")
	}

	if !slices.Equal(values.shape, gradient.shape) || len(first) != values.size || len(second) != values.size {
		return nil, fmt.Errorf("tensor: adam: parameter, gradient, and moment shapes must match")
	}

	if step == 0 || !(rate > 0) || math.IsInf(float64(rate), 0) || !(epsilon > 0) || math.IsInf(float64(epsilon), 0) ||
		!(beta1 >= 0 && beta1 < 1 && beta2 >= 0 && beta2 < 1) {
		return nil, fmt.Errorf("tensor: adam: invalid step or hyperparameters")
	}

	out, err := allocated(values.shape)
	if err != nil {
		return nil, err
	}

	firstCorrection := 1 - math.Pow(float64(beta1), float64(step))
	secondCorrection := 1 - math.Pow(float64(beta2), float64(step))
	valuesContiguous, gradientContiguous := values.IsContiguous(), gradient.IsContiguous()

	for i := range out.data {
		vi, gi := values.offset+i, gradient.offset+i
		if !valuesContiguous {
			vi = values.index(i)
		}

		if !gradientContiguous {
			gi = gradient.index(i)
		}

		g := gradient.data[gi]
		first[i] = beta1*first[i] + (1-beta1)*g
		second[i] = beta2*second[i] + (1-beta2)*g*g
		m := float64(first[i]) / firstCorrection
		v := float64(second[i]) / secondCorrection
		out.data[i] = values.data[vi] - float32(float64(rate)*m/(math.Sqrt(v)+float64(epsilon)))
	}

	return out, nil
}
