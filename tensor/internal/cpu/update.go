package cpu

import (
	"fmt"
	"math"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// SubScaled writes a-factor*b without mutating either input.
func (Device) SubScaled(dst, left, right backend.View, factor float32) error {
	output, err := writeView(dst, true)
	if err != nil {
		return err
	}

	av, err := readView(left)
	if err != nil {
		return err
	}

	bv, err := readView(right)
	if err != nil {
		return err
	}

	out, a, b := &output, &av, &bv
	if !slices.Equal(out.shape, a.shape) || !slices.Equal(a.shape, b.shape) || dst.Buffer == left.Buffer || dst.Buffer == right.Buffer {
		return backend.ErrInvalidView
	}

	if out.size == 0 {
		return nil
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

	return nil
}

// Adam writes an immutable bias-corrected update and changes caller-owned moments.
func (Device) Adam(dst, src, grad backend.View, first, second []float32, options backend.AdamOptions) error {
	output, err := writeView(dst, true)
	if err != nil {
		return err
	}

	valuesArray, err := readView(src)
	if err != nil {
		return err
	}

	gradientArray, err := readView(grad)
	if err != nil {
		return err
	}

	out, values, gradient := &output, &valuesArray, &gradientArray
	rate, beta1, beta2, epsilon, step := options.Rate, options.Beta1, options.Beta2, options.Epsilon, options.Step

	if !slices.Equal(out.shape, values.shape) || dst.Buffer == src.Buffer || dst.Buffer == grad.Buffer {
		return backend.ErrInvalidView
	}

	if !slices.Equal(values.shape, gradient.shape) || len(first) != values.size || len(second) != values.size {
		return fmt.Errorf("tensor: adam: parameter, gradient, and moment shapes must match")
	}

	if step == 0 || !(rate > 0) || math.IsInf(float64(rate), 0) || !(epsilon > 0) || math.IsInf(float64(epsilon), 0) ||
		!(beta1 >= 0 && beta1 < 1 && beta2 >= 0 && beta2 < 1) {
		return fmt.Errorf("tensor: adam: invalid step or hyperparameters")
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

	return nil
}
