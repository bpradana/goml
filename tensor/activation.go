package tensor

import (
	"fmt"
	"math"
	"slices"
)

// Activation computes relu, sigmoid, tanh, or silu on the input device.
// The result owns independent storage and preserves the input shape.
func Activation(kind string, input *Tensor) (*Tensor, error) {
	if err := valid("activation", input); err != nil {
		return nil, err
	}

	if kind != "relu" && kind != "sigmoid" && kind != "tanh" && kind != "silu" {
		return nil, fmt.Errorf("tensor: unknown activation %q", kind)
	}

	if input.gpu != nil {
		return gpuUnary(kind, input, 0)
	}

	return unary(kind, input, func(x float64) float64 {
		switch kind {
		case "relu":
			return math.Max(0, x)
		case "tanh":
			return math.Tanh(x)
		}

		var probability float64
		if x >= 0 {
			probability = 1 / (1 + math.Exp(-x))
		} else {
			e := math.Exp(x)
			probability = e / (1 + e)
		}

		if kind == "sigmoid" {
			return probability
		}

		if math.IsInf(x, -1) {
			return math.Copysign(0, -1)
		}

		return x * probability
	})
}

// ActivationBackward computes an activation's input derivative times gradient.
// Inputs must have matching shapes and devices. It does not build a graph.
func ActivationBackward(kind string, input, gradient *Tensor) (*Tensor, error) {
	if err := valid("activation backward", input, gradient); err != nil {
		return nil, err
	}

	if err := compatible("activation backward", input, gradient); err != nil {
		return nil, err
	}

	if !slices.Equal(input.shape, gradient.shape) {
		return nil, fmt.Errorf("tensor: activation backward: gradient shape %v must match %v", gradient.shape, input.shape)
	}

	if kind != "relu" && kind != "sigmoid" && kind != "tanh" && kind != "silu" {
		return nil, fmt.Errorf("tensor: unknown activation %q", kind)
	}

	if input.gpu == nil {
		derivative, err := unary("activation backward", input, func(x float64) float64 {
			if kind == "relu" {
				if math.IsNaN(x) {
					return x
				}

				if x > 0 {
					return 1
				}

				return 0
			}

			if kind == "tanh" {
				y := math.Tanh(x)
				return 1 - y*y
			}

			var probability float64
			if x >= 0 {
				probability = 1 / (1 + math.Exp(-x))
			} else {
				e := math.Exp(x)
				probability = e / (1 + e)
			}

			if kind == "sigmoid" {
				return probability * (1 - probability)
			}

			if math.IsInf(x, 0) {
				return probability
			}

			return probability + x*probability*(1-probability)
		})
		if err != nil {
			return nil, err
		}

		return Mul(derivative, gradient)
	}

	derivative, err := gpuUnary(kind+"_grad", input, 0)
	if err != nil {
		return nil, err
	}

	defer func() { _ = derivative.Close() }()

	return Mul(derivative, gradient)
}

// AddScalar adds a host scalar without transferring tensor data.
func AddScalar(input *Tensor, value float32) (*Tensor, error) {
	if err := valid("add scalar", input); err != nil {
		return nil, err
	}

	if input.gpu != nil {
		return gpuUnary("add_scalar", input, value)
	}

	out := input.Contiguous()
	for i := range out.data {
		out.data[i] += value
	}

	return out, nil
}
