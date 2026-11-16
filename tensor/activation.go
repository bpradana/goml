package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

func activationOp(kind string, derivative bool) (backend.UnaryOp, error) {
	var op backend.UnaryOp

	switch kind {
	case "relu":
		op = backend.ReLU
	case "sigmoid":
		op = backend.Sigmoid
	case "tanh":
		op = backend.Tanh
	case "silu":
		op = backend.SiLU
	default:
		return 0, fmt.Errorf("tensor: unknown activation %q", kind)
	}

	if derivative {
		op += backend.ReLUGrad - backend.ReLU
	}

	return op, nil
}

// Activation allocates relu, sigmoid, tanh, or silu on input's device.
func Activation(kind string, input *Tensor) (*Tensor, error) {
	op, err := activationOp(kind, false)
	if err != nil {
		return nil, err
	}

	return unary(op, input)
}

// ActivationBackward multiplies the input derivative by an equally shaped
// gradient. Values remain on-device; saved derivative storage is released.
func ActivationBackward(kind string, input, gradient *Tensor) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input, gradient})

	if err := valid("activation backward", input, gradient); err != nil {
		return nil, err
	}

	if err := compatible("activation backward", input, gradient); err != nil {
		return nil, err
	}

	if !slices.Equal(input.shape, gradient.shape) {
		return nil, fmt.Errorf("tensor: activation backward: shapes %v and %v differ", input.shape, gradient.shape)
	}

	op, err := activationOp(kind, true)
	if err != nil {
		return nil, err
	}

	derivative, err := unary(op, input)
	if err != nil {
		return nil, err
	}
	defer func() { _ = derivative.Close() }()

	return Mul(derivative, gradient)
}

// AddScalar allocates input plus a host scalar, preserving shape and device.
func AddScalar(input *Tensor, value float32) (*Tensor, error) {
	return scalar(backend.AddScalar, input, value)
}
