package tensor

import (
	"fmt"
	"runtime"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/shape"
)

func binary(op backend.BinaryOp, a, b *Tensor) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{a, b})

	if err := valid("binary", a, b); err != nil {
		return nil, err
	}

	if err := compatible("binary", a, b); err != nil {
		return nil, err
	}

	dimensions, err := shape.Broadcast(a.shape, b.shape)
	if err != nil {
		return nil, fmt.Errorf("tensor: binary: %w", err)
	}

	compute, ok := a.device.(backend.BinaryBackend)
	if !ok {
		return nil, unsupported("binary", a.device)
	}

	out, err := allocatedOn(dimensions, a, b)
	if err != nil {
		return nil, err
	}

	if err := compute.Binary(op, out.deviceView(), a.deviceView(), b.deviceView()); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func unary(op backend.UnaryOp, a *Tensor) (*Tensor, error) {
	defer runtime.KeepAlive(a)

	if err := valid("unary", a); err != nil {
		return nil, err
	}

	compute, ok := a.device.(backend.UnaryBackend)
	if !ok {
		return nil, unsupported("unary", a.device)
	}

	out, err := allocatedOn(a.shape, a)
	if err != nil {
		return nil, err
	}

	if err := compute.Unary(op, out.deviceView(), a.deviceView()); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func scalar(op backend.ScalarOp, a *Tensor, value float32) (*Tensor, error) {
	defer runtime.KeepAlive(a)

	if err := valid("scalar", a); err != nil {
		return nil, err
	}

	compute, ok := a.device.(backend.ScalarBackend)
	if !ok {
		return nil, unsupported("scalar", a.device)
	}

	out, err := allocatedOn(a.shape, a)
	if err != nil {
		return nil, err
	}

	if err := compute.Scalar(op, out.deviceView(), a.deviceView(), value); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

// Add allocates the trailing-dimension broadcast sum on the input device.
func Add(a, b *Tensor) (*Tensor, error) { return binary(backend.Add, a, b) }

// Sub allocates the trailing-dimension broadcast difference on the input device.
func Sub(a, b *Tensor) (*Tensor, error) { return binary(backend.Sub, a, b) }

// Mul allocates the trailing-dimension broadcast product on the input device.
func Mul(a, b *Tensor) (*Tensor, error) { return binary(backend.Mul, a, b) }

// Div allocates IEEE Float32 broadcast division on the input device.
func Div(a, b *Tensor) (*Tensor, error) { return binary(backend.Div, a, b) }

// Scale allocates a times factor on a's device, preserving shape.
func Scale(a *Tensor, factor float32) (*Tensor, error) { return scalar(backend.Scale, a, factor) }

// Neg allocates the element-wise negation of a on its device.
func Neg(a *Tensor) (*Tensor, error) { return Scale(a, -1) }

// Exp allocates the element-wise exponential, preserving shape and device.
func Exp(a *Tensor) (*Tensor, error) { return unary(backend.Exp, a) }

// Log allocates the element-wise natural logarithm, with IEEE special values.
func Log(a *Tensor) (*Tensor, error) { return unary(backend.Log, a) }

// Sqrt allocates element-wise square roots, with IEEE special values.
func Sqrt(a *Tensor) (*Tensor, error) { return unary(backend.Sqrt, a) }
