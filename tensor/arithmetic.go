package tensor

import (
	"fmt"
	"math"
	"slices"

	internalshape "github.com/bpradana/goml/tensor/internal/shape"

	"github.com/bpradana/goml/simdops"
)

func binary(op string, a, b *Tensor) (*Tensor, error) {
	if err := valid(op, a, b); err != nil {
		return nil, err
	}

	shape, err := internalshape.Broadcast(a.shape, b.shape)
	if err != nil {
		return nil, fmt.Errorf("tensor: %s: %w", op, err)
	}

	out, err := allocated(shape)
	if err != nil {
		return nil, err
	}

	if slices.Equal(a.shape, b.shape) && a.IsContiguous() && b.IsContiguous() {
		av, bv := a.data[a.offset:a.offset+a.size], b.data[b.offset:b.offset+b.size]

		switch op {
		case "add":
			err = simdops.Add(out.data, av, bv)
		case "sub":
			err = simdops.Sub(out.data, av, bv)
		case "mul":
			err = simdops.Mul(out.data, av, bv)
		case "div":
			err = simdops.Div(out.data, av, bv)
		}

		return out, err
	}

	for i := range out.data {
		x, y := a.data[broadcastIndex(a, shape, i)], b.data[broadcastIndex(b, shape, i)]

		switch op {
		case "add":
			out.data[i] = x + y
		case "sub":
			out.data[i] = x - y
		case "mul":
			out.data[i] = x * y
		case "div":
			out.data[i] = x / y
		}
	}

	return out, nil
}

// Add allocates the element-wise sum using trailing-dimension broadcasting.
// Incompatible shapes or uninitialized inputs return an error.
func Add(a, b *Tensor) (*Tensor, error) { return binary("add", a, b) }

// Sub allocates a-b using trailing-dimension broadcasting, or returns a shape error.
func Sub(a, b *Tensor) (*Tensor, error) { return binary("sub", a, b) }

// Mul allocates a*b using trailing-dimension broadcasting, or returns a shape error.
func Mul(a, b *Tensor) (*Tensor, error) { return binary("mul", a, b) }

// Div allocates a/b with broadcasting. Zero division follows IEEE 754;
// incompatible shapes return an error.
func Div(a, b *Tensor) (*Tensor, error) { return binary("div", a, b) }

// Scale allocates a tensor with a's shape and values multiplied by factor.
// Uninitialized inputs return an error; NaNs and infinities propagate.
func Scale(a *Tensor, factor float32) (*Tensor, error) {
	if err := valid("scale", a); err != nil {
		return nil, err
	}

	out := a.Contiguous()

	return out, simdops.Scale(out.data, out.data, factor)
}

// Neg allocates -a with the same shape; invalid inputs return an error.
func Neg(a *Tensor) (*Tensor, error) { return Scale(a, -1) }

func unary(op string, a *Tensor, f func(float64) float64) (*Tensor, error) {
	if err := valid(op, a); err != nil {
		return nil, err
	}

	out := a.Contiguous()
	for i, x := range out.data {
		out.data[i] = float32(f(float64(x)))
	}

	return out, nil
}

// Exp allocates exp(a) with the same shape. Overflow produces infinity;
// invalid inputs return an error.
func Exp(a *Tensor) (*Tensor, error) { return unary("exp", a, math.Exp) }

// Log allocates the natural logarithm. Zero yields -Inf, negatives yield NaN;
// invalid inputs return an error.
func Log(a *Tensor) (*Tensor, error) { return unary("log", a, math.Log) }

// Sqrt allocates sqrt(a). Negative elements yield NaN;
// invalid inputs return an error.
func Sqrt(a *Tensor) (*Tensor, error) { return unary("sqrt", a, math.Sqrt) }
