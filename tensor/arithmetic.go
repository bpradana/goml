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

		return out, binaryContiguous(op, out.data, av, bv)
	}

	if out.size == 0 {
		return out, nil
	}

	var local [24]int

	work := local[:]
	if len(shape) > 8 {
		work = make([]int, 3*len(shape))
	}

	rank := len(shape)
	coordinates, as, bs := work[:rank], work[rank:2*rank], work[2*rank:3*rank]
	broadcastStrides(a, shape, as)
	broadcastStrides(b, shape, bs)

	width, astep, bstep := 1, 0, 0

	if rank > 0 {
		width, astep, bstep = shape[rank-1], as[rank-1], bs[rank-1]
	}

	ap, bp := a.offset, b.offset

	for base := 0; base < out.size; base += width {
		ai, bi := ap, bp
		if astep == 1 && bstep == 1 {
			err = binaryContiguous(op, out.data[base:base+width], a.data[ai:ai+width], b.data[bi:bi+width])
			if err != nil {
				return nil, err
			}
		} else {
			for j := range width {
				x, y := a.data[ai], b.data[bi]

				switch op {
				case "add":
					out.data[base+j] = x + y
				case "sub":
					out.data[base+j] = x - y
				case "mul":
					out.data[base+j] = x * y
				case "div":
					out.data[base+j] = x / y
				}

				ai += astep
				bi += bstep
			}
		}

		for axis := rank - 2; axis >= 0; axis-- {
			coordinates[axis]++
			ap += as[axis]

			bp += bs[axis]
			if coordinates[axis] < shape[axis] {
				break
			}

			ap -= coordinates[axis] * as[axis]
			bp -= coordinates[axis] * bs[axis]
			coordinates[axis] = 0
		}
	}

	return out, nil
}

func binaryContiguous(op string, dst, a, b []float32) error {
	switch op {
	case "add":
		return simdops.Add(dst, a, b)
	case "sub":
		return simdops.Sub(dst, a, b)
	case "mul":
		return simdops.Mul(dst, a, b)
	default:
		return simdops.Div(dst, a, b)
	}
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
