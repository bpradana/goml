package cpu

import (
	"fmt"
	"math"
	"slices"

	"github.com/bpradana/goml/simdops"
	"github.com/bpradana/goml/tensor/backend"
)

// Binary computes trailing-dimension broadcast arithmetic into contiguous dst.
func (Device) Binary(op backend.BinaryOp, dst, left, right backend.View) error {
	if op < backend.Add || op > backend.Div {
		return fmt.Errorf("%w: binary operation %d", backend.ErrUnsupported, op)
	}

	out, err := writeView(dst, true)
	if err != nil {
		return err
	}

	a, err := readView(left)
	if err != nil {
		return err
	}

	b, err := readView(right)
	if err != nil {
		return err
	}

	if dst.Buffer == left.Buffer || dst.Buffer == right.Buffer {
		return fmt.Errorf("%w: aliased output", backend.ErrInvalidView)
	}

	shape := out.shape
	for _, input := range []array{a, b} {
		if input.Rank() > out.Rank() {
			return backend.ErrInvalidView
		}

		leading := out.Rank() - input.Rank()
		for axis, n := range input.shape {
			if n != 1 && n != shape[leading+axis] {
				return backend.ErrInvalidView
			}
		}
	}

	if slices.Equal(a.shape, b.shape) && slices.Equal(a.shape, out.shape) && a.IsContiguous() && b.IsContiguous() {
		if out.size == 0 {
			return nil
		}

		av, bv := a.data[a.offset:a.offset+a.size], b.data[b.offset:b.offset+b.size]

		return binaryContiguous(op, out.data, av, bv)
	}

	if out.size == 0 {
		return nil
	}

	var local [24]int

	work := local[:]
	if len(shape) > 8 {
		work = make([]int, 3*len(shape))
	}

	rank := len(shape)
	coordinates, as, bs := work[:rank], work[rank:2*rank], work[2*rank:3*rank]
	broadcastStrides(&a, shape, as)
	broadcastStrides(&b, shape, bs)

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
				return err
			}
		} else {
			for j := range width {
				x, y := a.data[ai], b.data[bi]

				switch op {
				case backend.Add:
					out.data[base+j] = x + y
				case backend.Sub:
					out.data[base+j] = x - y
				case backend.Mul:
					out.data[base+j] = x * y
				case backend.Div:
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

	return nil
}

func binaryContiguous(op backend.BinaryOp, dst, a, b []float32) error {
	switch op {
	case backend.Add:
		return simdops.Add(dst, a, b)
	case backend.Sub:
		return simdops.Sub(dst, a, b)
	case backend.Mul:
		return simdops.Mul(dst, a, b)
	default:
		return simdops.Div(dst, a, b)
	}
}

// broadcastStrides aligns input strides with output axes. Broadcast dimensions
// have a zero stride so the arithmetic loop can advance without division.
func broadcastStrides(t *array, shape []int, strides []int) {
	leading := len(shape) - t.Rank()
	for axis, size := range t.shape {
		if size != 1 {
			strides[leading+axis] = t.strides[axis]
		}
	}
}

// Scalar preserves shape while applying a scalar primitive to logical values.
func (Device) Scalar(op backend.ScalarOp, dst, src backend.View, value float32) error {
	if op < backend.Scale || op > backend.MaskUpperTriangle {
		return fmt.Errorf("%w: scalar operation %d", backend.ErrUnsupported, op)
	}

	out, err := writeView(dst, true)
	if err != nil {
		return err
	}

	a, err := readView(src)
	if err != nil {
		return err
	}

	if !slices.Equal(out.shape, a.shape) || dst.Buffer == src.Buffer {
		return fmt.Errorf("%w: scalar shape or alias", backend.ErrInvalidView)
	}

	if op == backend.MaskUpperTriangle && a.Rank() != 2 {
		return fmt.Errorf("%w: triangle requires rank two", backend.ErrUnsupported)
	}

	if out.size == 0 {
		return nil
	}

	if op == backend.Scale && a.IsContiguous() {
		return simdops.Scale(out.data, a.data[a.offset:a.offset+a.size], value)
	}

	for i := range out.data {
		x := a.data[a.index(i)]

		switch op {
		case backend.Scale:
			out.data[i] = x * value
		case backend.AddScalar:
			out.data[i] = x + value
		case backend.MaskUpperTriangle:
			if i%a.shape[1] > i/a.shape[1] {
				out.data[i] = value
			} else {
				out.data[i] = x
			}
		}
	}

	return nil
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}

	e := math.Exp(x)

	return e / (1 + e)
}

// Unary applies a typed mathematical primitive, preserving logical shape.
func (Device) Unary(op backend.UnaryOp, dst, src backend.View) error {
	var function func(float64) float64

	switch op {
	case backend.Exp:
		function = math.Exp
	case backend.Log:
		function = math.Log
	case backend.Sqrt:
		function = math.Sqrt
	case backend.ReLU:
		function = func(x float64) float64 { return math.Max(0, x) }
	case backend.Sigmoid:
		function = sigmoid
	case backend.Tanh:
		function = math.Tanh
	case backend.SiLU:
		function = func(x float64) float64 {
			if math.IsInf(x, -1) {
				return math.Copysign(0, -1)
			}

			return x * sigmoid(x)
		}
	case backend.ReLUGrad:
		function = func(x float64) float64 {
			if math.IsNaN(x) {
				return x
			}

			if x > 0 {
				return 1
			}

			return 0
		}
	case backend.SigmoidGrad:
		function = func(x float64) float64 { p := sigmoid(x); return p * (1 - p) }
	case backend.TanhGrad:
		function = func(x float64) float64 { p := math.Tanh(x); return 1 - p*p }
	case backend.SiLUGrad:
		function = func(x float64) float64 {
			p := sigmoid(x)
			if math.IsInf(x, 0) {
				return p
			}

			return p + x*p*(1-p)
		}
	default:
		return fmt.Errorf("%w: unary operation %d", backend.ErrUnsupported, op)
	}

	out, err := writeView(dst, true)
	if err != nil {
		return err
	}

	a, err := readView(src)
	if err != nil {
		return err
	}

	if !slices.Equal(out.shape, a.shape) || dst.Buffer == src.Buffer {
		return fmt.Errorf("%w: unary shape or alias", backend.ErrInvalidView)
	}

	if out.size == 0 {
		return nil
	}

	if a.IsContiguous() {
		for i, x := range a.data[a.offset : a.offset+a.size] {
			out.data[i] = float32(function(float64(x)))
		}
	} else {
		for i := range out.data {
			out.data[i] = float32(function(float64(a.data[a.index(i)])))
		}
	}

	return nil
}
