package metal

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// unary encodes an element-wise primitive without synchronization.
func (d *Device) unary(code uint32, dst, a backend.View, scalar float32) error {
	if !slices.Equal(dst.Shape, a.Shape) {
		return fmt.Errorf("%w: Metal unary shapes differ", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	source, err := d.view(a)
	if err != nil {
		return err
	}

	output, err := d.output(dst, a)
	if err != nil {
		return err
	}

	p := viewParams(a.Shape, a, backend.View{})
	if int(p.n) != output.length {
		return fmt.Errorf("%w: metal: unary output size mismatch", backend.ErrInvalidView)
	}

	p.op = code
	p.scalars[0] = scalar

	return nativeCompute(d.handle, "unary_values", source.handle, nil, output.handle, p)
}

// Binary encodes trailing-dimension broadcast arithmetic.
func (d *Device) Binary(op backend.BinaryOp, dst, a, b backend.View) error {
	shape := dst.Shape

	code := uint32(op - 1)
	if op < backend.Add || op > backend.Div {
		return fmt.Errorf("%w: binary operation %d", backend.ErrUnsupported, op)
	}

	if _, err := elements(shape); err != nil {
		return err
	}

	for _, operand := range []backend.View{a, b} {
		if len(operand.Shape) > len(shape) {
			return fmt.Errorf("%w: metal: binary: output rank is smaller than input rank", backend.ErrInvalidView)
		}

		for axis, size := range operand.Shape {
			if size != 1 && size != shape[axis+len(shape)-len(operand.Shape)] {
				return fmt.Errorf("%w: metal: binary: incompatible broadcast shape", backend.ErrInvalidView)
			}
		}
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	av, err := d.view(a)
	if err != nil {
		return err
	}

	bv, err := d.view(b)
	if err != nil {
		return err
	}

	out, err := d.output(dst, a, b)
	if err != nil {
		return err
	}

	p := viewParams(shape, a, b)

	p.op = code
	if int(p.n) != out.length {
		return fmt.Errorf("%w: metal: binary output size mismatch", backend.ErrInvalidView)
	}

	return nativeCompute(d.handle, "binary_values", av.handle, bv.handle, out.handle, p)
}

// Unary encodes a typed element-wise operation without synchronizing.
func (d *Device) Unary(op backend.UnaryOp, dst, src backend.View) error {
	if op < backend.Exp || op > backend.SiLUGrad {
		return fmt.Errorf("%w: unary %d", backend.ErrUnsupported, op)
	}

	return d.unary(uint32(op), dst, src, 0)
}

// Scalar encodes a scalar operation without transferring tensor storage.
func (d *Device) Scalar(op backend.ScalarOp, dst, src backend.View, value float32) error {
	var code uint32

	switch op {
	case backend.Scale:
		code = 0
	case backend.AddScalar:
		code = 12
	case backend.MaskUpperTriangle:
		if len(src.Shape) != 2 {
			return backend.ErrInvalidView
		}

		code = 13
	default:
		return fmt.Errorf("%w: scalar %d", backend.ErrUnsupported, op)
	}

	return d.unary(code, dst, src, value)
}
