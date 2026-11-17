package webgpu

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// Unary writes a shape-preserving mathematical primitive or activation derivative
// on the GPU. Strided inputs and nonoverlapping strided outputs are supported.
func (d *Device) Unary(op backend.UnaryOp, dst, src backend.View) error {
	if op < backend.Exp || op > backend.SiLUGrad {
		return fmt.Errorf("webgpu: %w: unary %v", backend.ErrUnsupported, op)
	}

	if !slices.Equal(dst.Shape, src.Shape) {
		return fmt.Errorf("webgpu: unary: %w: shapes differ", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, src)
	if err != nil {
		return err
	}

	p := params(dst, src, backend.View{}, backend.View{})
	p.op = uint32(op)

	return d.dispatch("unary_values", p, handles)
}

// Binary writes trailing-dimension broadcast arithmetic without host transfers.
// Invalid shapes, foreign buffers, and overlapping destinations return errors.
func (d *Device) Binary(op backend.BinaryOp, dst, a, b backend.View) error {
	if op < backend.Add || op > backend.Div {
		return fmt.Errorf("webgpu: %w: binary %v", backend.ErrUnsupported, op)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, a, b)
	if err != nil {
		return err
	}

	p := params(dst, a, b, backend.View{})
	p.op = uint32(op)

	p.as, err = broadcastStrides(dst.Shape, a)
	if err != nil {
		return err
	}

	p.bs, err = broadcastStrides(dst.Shape, b)
	if err != nil {
		return err
	}

	return d.dispatch("binary_values", p, handles)
}

func broadcastStrides(shape []int, input backend.View) ([8]uint32, error) {
	var strides [8]uint32
	if len(input.Shape) > len(shape) {
		return strides, fmt.Errorf("webgpu: binary: %w: input rank exceeds output rank", backend.ErrInvalidView)
	}

	for axis, size := range input.Shape {
		target := axis + len(shape) - len(input.Shape)
		if size != 1 && size != shape[target] {
			return strides, fmt.Errorf("webgpu: binary: %w: incompatible broadcasting", backend.ErrInvalidView)
		}

		if size != 1 {
			strides[target] = uint32(input.Strides[axis])
		}
	}

	return strides, nil
}

// Scalar writes scaling, scalar addition, or 2D upper-triangle masking.
// The output shape must equal the input shape; no data is downloaded.
func (d *Device) Scalar(op backend.ScalarOp, dst, src backend.View, value float32) error {
	if op < backend.Scale || op > backend.MaskUpperTriangle {
		return fmt.Errorf("webgpu: %w: scalar %v", backend.ErrUnsupported, op)
	}

	if !slices.Equal(dst.Shape, src.Shape) || (op == backend.MaskUpperTriangle && len(src.Shape) != 2) {
		return fmt.Errorf("webgpu: scalar: %w: invalid shape", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, src)
	if err != nil {
		return err
	}

	p := params(dst, src, backend.View{}, backend.View{})
	p.op = uint32(op)
	p.scalars[0] = value

	return d.dispatch("scalar_values", p, handles)
}
