package metal

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
)

type kernelParams struct {
	n, rank, op, flags uint32
	shape, as, bs      [8]uint32
	ao, bo             uint32
	ints               [8]uint32
	scalars            [4]float32
}

func elements(shape []int) (int, error) {
	if len(shape) > 8 {
		return 0, fmt.Errorf("metal: rank exceeds eight")
	}

	empty := false

	for _, size := range shape {
		if size < 0 || uint64(size) > uint64(^uint32(0)) {
			return 0, fmt.Errorf("metal: unsupported dimension %d", size)
		}

		empty = empty || size == 0
	}

	if empty {
		return 0, nil
	}

	count := uint64(1)
	for _, size := range shape {
		if count > uint64(^uint32(0))/uint64(size) {
			return 0, fmt.Errorf("metal: element count exceeds 32-bit indexing")
		}

		count *= uint64(size)
	}

	return int(count), nil
}

func (d *Device) view(v tensor.DeviceView) (*buffer, error) {
	b, err := d.checked(v.Buffer)
	if err != nil {
		return nil, err
	}

	count, err := elements(v.Shape)
	if err != nil {
		return nil, err
	}

	if len(v.Shape) != len(v.Strides) || v.Offset < 0 || uint64(v.Offset) > uint64(^uint32(0)) {
		return nil, fmt.Errorf("metal: invalid view or rank exceeds eight")
	}

	maximum := uint64(v.Offset)
	for axis, size := range v.Shape {
		if size < 0 || v.Strides[axis] < 0 || uint64(size) > uint64(^uint32(0)) || uint64(v.Strides[axis]) > uint64(^uint32(0)) {
			return nil, fmt.Errorf("metal: unsupported shape or strides")
		}

		if count != 0 {
			distance := uint64(size-1) * uint64(v.Strides[axis])
			if maximum >= uint64(b.length) || distance >= uint64(b.length)-maximum {
				return nil, fmt.Errorf("metal: view exceeds buffer")
			}

			maximum += distance
		}
	}

	if count != 0 && maximum >= uint64(b.length) {
		return nil, fmt.Errorf("metal: view exceeds buffer")
	}

	return b, nil
}

func viewParams(shape []int, a, b tensor.DeviceView) kernelParams {
	p := kernelParams{n: 1, rank: uint32(len(shape)), ao: uint32(a.Offset), bo: uint32(b.Offset)}
	for axis, size := range shape {
		p.n *= uint32(size)
		p.shape[axis] = uint32(size)

		ai := axis - len(shape) + len(a.Shape)
		if ai >= 0 && a.Shape[ai] != 1 {
			p.as[axis] = uint32(a.Strides[ai])
		}

		bi := axis - len(shape) + len(b.Shape)
		if bi >= 0 && b.Shape[bi] != 1 {
			p.bs[axis] = uint32(b.Strides[bi])
		}
	}

	return p
}

// Unary encodes an element-wise numerical operation without synchronization.
func (d *Device) Unary(op string, dst tensor.Buffer, a tensor.DeviceView, scalar float32) error {
	codes := map[string]uint32{"scale": 0, "exp": 1, "log": 2, "sqrt": 3, "relu": 4, "sigmoid": 5, "tanh": 6, "silu": 7, "relu_grad": 8, "sigmoid_grad": 9, "tanh_grad": 10, "silu_grad": 11, "add_scalar": 12, "causal": 13}

	code, ok := codes[op]
	if !ok {
		return fmt.Errorf("metal: unsupported unary %q", op)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	source, err := d.view(a)
	if err != nil {
		return err
	}

	output, err := d.checked(dst)
	if err != nil {
		return err
	}

	p := viewParams(a.Shape, a, tensor.DeviceView{})
	if int(p.n) != output.length {
		return fmt.Errorf("metal: unary output size mismatch")
	}

	p.op = code
	p.scalars[0] = scalar

	return nativeCompute(d.handle, "unary_values", source.handle, nil, output.handle, p)
}

// Binary encodes trailing-dimension broadcast arithmetic.
func (d *Device) Binary(op string, dst tensor.Buffer, a, b tensor.DeviceView, shape []int) error {
	codes := map[string]uint32{"add": 0, "sub": 1, "mul": 2, "div": 3}

	code, ok := codes[op]
	if !ok || len(shape) > 8 {
		return fmt.Errorf("metal: unsupported binary operation or rank")
	}

	if _, err := elements(shape); err != nil {
		return err
	}

	for _, operand := range []tensor.DeviceView{a, b} {
		if len(operand.Shape) > len(shape) {
			return fmt.Errorf("metal: binary: output rank is smaller than input rank")
		}

		for axis, size := range operand.Shape {
			if size != 1 && size != shape[axis+len(shape)-len(operand.Shape)] {
				return fmt.Errorf("metal: binary: incompatible broadcast shape")
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

	out, err := d.checked(dst)
	if err != nil {
		return err
	}

	p := viewParams(shape, a, b)

	p.op = code
	if int(p.n) != out.length {
		return fmt.Errorf("metal: binary output size mismatch")
	}

	return nativeCompute(d.handle, "binary_values", av.handle, bv.handle, out.handle, p)
}

// Fill encodes initialization of an allocation.
func (d *Device) Fill(dst tensor.Buffer, value float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	out, err := d.checked(dst)
	if err != nil {
		return err
	}

	p := kernelParams{n: uint32(out.length)}
	p.scalars[0] = value

	return nativeCompute(d.handle, "fill_values", nil, nil, out.handle, p)
}

// Copy materializes logical source values into an equally shaped destination view.
func (d *Device) Copy(dst, src tensor.DeviceView) error {
	if !slices.Equal(dst.Shape, src.Shape) {
		return fmt.Errorf("metal: copy: shapes must match")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	out, err := d.view(dst)
	if err != nil {
		return err
	}

	input, err := d.view(src)
	if err != nil {
		return err
	}

	p := viewParams(src.Shape, src, dst)

	return nativeCompute(d.handle, "copy_values", input.handle, nil, out.handle, p)
}

// Reduce encodes a per-output reduction over selected axes.
func (d *Device) Reduce(op string, dst tensor.Buffer, a tensor.DeviceView, axes []int) error {
	codes := map[string]uint32{"sum": 0, "mean": 1, "max": 2}

	code, ok := codes[op]
	if !ok {
		return fmt.Errorf("metal: unsupported reduction")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	input, err := d.view(a)
	if err != nil {
		return err
	}

	output, err := d.checked(dst)
	if err != nil {
		return err
	}

	p := viewParams(a.Shape, a, tensor.DeviceView{})
	p.n = uint32(output.length)

	p.op = code

	count, reduced := 1, uint32(0)
	for _, axis := range axes {
		if axis < 0 || axis >= len(a.Shape) || reduced&(1<<axis) != 0 {
			return fmt.Errorf("metal: reduction: invalid or repeated axis")
		}

		reduced |= 1 << axis
		p.flags |= 1 << axis
	}

	for axis, size := range a.Shape {
		if reduced&(1<<axis) == 0 {
			count *= size
		} else if size == 0 && op != "sum" {
			return fmt.Errorf("metal: reduction: empty mean or max")
		}
	}

	if count != output.length {
		return fmt.Errorf("metal: reduction: output size mismatch")
	}

	return nativeCompute(d.handle, "reduce_values", input.handle, nil, output.handle, p)
}

// MatMul uses MPSMatrixMultiplication. Strided operands are first materialized
// on-device; native matrix inputs persist across calls without host transfers.
func (d *Device) MatMul(dst tensor.Buffer, a, b tensor.DeviceView) error {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return fmt.Errorf("metal: matmul: expected two matrices")
	}

	av, releaseA, err := d.contiguous(a)
	if err != nil {
		return err
	}
	defer releaseA()

	bv, releaseB, err := d.contiguous(b)
	if err != nil {
		return err
	}
	defer releaseB()

	d.mu.Lock()
	defer d.mu.Unlock()

	aa, err := d.view(av)
	if err != nil {
		return err
	}

	bb, err := d.view(bv)
	if err != nil {
		return err
	}

	out, err := d.checked(dst)
	if err != nil {
		return err
	}

	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	if out.length != m*n || b.Shape[0] != k {
		return fmt.Errorf("metal: matmul shape mismatch")
	}

	if m == 0 || n == 0 || k == 0 {
		return nativeCompute(d.handle, "fill_values", nil, nil, out.handle, kernelParams{n: uint32(out.length)})
	}

	args := [9]uint32{uint32(m), uint32(k), uint32(n), uint32(av.Offset), uint32(bv.Offset), uint32(k), uint32(n)}

	return nativeMatMul(d.handle, aa.handle, bb.handle, out.handle, args, false, false)
}

func (d *Device) contiguous(v tensor.DeviceView) (tensor.DeviceView, func(), error) {
	d.mu.Lock()
	_, err := d.view(v)
	d.mu.Unlock()

	if err != nil {
		return v, nil, err
	}

	size, stride := 1, 1
	contiguous := true

	for axis := len(v.Shape) - 1; axis >= 0; axis-- {
		if v.Shape[axis] > 1 && v.Strides[axis] != stride {
			contiguous = false
		}

		stride *= v.Shape[axis]
		size *= v.Shape[axis]
	}

	if contiguous && v.Offset == 0 {
		return v, func() {}, nil
	}

	b, err := d.Allocate(size)
	if err != nil {
		return v, nil, err
	}

	strides := make([]int, len(v.Shape))

	stride = 1
	for axis := len(strides) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= v.Shape[axis]
	}

	out := tensor.DeviceView{Buffer: b, Shape: v.Shape, Strides: strides}
	if err := d.Copy(out, v); err != nil {
		_ = b.Close()
		return v, nil, err
	}

	return out, func() { _ = b.Close() }, nil
}
