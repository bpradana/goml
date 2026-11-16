package metal

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// Fill encodes initialization of an allocation.
func (d *Device) Fill(dst backend.View, value float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	out, err := d.output(dst)
	if err != nil {
		return err
	}

	p := kernelParams{n: uint32(out.length)}
	p.scalars[0] = value

	return nativeCompute(d.handle, "fill_values", nil, nil, out.handle, p)
}

// Copy materializes logical source values into an equally shaped destination view.
func (d *Device) Copy(dst, src backend.View) error {
	if !slices.Equal(dst.Shape, src.Shape) {
		return fmt.Errorf("%w: Metal copy: shape or alias", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	out, err := d.view(dst)
	if err != nil {
		return err
	}

	if err := dst.ValidateLayout(out.length, true); err != nil {
		return err
	}

	input, err := d.view(src)
	if err != nil {
		return err
	}

	if out == input {
		return fmt.Errorf("%w: aliased Metal copy", backend.ErrInvalidView)
	}

	p := viewParams(src.Shape, src, dst)

	return nativeCompute(d.handle, "copy_values", input.handle, nil, out.handle, p)
}

func (d *Device) contiguous(v backend.View) (backend.View, func(), error) {
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

	out := backend.View{Buffer: b, Shape: v.Shape, Strides: strides}
	if err := d.Copy(out, v); err != nil {
		_ = b.Close()
		return v, nil, err
	}

	return out, func() { _ = b.Close() }, nil
}

func (d *Device) temporary(shape []int) (backend.View, func(), error) {
	count, err := elements(shape)
	if err != nil {
		return backend.View{}, nil, err
	}

	buffer, err := d.Allocate(count)
	if err != nil {
		return backend.View{}, nil, err
	}

	strides := make([]int, len(shape))

	stride := 1
	for axis := len(shape) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= shape[axis]
	}

	return backend.View{Buffer: buffer, Shape: shape, Strides: strides}, func() { _ = buffer.Close() }, nil
}
