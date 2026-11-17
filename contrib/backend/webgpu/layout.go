package webgpu

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// Copy copies logical values between equal-shaped views on the GPU, including
// transposed materialization and strided scatter. Source/destination aliasing is
// rejected. No host transfer is performed.
func (d *Device) Copy(dst, src backend.View) error {
	if !slices.Equal(dst.Shape, src.Shape) {
		return fmt.Errorf("webgpu: copy: %w: shapes differ", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, src)
	if err != nil {
		return err
	}

	return d.dispatch("copy_values", params(dst, src, backend.View{}, backend.View{}), handles)
}

// Fill initializes every logical destination element. Strided nonoverlapping
// views and empty destinations are supported; invalid views return errors.
func (d *Device) Fill(dst backend.View, value float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst)
	if err != nil {
		return err
	}

	p := params(dst, backend.View{}, backend.View{}, backend.View{})
	p.scalars[0] = value

	return d.dispatch("fill_values", p, handles)
}

// Concat joins equally ranked inputs along axis using device-resident copies.
// All input shapes and buffers are validated before encoding any write.
func (d *Device) Concat(dst backend.View, axis int, inputs []backend.View) error {
	if len(inputs) == 0 || axis < 0 || axis >= len(dst.Shape) {
		return fmt.Errorf("webgpu: concat: %w: invalid axis or missing inputs", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	output, err := d.view(dst, true)
	if err != nil {
		return err
	}

	joined := 0

	for _, input := range inputs {
		b, err := d.view(input, false)
		if err != nil {
			return err
		}

		if b == output || len(input.Shape) != len(dst.Shape) {
			return fmt.Errorf("webgpu: concat: %w: alias or rank mismatch", backend.ErrInvalidView)
		}

		for i, size := range input.Shape {
			if i != axis && size != dst.Shape[i] {
				return fmt.Errorf("webgpu: concat: %w: dimensions differ", backend.ErrInvalidView)
			}
		}

		if input.Shape[axis] > dst.Shape[axis]-joined {
			return fmt.Errorf("webgpu: concat: %w: joined axis exceeds output", backend.ErrInvalidView)
		}

		joined += input.Shape[axis]
	}

	if joined != dst.Shape[axis] {
		return fmt.Errorf("webgpu: concat: %w: output axis mismatch", backend.ErrInvalidView)
	}

	start := 0

	for _, input := range inputs {
		target := dst
		target.Shape = slices.Clone(dst.Shape)
		target.Shape[axis] = input.Shape[axis]
		target.Offset += start * dst.Strides[axis]

		handles, err := d.bindings(target, input)
		if err != nil {
			return err
		}

		if err := d.dispatch("copy_values", params(target, input, backend.View{}, backend.View{}), handles); err != nil {
			return err
		}

		start += input.Shape[axis]
	}

	return nil
}
