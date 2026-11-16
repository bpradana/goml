package cpu

import (
	"fmt"
	"math"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// Fill initializes a validated nonoverlapping logical destination.
func (Device) Fill(dst backend.View, value float32) error {
	out, err := writeView(dst, false)
	if err != nil {
		return err
	}

	if out.IsContiguous() {
		for i := range out.size {
			out.data[out.offset+i] = value
		}
	} else {
		for i := range out.size {
			out.data[out.index(i)] = value
		}
	}

	return nil
}

// Copy materializes or scatters equally shaped values without aliasing inputs.
func (Device) Copy(dst, src backend.View) error {
	out, err := writeView(dst, false)
	if err != nil {
		return err
	}

	a, err := readView(src)
	if err != nil {
		return err
	}

	if !slices.Equal(out.shape, a.shape) {
		return fmt.Errorf("%w: copy shapes differ", backend.ErrInvalidView)
	}

	if dst.Buffer == src.Buffer {
		return fmt.Errorf("%w: overlapping copy allocations", backend.ErrInvalidView)
	}

	if a.size == 0 {
		return nil
	}

	if out.IsContiguous() {
		return readInto(&a, out.data[out.offset:out.offset+out.size])
	}

	if a.IsContiguous() {
		for i, x := range a.data[a.offset : a.offset+a.size] {
			out.data[out.index(i)] = x
		}
	} else {
		for i := range a.size {
			out.data[out.index(i)] = a.data[a.index(i)]
		}
	}

	return nil
}

// Concat joins equally ranked inputs along axis without allocating tensor views.
func (Device) Concat(dst backend.View, axis int, inputs []backend.View) error {
	out, err := writeView(dst, true)
	if err != nil {
		return err
	}

	if axis < 0 || axis >= out.Rank() {
		return fmt.Errorf("%w: concat axis", backend.ErrInvalidView)
	}

	inner := out.strides[axis]

	outer := 0
	if out.size > 0 {
		outer = out.size / (out.shape[axis] * inner)
	}

	offset := 0

	for _, view := range inputs {
		input, err := readView(view)
		if err != nil {
			return err
		}

		if view.Buffer == dst.Buffer {
			return fmt.Errorf("%w: aliased concat", backend.ErrInvalidView)
		}

		if input.Rank() != out.Rank() {
			return fmt.Errorf("%w: concat rank", backend.ErrInvalidView)
		}

		for i, n := range input.shape {
			if i != axis && n != out.shape[i] {
				return backend.ErrInvalidView
			}
		}

		width := input.shape[axis] * inner
		if width > out.shape[axis]*inner-offset {
			return backend.ErrInvalidView
		}

		contiguous := input.IsContiguous()

		for row := range outer {
			target := out.data[row*out.shape[axis]*inner+offset:][:width]
			if contiguous {
				start := input.offset + row*width
				copy(target, input.data[start:start+width])
			} else {
				for j := range target {
					target[j] = input.data[input.index(row*width+j)]
				}
			}
		}

		offset += width
	}

	if offset != out.shape[axis]*inner {
		return backend.ErrInvalidView
	}

	return nil
}

// CheckFinite rejects NaN and infinity in logical element order.
func (Device) CheckFinite(src backend.View) error {
	a, err := readView(src)
	if err != nil {
		return err
	}

	for i := range a.size {
		x := float64(a.data[a.index(i)])
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return fmt.Errorf("cpu: nonfinite value at logical index %d", i)
		}
	}

	return nil
}
