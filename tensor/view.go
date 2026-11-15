package tensor

import (
	"fmt"
	"slices"

	internalshape "github.com/bpradana/goml/tensor/internal/shape"
)

// Reshape returns a storage-sharing view with the requested shape.
// The input must be contiguous and element counts must match; no implicit copy
// or inferred dimensions are supported.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if err := valid("reshape", t); err != nil {
		return nil, err
	}

	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	if n != t.size || !t.IsContiguous() {
		return nil, fmt.Errorf("tensor: reshape: cannot view shape %v as %v (contiguous=%t)", t.shape, shape, t.IsContiguous())
	}

	return &Tensor{data: t.data, shape: slices.Clone(shape), strides: strides, offset: t.offset, size: n}, nil
}

// Transpose returns a storage-sharing view with axes permuted.
// With no axes it reverses the dimensions. Explicit axes must form a permutation.
func (t *Tensor) Transpose(axes ...int) (*Tensor, error) {
	if err := valid("transpose", t); err != nil {
		return nil, err
	}

	if len(axes) == 0 {
		axes = make([]int, t.Rank())
		for i := range axes {
			axes[i] = len(axes) - 1 - i
		}
	}

	if len(axes) != t.Rank() {
		return nil, fmt.Errorf("tensor: transpose: axes %v do not match shape %v", axes, t.shape)
	}

	shape, strides := make([]int, len(axes)), make([]int, len(axes))

	seen := make([]bool, len(axes))
	for i, a := range axes {
		if a < 0 || a >= len(axes) || seen[a] {
			return nil, fmt.Errorf("tensor: transpose: invalid permutation %v for %v", axes, t.shape)
		}

		seen[a] = true
		shape[i] = t.shape[a]
		strides[i] = t.strides[a]
	}

	return &Tensor{data: t.data, shape: shape, strides: strides, offset: t.offset, size: t.size}, nil
}

// Slice returns a view selecting [start,end) on axis with unit step.
// Negative axes and bounds are invalid. Empty slices are valid.
func (t *Tensor) Slice(axis, start, end int) (*Tensor, error) {
	if err := valid("slice", t); err != nil {
		return nil, err
	}

	if axis < 0 || axis >= t.Rank() || start < 0 || end < start || end > t.shape[axis] {
		return nil, fmt.Errorf("tensor: slice: axis=%d range=[%d,%d) outside shape %v", axis, start, end, t.shape)
	}

	shape := slices.Clone(t.shape)
	shape[axis] = end - start

	n, _, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	offset := t.offset
	// Empty views never dereference storage; keep their offsets within bounds.
	if n != 0 {
		offset += start * t.strides[axis]
	}

	return &Tensor{data: t.data, shape: shape, strides: slices.Clone(t.strides), offset: offset, size: n}, nil
}

// Contiguous materializes independent row-major storage in logical order,
// including when the receiver is already contiguous.
func (t *Tensor) Contiguous() *Tensor {
	_, strides, _ := internalshape.Layout(t.shape)
	return &Tensor{data: t.Data(), shape: slices.Clone(t.shape), strides: strides, size: t.size}
}
