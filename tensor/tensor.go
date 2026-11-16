package tensor

import (
	"fmt"
	"math/rand/v2"
	"slices"

	internalshape "github.com/bpradana/goml/tensor/internal/shape"
)

// Tensor is an immutable float32 array. Obtain one through a constructor.
// Its private storage may be shared by views; methods do not mutate it.
type Tensor struct {
	gpu     *storageLease
	data    []float32
	shape   []int
	strides []int
	offset  int
	size    int
}

func allocated(shape []int) (*Tensor, error) {
	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	return &Tensor{data: make([]float32, n), shape: slices.Clone(shape), strides: strides, size: n}, nil
}

// From copies data into a row-major tensor of shape. With no dimensions it
// creates a scalar. Invalid shapes or element counts return an error.
func From(data []float32, shape ...int) (*Tensor, error) {
	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	if n != len(data) {
		return nil, fmt.Errorf("tensor: from: shape %v needs %d elements, got %d", shape, n, len(data))
	}

	return &Tensor{data: slices.Clone(data), shape: slices.Clone(shape), strides: strides, size: n}, nil
}

// Scalar allocates an independent rank-zero tensor containing value.
func Scalar(value float32) *Tensor { return &Tensor{data: []float32{value}, size: 1} }

// Zeros allocates a row-major tensor filled with zero, or returns a shape error.
func Zeros(shape ...int) (*Tensor, error) { return allocated(shape) }

// Ones allocates a row-major tensor filled with one, or returns a shape error.
func Ones(shape ...int) (*Tensor, error) {
	t, err := allocated(shape)
	if err != nil {
		return nil, err
	}

	for i := range t.data {
		t.data[i] = 1
	}

	return t, nil
}

// Rand allocates uniform samples in [0,1). The caller owns and seeds rng.
// A nil generator or invalid shape returns an error.
func Rand(rng *rand.Rand, shape ...int) (*Tensor, error) {
	if rng == nil {
		return nil, fmt.Errorf("tensor: rand: nil generator")
	}

	t, err := allocated(shape)
	if err != nil {
		return nil, err
	}

	for i := range t.data {
		t.data[i] = rng.Float32()
	}

	return t, nil
}

// Randn allocates standard normal samples using a caller-owned generator.
// A nil generator or invalid shape returns an error.
func Randn(rng *rand.Rand, shape ...int) (*Tensor, error) {
	if rng == nil {
		return nil, fmt.Errorf("tensor: randn: nil generator")
	}

	t, err := allocated(shape)
	if err != nil {
		return nil, err
	}

	for i := range t.data {
		t.data[i] = float32(rng.NormFloat64())
	}

	return t, nil
}

// Shape returns a copy of the dimensions; scalars have no dimensions.
func (t *Tensor) Shape() []int { return slices.Clone(t.shape) }

// Strides returns a copy of the storage strides, measured in elements.
func (t *Tensor) Strides() []int { return slices.Clone(t.strides) }

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int { return len(t.shape) }

// Numel returns the logical element count, including one for a scalar.
func (t *Tensor) Numel() int { return t.size }

// IsContiguous reports whether logical values occupy row-major storage.
// Singleton strides are immaterial; empty arrays are contiguous.
func (t *Tensor) IsContiguous() bool {
	if t.size == 0 {
		return true
	}

	expected := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		if t.shape[i] > 1 && t.strides[i] != expected {
			return false
		}

		expected *= t.shape[i]
	}

	return true
}

func (t *Tensor) index(i int) int {
	if t.IsContiguous() {
		return t.offset + i
	}

	p := t.offset
	for axis := len(t.shape) - 1; axis >= 0; axis-- {
		p += (i % t.shape[axis]) * t.strides[axis]
		i /= t.shape[axis]
	}

	return p
}

// Data returns an independent CPU copy of values in logical row-major order.
// GPU inputs panic; use HostData for explicit, fallible device downloads.
func (t *Tensor) Data() []float32 {
	if t.gpu != nil {
		panic("tensor: Data is CPU-only; use HostData")
	}

	data := make([]float32, t.size)
	if t.IsContiguous() {
		copy(data, t.data[t.offset:t.offset+t.size])
		return data
	}

	var local [8]int

	coordinates := local[:]
	if t.Rank() > len(local) {
		coordinates = make([]int, t.Rank())
	}

	last := t.Rank() - 1
	width, source := t.shape[last], t.offset

	for base := 0; base < t.size; base += width {
		if t.strides[last] == 1 {
			copy(data[base:base+width], t.data[source:source+width])
		} else {
			for j := range width {
				data[base+j] = t.data[source+j*t.strides[last]]
			}
		}

		for axis := last - 1; axis >= 0; axis-- {
			coordinates[axis]++

			source += t.strides[axis]
			if coordinates[axis] < t.shape[axis] {
				break
			}

			source -= coordinates[axis] * t.strides[axis]
			coordinates[axis] = 0
		}
	}

	return data
}

// At reads an element. Indices must match the rank and lie within the shape.
// It allocates no storage on success; scalar At takes no indices.
func (t *Tensor) At(indices ...int) (float32, error) {
	if err := valid("at", t); err != nil {
		return 0, err
	}

	if t.gpu != nil {
		return 0, fmt.Errorf("tensor: at: GPU storage; explicitly transfer to CPU")
	}

	if len(indices) != len(t.shape) {
		return 0, fmt.Errorf("tensor: at: indices %v do not match shape %v", indices, t.shape)
	}

	p := t.offset
	for i, x := range indices {
		if x < 0 || x >= t.shape[i] {
			return 0, fmt.Errorf("tensor: at: index %v outside shape %v", indices, t.shape)
		}

		p += x * t.strides[i]
	}

	return t.data[p], nil
}

// String returns at most eight dimensions and eight logical elements.
func (t *Tensor) String() string {
	if t == nil {
		return "tensor(<nil>)"
	}

	if t.gpu != nil {
		shape := t.shape

		suffix := ""
		if len(shape) > 8 {
			shape, suffix = shape[:8], " ..."
		}

		return fmt.Sprintf("tensor(shape=%v%s, device=%s, GPU data)", shape, suffix, t.Device().Name())
	}

	preview := make([]float32, min(8, t.size))
	for i := range preview {
		preview[i] = t.data[t.index(i)]
	}

	suffix := ""
	if t.size > len(preview) {
		suffix = " ..."
	}

	shape := t.shape
	shapeSuffix := ""

	if len(shape) > 8 {
		shape = shape[:8]
		shapeSuffix = " ..."
	}

	return fmt.Sprintf("tensor(shape=%v%s, data=%v%s)", shape, shapeSuffix, preview, suffix)
}

func valid(op string, ts ...*Tensor) error {
	for _, t := range ts {
		if t == nil || (t.gpu == nil && t.data == nil && t.shape == nil && t.size == 0) {
			return fmt.Errorf("tensor: %s: nil or uninitialized tensor", op)
		}

		if t.gpu != nil {
			if t.gpu.closed.Load() {
				return fmt.Errorf("tensor: %s: closed GPU tensor", op)
			}

			if err := t.gpu.storage.buffer.Valid(); err != nil {
				return fmt.Errorf("tensor: %s: %w", op, err)
			}
		}
	}

	return nil
}
