package cpu

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

type array struct {
	data           []float32
	shape, strides []int
	offset, size   int
}

func (t *array) Rank() int    { return len(t.shape) }
func (t *array) Shape() []int { return slices.Clone(t.shape) }

func readView(v backend.View) (array, error) {
	b, err := checked(v.Buffer)
	if err != nil {
		return array{}, err
	}

	data, err := b.values()
	if err != nil {
		return array{}, err
	}

	if err := v.ValidateLayout(len(data), false); err != nil {
		return array{}, err
	}

	n, _ := v.Numel()

	return array{data: data, shape: v.Shape, strides: v.Strides, offset: v.Offset, size: n}, nil
}

func writeView(v backend.View, contiguous bool) (array, error) {
	a, err := readView(v)
	if err != nil {
		return array{}, err
	}

	if !a.IsContiguous() {
		if err := v.ValidateLayout(len(a.data), true); err != nil {
			return array{}, err
		}

		if contiguous {
			return array{}, fmt.Errorf("%w: CPU numerical output must be contiguous", backend.ErrUnsupported)
		}
	}

	if contiguous {
		if a.size == 0 {
			a.data = a.data[:0]
		} else {
			a.data = a.data[a.offset : a.offset+a.size]
		}

		a.offset = 0
	}

	return a, nil
}

// IsContiguous reports whether logical values occupy row-major storage.
// Singleton strides are immaterial; empty arrays are contiguous.
func (t *array) IsContiguous() bool {
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

func (t *array) index(i int) int {
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

func readInto(t *array, data []float32) error {
	if len(data) != t.size {
		return fmt.Errorf("%w: host read length mismatch", backend.ErrInvalidView)
	}

	if t.size == 0 {
		return nil
	}

	if t.IsContiguous() {
		copy(data, t.data[t.offset:t.offset+t.size])
		return nil
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

	return nil
}

func (t *array) Contiguous() *array {
	data := make([]float32, t.size)
	_ = readInto(t, data)
	strides := make([]int, t.Rank())

	stride := 1
	for axis := len(strides) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= t.shape[axis]
	}

	return &array{data: data, shape: t.shape, strides: strides, size: t.size}
}

// Read copies logical values into an exactly sized caller-owned host slice.
func (Device) Read(src backend.View, values []float32) error {
	a, err := readView(src)
	if err != nil {
		return err
	}

	return readInto(&a, values)
}

// At reads one logical element without transferring or allocating storage.
func (Device) At(src backend.View, indices []int) (float32, error) {
	a, err := readView(src)
	if err != nil {
		return 0, err
	}

	if len(indices) != a.Rank() {
		return 0, fmt.Errorf("%w: index rank mismatch", backend.ErrInvalidView)
	}

	offset := a.offset
	for axis, index := range indices {
		if index < 0 || index >= a.shape[axis] {
			return 0, fmt.Errorf("%w: index out of bounds", backend.ErrInvalidView)
		}

		offset += index * a.strides[axis]
	}

	return a.data[offset], nil
}
