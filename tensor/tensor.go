package tensor

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/cpu"

	internalshape "github.com/bpradana/goml/tensor/internal/shape"
)

// Tensor is an immutable float32 array. Obtain one through a constructor.
// Its private storage may be shared by views; methods do not mutate it.
type Tensor struct {
	allocation atomic.Pointer[storage]
	closeMu    sync.Mutex
	scope      atomic.Pointer[Scope]
	device     backend.Device
	shape      []int
	strides    []int
	offset     int
	size       int
}

func cpuTensor(data []float32, shape, strides []int) *Tensor {
	t, _ := attachStorage(&Tensor{device: CPU(), shape: shape, strides: strides, size: len(data)}, newStorage(cpu.Own(data)), nil)
	return t
}

func allocated(shape []int) (*Tensor, error) {
	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	if n > int(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("tensor: allocate: %w: Float32 byte count exceeds address space", backend.ErrAllocation)
	}

	return cpuTensor(make([]float32, n), slices.Clone(shape), strides), nil
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

	return cpuTensor(slices.Clone(data), slices.Clone(shape), strides), nil
}

// Scalar allocates an independent rank-zero tensor containing value.
func Scalar(value float32) *Tensor { return cpuTensor([]float32{value}, nil, nil) }

// Zeros allocates a row-major tensor filled with zero, or returns a shape error.
func Zeros(shape ...int) (*Tensor, error) { return allocated(shape) }

// Ones allocates a row-major tensor filled with one, or returns a shape error.
func Ones(shape ...int) (*Tensor, error) { return OnesOn(CPU(), shape...) }

// Rand allocates uniform samples in [0,1). The caller owns and seeds rng.
// A nil generator or invalid shape returns an error.
func Rand(rng *rand.Rand, shape ...int) (*Tensor, error) {
	if rng == nil {
		return nil, fmt.Errorf("tensor: rand: nil generator")
	}

	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	if n > int(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("tensor: allocate: %w: Float32 byte count exceeds address space", backend.ErrAllocation)
	}

	data := make([]float32, n)
	for i := range data {
		data[i] = rng.Float32()
	}

	return cpuTensor(data, slices.Clone(shape), strides), nil
}

// Randn allocates standard normal samples using a caller-owned generator.
// A nil generator or invalid shape returns an error.
func Randn(rng *rand.Rand, shape ...int) (*Tensor, error) {
	if rng == nil {
		return nil, fmt.Errorf("tensor: randn: nil generator")
	}

	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	if n > int(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("tensor: allocate: %w: Float32 byte count exceeds address space", backend.ErrAllocation)
	}

	data := make([]float32, n)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}

	return cpuTensor(data, slices.Clone(shape), strides), nil
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

// At reads an element. Indices must match the rank and lie within the shape.
// It allocates no storage on success; scalar At takes no indices.
func (t *Tensor) At(indices ...int) (float32, error) {
	if err := valid("at", t); err != nil {
		return 0, err
	}

	host, ok := t.device.(backend.HostAccess)
	if !ok {
		return 0, unsupported("at: explicitly read HostData", t.device)
	}
	defer runtime.KeepAlive(t)

	return host.At(t.deviceView(), indices)
}

// String returns bounded shape and device metadata without host reads or synchronization.
func (t *Tensor) String() string {
	if t == nil {
		return "tensor(<nil>)"
	}

	shape, suffix := t.shape, ""
	if len(shape) > 8 {
		shape, suffix = shape[:8], " ..."
	}

	name := "uninitialized"
	if t.device != nil {
		name = t.device.Name()
	}

	return fmt.Sprintf("tensor(shape=%v%s, device=%s, elements=%d)", shape, suffix, name, t.size)
}

func valid(op string, ts ...*Tensor) error {
	for _, t := range ts {
		if t == nil || t.device == nil {
			return fmt.Errorf("tensor: %s: nil or uninitialized tensor", op)
		}

		allocation := t.allocation.Load()
		if allocation == nil {
			return fmt.Errorf("tensor: %s: %w", op, backend.ErrClosed)
		}

		if err := allocation.buffer.Valid(); err != nil {
			return fmt.Errorf("tensor: %s: %w", op, err)
		}
	}

	return nil
}
