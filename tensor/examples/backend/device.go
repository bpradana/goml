// Package examplebackend is a minimal, independently implementable backend.
// It uses ordinary Go memory and implements only MatMul. It imports goml's
// public contracts, never Tensor internals or a concrete backend.
package examplebackend

import (
	"crypto/rand"
	"fmt"
	"sync"

	"github.com/bpradana/goml/tensor/backend"
)

// Device owns synchronous Float32 allocations. Methods are safe for concurrent
// callers. Close invalidates all buffers; repeated Close is harmless.
type Device struct {
	mu      sync.Mutex
	id      string
	buffers map[*buffer]struct{}
	closed  bool
}

// Open creates an independent execution context with no global registration.
func Open() (*Device, error) {
	var identity [16]byte
	if _, err := rand.Read(identity[:]); err != nil {
		return nil, err
	}

	return &Device{id: fmt.Sprintf("example:%x", identity), buffers: make(map[*buffer]struct{})}, nil
}

// ID returns a stable identity for this context.
func (d *Device) ID() string { return d.id }

// Name returns a human-readable label; dispatch never uses it.
func (*Device) Name() string { return "Example Go memory" }

// Allocate creates one private allocation measured in Float32 elements.
func (d *Device) Allocate(elements int) (backend.Buffer, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return nil, backend.ErrClosed
	}

	if elements < 0 || elements > int(^uint(0)>>1)/4 {
		return nil, backend.ErrAllocation
	}

	b := &buffer{owner: d, data: make([]float32, elements), length: elements}
	d.buffers[b] = struct{}{}

	return b, nil
}

func (d *Device) checked(value backend.Buffer) (*buffer, error) {
	if d.closed {
		return nil, backend.ErrClosed
	}

	b, ok := value.(*buffer)
	if !ok || b == nil || b.owner != d {
		return nil, backend.ErrDeviceMismatch
	}

	if b.closed {
		return nil, backend.ErrClosed
	}

	return b, nil
}

// Upload copies exactly one allocation without retaining the caller's slice.
func (d *Device) Upload(dst backend.Buffer, values []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	b, err := d.checked(dst)
	if err != nil {
		return err
	}

	if len(values) != b.length {
		return backend.ErrInvalidView
	}

	copy(b.data, values)

	return nil
}

// Download copies one allocation into caller-owned memory.
func (d *Device) Download(src backend.Buffer, values []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	b, err := d.checked(src)
	if err != nil {
		return err
	}

	if len(values) != b.length {
		return backend.ErrInvalidView
	}

	copy(values, b.data)

	return nil
}

// Synchronize reports liveness; all example operations finish synchronously.
func (d *Device) Synchronize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return backend.ErrClosed
	}

	return nil
}

// Close invalidates all allocations and waits for any synchronous caller.
func (d *Device) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return nil
	}

	d.closed = true
	for b := range d.buffers {
		b.closed = true
		b.data = nil
	}

	d.buffers = nil

	return nil
}

var _ backend.Device = (*Device)(nil)
