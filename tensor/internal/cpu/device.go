// Package cpu implements goml's host-native backend with the existing scalar,
// SIMD, blocked matrix, and tiled convolution kernels. It does not import Tensor.
package cpu

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

// Device is the shared, stateless CPU execution context. Close is a no-op.
// Each buffer owns its allocation independently. Operations are synchronous.
type Device struct{}

// ID is the identity of the one shared CPU context.
const ID = "cpu"

// ID returns the identity of the shared stateless CPU execution context.
func (Device) ID() string { return ID }

// Name returns the human-readable device label.
func (Device) Name() string { return "CPU" }

// Synchronize succeeds because CPU operations finish before returning.
func (Device) Synchronize() error { return nil }

// Close is harmless for the shared CPU context; buffers retain independent lifetimes.
func (Device) Close() error { return nil }

// Allocate creates private Go-managed Float32 storage or reports an invalid count.
func (Device) Allocate(elements int) (backend.Buffer, error) {
	if elements < 0 || elements > int(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("%w: invalid CPU element count %d", backend.ErrAllocation, elements)
	}

	return Own(make([]float32, elements)), nil
}

// Upload copies an exactly sized host slice into the allocation.
func (Device) Upload(dst backend.Buffer, values []float32) error {
	b, err := checked(dst)
	if err != nil {
		return err
	}

	data, err := b.values()
	if err != nil {
		return err
	}

	if len(values) != len(data) {
		return fmt.Errorf("%w: upload length mismatch", backend.ErrInvalidView)
	}

	copy(data, values)

	return nil
}

// Download copies an allocation into an exactly sized caller-owned slice.
func (Device) Download(src backend.Buffer, values []float32) error {
	b, err := checked(src)
	if err != nil {
		return err
	}

	data, err := b.values()
	if err != nil {
		return err
	}

	if len(values) != len(data) {
		return fmt.Errorf("%w: download length mismatch", backend.ErrInvalidView)
	}

	copy(values, data)

	return nil
}
