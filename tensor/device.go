package tensor

import (
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

// CPU returns the shared stateless CPU execution context. Closing it is harmless;
// individual tensor handles and buffers retain independent lifetimes.
func CPU() backend.Device {
	return cpu.Device{}
}

// Device returns this tensor's execution context. Constructed tensors always
// have a device; an uninitialized or nil tensor returns nil.
func (t *Tensor) Device() backend.Device {
	if t == nil {
		return nil
	}

	return t.device
}

// HostAccessible reports support for synchronous host-native reads. Accelerators
// do not claim this capability, even when native memory is CPU-mapped.
func (t *Tensor) HostAccessible() bool {
	if t == nil {
		return false
	}

	_, ok := t.device.(backend.HostAccess)

	return ok
}

// IsCPU reports whether this tensor uses goml's shared CPU context.
// Numerical dispatch uses capabilities, never device names or this predicate.
func (t *Tensor) IsCPU() bool {
	if t == nil {
		return false
	}

	_, ok := t.device.(cpu.Device)

	return ok
}
