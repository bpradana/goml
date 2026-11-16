package tensor

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

type deviceStorage struct {
	device Device
	buffer Buffer
	refs   atomic.Int64
}

type storageLease struct {
	storage *deviceStorage
	closed  atomic.Bool
	closeMu sync.Mutex
	scope   *Scope
}

func (l *storageLease) close() error {
	// Cleanup and explicit Close can race. Wait for the winning close to
	// release its reference before returning from the explicit lifetime boundary.
	l.closeMu.Lock()
	defer l.closeMu.Unlock()

	if l.closed.Swap(true) {
		return nil
	}

	if l.storage.refs.Add(-1) == 0 {
		return l.storage.buffer.Close()
	}

	return nil
}

func attachStorage(t *Tensor, storage *deviceStorage, scope *Scope) *Tensor {
	storage.refs.Add(1)
	lease := &storageLease{storage: storage, scope: scope}
	t.gpu = lease

	if scope != nil {
		scope.mu.Lock()
		if scope.closed {
			_ = lease.close()
		} else {
			scope.leases = append(scope.leases, lease)
		}
		scope.mu.Unlock()
	}

	runtime.AddCleanup(t, func(l *storageLease) { _ = l.close() }, lease)

	return t
}

// Device returns nil for CPU storage or the owner of this GPU allocation.
func (t *Tensor) Device() Device {
	if t == nil || t.gpu == nil {
		return nil
	}

	return t.gpu.storage.device
}

// IsCPU reports whether values reside in Go-managed CPU storage.
func (t *Tensor) IsCPU() bool {
	return t != nil && t.gpu == nil
}

// Close releases this tensor's GPU lease. Other views keep their own leases.
// CPU Close is a no-op. Closed GPU tensors retain metadata but reject execution.
func (t *Tensor) Close() error {
	if t == nil || t.gpu == nil {
		return nil
	}

	return t.gpu.close()
}

func (t *Tensor) deviceView() DeviceView {
	return DeviceView{Buffer: t.gpu.storage.buffer, Shape: t.shape, Strides: t.strides, Offset: t.offset}
}

func compatible(op string, inputs ...*Tensor) error {
	first := inputs[0].Device()
	for _, input := range inputs[1:] {
		other := input.Device()
		if (first == nil) != (other == nil) || (first != nil && first.ID() != other.ID()) {
			return fmt.Errorf("tensor: %s: incompatible devices", op)
		}
	}

	return nil
}

func allocatedOn(shape []int, inputs ...*Tensor) (*Tensor, error) {
	device := inputs[0].Device()
	if device == nil {
		return allocated(shape)
	}

	var scope *Scope
	for _, input := range inputs {
		if input.gpu != nil && input.gpu.scope != nil {
			scope = input.gpu.scope
			break
		}
	}

	return allocateDevice(device, scope, shape)
}

// Scope explicitly owns temporary GPU tensors created from its input tensors.
// Operations propagate the scope from their inputs. Keep detaches an output's
// lease before Close. A scope may be used by multiple goroutines; synchronize
// those goroutines before closing it. CPU tensors do not need a scope.
type Scope struct {
	device Device
	mu     sync.Mutex
	leases []*storageLease
	closed bool
}

// NewScope creates a lifetime boundary on device. It performs no allocation.
func NewScope(device Device) (*Scope, error) {
	if device == nil {
		return nil, fmt.Errorf("tensor: scope: GPU device required")
	}

	return &Scope{device: device}, nil
}

// To transfers or aliases input onto this scope's device. The returned tensor
// has an independent lease and is invalidated by Close unless kept explicitly.
func (s *Scope) To(input *Tensor) (*Tensor, error) {
	if s == nil || s.device == nil {
		return nil, fmt.Errorf("tensor: scope: use NewScope before To")
	}

	result, err := input.To(s.device)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		_ = result.Close()
		return nil, fmt.Errorf("tensor: scope: already closed")
	}

	result.gpu.scope = s
	s.leases = append(s.leases, result.gpu)

	return result, nil
}

// Keep preserves this tensor's lease beyond its scope. Close remains the
// caller's responsibility. Each storage-sharing view owns its own lease.
func (t *Tensor) Keep() {
	if t == nil || t.gpu == nil || t.gpu.scope == nil {
		return
	}

	scope := t.gpu.scope
	scope.mu.Lock()
	t.gpu.scope = nil
	scope.mu.Unlock()
}

// Close releases all unkept temporary leases. It does not wait for the GPU;
// pending native commands retain their resources until completion.
func (s *Scope) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true

	var first error

	for _, lease := range s.leases {
		if lease.scope == s {
			if err := lease.close(); err != nil && first == nil {
				first = err
			}
		}
	}

	s.leases = nil

	return first
}

// IsClosed reports whether this GPU tensor or its device has been closed.
// CPU tensors do not have a closeable allocation and return false.
func (t *Tensor) IsClosed() bool {
	return t != nil && t.gpu != nil && (t.gpu.closed.Load() || t.gpu.storage.buffer.Valid() != nil)
}
