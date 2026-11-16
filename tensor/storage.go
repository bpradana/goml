package tensor

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/bpradana/goml/tensor/backend"
)

type storage struct {
	mu       sync.Mutex
	buffer   backend.Buffer
	refs     int
	released bool
	cleanup  runtime.Cleanup
}

func newStorage(buffer backend.Buffer) *storage {
	allocation := &storage{buffer: buffer}
	// One cleanup per physical allocation covers abandoned handles without a
	// cleanup object per view. Explicit Close and Scope are the primary lifetime.
	managed, ok := buffer.(backend.GCManagedBuffer)
	if !ok || !managed.GCManaged() {
		allocation.cleanup = runtime.AddCleanup(allocation, func(buffer backend.Buffer) { _ = buffer.Close() }, buffer)
	}

	return allocation
}

func (s *storage) retain() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.released {
		return backend.ErrClosed
	}

	s.refs++

	return nil
}

func (s *storage) release() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.refs--
	if s.refs != 0 {
		return nil
	}

	s.released = true
	err := s.buffer.Close()
	s.cleanup.Stop()

	return err
}

func attachStorage(t *Tensor, allocation *storage, scope *Scope) (*Tensor, error) {
	if err := allocation.retain(); err != nil {
		return nil, err
	}

	t.allocation.Store(allocation)

	if scope != nil {
		scope.mu.Lock()
		defer scope.mu.Unlock()

		if scope.closed {
			_ = t.Close()
			return nil, fmt.Errorf("tensor: scope: %w", backend.ErrClosed)
		}

		t.scope.Store(scope)
		scope.tensors = append(scope.tensors, t)
	}

	return t, nil
}

// Close releases this handle on every device. Other retained views remain valid.
// Metadata remains readable. Repeated or concurrent Close waits for the winning
// release. Queued device operations retain physical allocations independently.
func (t *Tensor) Close() error {
	if t == nil {
		return nil
	}

	t.closeMu.Lock()
	defer t.closeMu.Unlock()

	allocation := t.allocation.Swap(nil)
	if allocation == nil {
		return nil
	}

	return allocation.release()
}

// IsClosed reports a closed or uninitialized handle, buffer, or owning device.
func (t *Tensor) IsClosed() bool {
	if t == nil {
		return true
	}

	allocation := t.allocation.Load()

	return allocation == nil || allocation.buffer.Valid() != nil
}

func (t *Tensor) deviceView() backend.View {
	allocation := t.allocation.Load()
	if allocation == nil {
		return backend.View{}
	}

	return backend.View{Buffer: allocation.buffer, Shape: t.shape, Strides: t.strides, Offset: t.offset}
}

// Retain creates an independent handle sharing immutable storage and metadata.
// Closing either handle does not invalidate the other. The returned handle is
// unscoped; its caller owns Close. No data transfer or numerical copy occurs.
func (t *Tensor) Retain() (*Tensor, error) {
	if err := valid("retain", t); err != nil {
		return nil, err
	}

	return t.share(&Tensor{shape: t.shape, strides: t.strides, offset: t.offset, size: t.size}, nil)
}

func (t *Tensor) share(view *Tensor, scope *Scope) (*Tensor, error) {
	defer runtime.KeepAlive(t)

	allocation := t.allocation.Load()
	if allocation == nil {
		return nil, backend.ErrClosed
	}

	view.device = t.device

	return attachStorage(view, allocation, scope)
}

func compatible(op string, inputs ...*Tensor) error {
	first := inputs[0].device
	for _, input := range inputs[1:] {
		if first.ID() != input.device.ID() {
			return fmt.Errorf("tensor: %s: %w (%s and %s)", op, backend.ErrDeviceMismatch, first.ID(), input.device.ID())
		}
	}

	return nil
}

func allocatedOn(shape []int, inputs ...*Tensor) (*Tensor, error) {
	var scope *Scope
	for _, input := range inputs {
		if s := input.scope.Load(); s != nil {
			scope = s
			break
		}
	}

	return allocateDevice(inputs[0].device, scope, shape)
}

// Scope owns temporary tensor handles on one device. Operations propagate it
// from their inputs. Keep detaches an output before closing the scope. Finish all
// goroutines using scoped tensors before Close. Physical queued work may finish
// later because asynchronous backends retain their native resources.
type Scope struct {
	device  backend.Device
	mu      sync.Mutex
	tensors []*Tensor
	closed  bool
}

// NewScope creates a lifetime boundary without allocating numerical storage.
func NewScope(device backend.Device) (*Scope, error) {
	if device == nil {
		return nil, fmt.Errorf("tensor: scope: device required")
	}

	return &Scope{device: device}, nil
}

// To returns a scoped independent handle on the scope's device. Same-device
// values share immutable storage; cross-device values are explicitly copied.
func (s *Scope) To(input *Tensor) (*Tensor, error) {
	if s == nil {
		return nil, fmt.Errorf("tensor: scope: use NewScope")
	}

	var (
		result *Tensor
		err    error
	)
	if input != nil && input.Device() != nil && input.Device().ID() == s.device.ID() {
		result, err = input.Retain()
	} else {
		result, err = input.To(s.device)
	}

	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		_ = result.Close()
		return nil, backend.ErrClosed
	}

	result.scope.Store(s)
	s.tensors = append(s.tensors, result)

	return result, nil
}

// Keep detaches this handle from its scope. The caller then owns Close.
func (t *Tensor) Keep() {
	if t == nil {
		return
	}

	s := t.scope.Load()
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	t.scope.CompareAndSwap(s, nil)
}

// Close releases every unkept handle. Repeated Close is harmless.
func (s *Scope) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true

	var first error

	for _, t := range s.tensors {
		if t.scope.Load() == s {
			if err := t.Close(); err != nil && first == nil {
				first = err
			}
		}
	}

	s.tensors = nil

	return first
}
