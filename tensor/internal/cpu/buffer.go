package cpu

import (
	"sync"
	"sync/atomic"

	"github.com/bpradana/goml/tensor/backend"
)

type buffer struct {
	mu     sync.Mutex
	data   []float32
	length int
	closed atomic.Bool
}

func (b *buffer) Len() int { return b.length }

func (*buffer) DeviceID() string { return ID }

func (b *buffer) Valid() error {
	if b.closed.Load() {
		return backend.ErrClosed
	}

	return nil
}

func (b *buffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.closed.Store(true)
	b.data = nil

	return nil
}

// values takes one allocation snapshot under the close lock. The local Go slice
// retains physical memory across concurrent Close. Liveness checks use an atomic
// flag, without allocating a separate memory-state object for every CPU array.
// Callers coordinate overlapping writes as required by the backend contract.
func (b *buffer) values() ([]float32, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed.Load() {
		return nil, backend.ErrClosed
	}

	return b.data, nil
}

func checked(value backend.Buffer) (*buffer, error) {
	b, ok := value.(*buffer)
	if !ok || b == nil {
		return nil, backend.ErrDeviceMismatch
	}

	return b, nil
}

// Own transfers already-private Go memory into CPU storage without another copy.
func Own(values []float32) backend.Buffer {
	return &buffer{data: values, length: len(values)}
}

// GCManaged reports that storage has no native resource or allocation registry.
func (*buffer) GCManaged() bool { return true }
