package examplebackend

import "github.com/bpradana/goml/tensor/backend"

type buffer struct {
	owner  *Device
	data   []float32
	length int
	closed bool
}

func (b *buffer) Len() int { return b.length }

func (b *buffer) DeviceID() string { return b.owner.ID() }

func (b *buffer) Valid() error {
	b.owner.mu.Lock()
	defer b.owner.mu.Unlock()

	if b.closed || b.owner.closed {
		return backend.ErrClosed
	}

	return nil
}

func (b *buffer) Close() error {
	b.owner.mu.Lock()
	defer b.owner.mu.Unlock()

	b.closed = true
	b.data = nil
	delete(b.owner.buffers, b)

	return nil
}
