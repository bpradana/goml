package webgpu

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

type buffer struct {
	owner  *Device
	handle *nativeBuffer
	length int
	closed bool
}

func (b *buffer) Len() int {
	return b.length
}

func (b *buffer) DeviceID() string {
	return b.owner.id
}

func (b *buffer) Valid() error {
	b.owner.mu.Lock()
	defer b.owner.mu.Unlock()

	if b.closed {
		return fmt.Errorf("webgpu: %w", backend.ErrClosed)
	}

	return b.owner.valid()
}

func (b *buffer) Close() error {
	b.owner.mu.Lock()
	defer b.owner.mu.Unlock()

	b.close()

	return nil
}

func (b *buffer) close() {
	if !b.closed {
		b.owner.native.retire(b.handle)
		b.handle = nil
		b.closed = true
		delete(b.owner.buffers, b)
	}
}

func (d *Device) checked(value backend.Buffer) (*buffer, error) {
	if err := d.valid(); err != nil {
		return nil, err
	}

	b, ok := value.(*buffer)
	if !ok || b == nil || b.owner != d {
		return nil, fmt.Errorf("webgpu: %w", backend.ErrDeviceMismatch)
	}

	if b.closed {
		return nil, fmt.Errorf("webgpu: %w", backend.ErrClosed)
	}

	return b, nil
}
