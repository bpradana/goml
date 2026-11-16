package metal

import (
	"fmt"
	"unsafe"

	"github.com/bpradana/goml/tensor/backend"
)

type buffer struct {
	owner  *Device
	handle unsafe.Pointer
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

	if b.closed || b.owner.closed {
		return fmt.Errorf("metal: %w", backend.ErrClosed)
	}

	return nil
}

func (b *buffer) Close() error {
	b.owner.mu.Lock()
	defer b.owner.mu.Unlock()

	b.close()

	return nil
}

func (b *buffer) close() {
	if !b.closed {
		nativeRelease(b.handle)
		b.closed = true
		b.handle = nil
		delete(b.owner.buffers, b)
	}
}

func (d *Device) checked(value backend.Buffer) (*buffer, error) {
	if d.closed || d.handle == nil {
		return nil, fmt.Errorf("metal: %w", backend.ErrClosed)
	}

	b, ok := value.(*buffer)
	if !ok || b == nil || b.owner != d {
		return nil, fmt.Errorf("metal: %w", backend.ErrDeviceMismatch)
	}

	if b.closed {
		return nil, fmt.Errorf("metal: %w", backend.ErrClosed)
	}

	return b, nil
}
