// Package metal implements goml's Float32 GPU backend on Apple Metal.
// Open a device explicitly, transfer tensors with To, and close scopes and devices
// to release native allocations. Only this module requires CGo and Apple frameworks.
package metal

import (
	"crypto/rand"
	"fmt"
	"sync"
	"unsafe"

	"github.com/bpradana/goml/tensor/backend"
)

// Device owns one native Metal device, queue, shader cache, and buffer registry.
// Methods serialize command encoding; independent devices have independent state.
// Close waits for pending commands and invalidates all outstanding buffers.
type Device struct {
	mu      sync.Mutex
	handle  unsafe.Pointer
	name    string
	id      string
	buffers map[*buffer]struct{}
	closed  bool
}

// Stats records native allocation bytes, including buffers retained by queued
// commands, and the high-water mark. MPS implementation-private workspaces are
// excluded from these tracked allocation totals.
type Stats struct {
	LiveBytes, PeakBytes uint64
	LiveBuffers          int
}

// Open selects the default Metal device. Unsupported platforms and native
// initialization or shader compilation failures return errors.
func Open() (*Device, error) {
	return OpenDevice(-1)
}

// OpenDevice selects a device by enumeration index; -1 selects the default GPU.
func OpenDevice(index int) (*Device, error) {
	if index < -1 || int64(index) > int64(1<<31-1) {
		return nil, fmt.Errorf("metal: device index must be -1 or a nonnegative 32-bit integer")
	}

	handle, name, err := nativeOpen(index)
	if err != nil {
		return nil, err
	}

	id := make([]byte, 16)
	if _, err := rand.Read(id); err != nil {
		nativeClose(handle)
		return nil, err
	}

	return &Device{
		handle:  handle,
		name:    name,
		id:      fmt.Sprintf("metal:%x", id),
		buffers: make(map[*buffer]struct{}),
	}, nil
}

// Name returns the selected hardware's display name.
func (d *Device) Name() string {
	return d.name
}

// ID returns a unique identity for this device instance.
func (d *Device) ID() string {
	return d.id
}

// Allocate creates uninitialized persistent Float32 storage. Zero elements are
// supported. Allocation limits and closed-device failures return errors.
func (d *Device) Allocate(elements int) (backend.Buffer, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed || d.handle == nil || elements < 0 || uint64(elements) > uint64(^uint32(0)) {
		if d.closed || d.handle == nil {
			return nil, fmt.Errorf("metal: allocation: %w", backend.ErrClosed)
		}

		return nil, fmt.Errorf("metal: allocation: %w", backend.ErrAllocation)
	}

	handle, err := nativeAllocate(d.handle, uint64(elements)*4)
	if err != nil {
		return nil, fmt.Errorf("metal: allocate: %w: %w", backend.ErrAllocation, err)
	}

	result := &buffer{owner: d, handle: handle, length: elements}
	d.buffers[result] = struct{}{}

	return result, nil
}

// Upload copies exactly one buffer's Float32 values into native storage. It
// synchronizes prior commands before modifying storage, preventing host races.
func (d *Device) Upload(dst backend.Buffer, values []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	b, err := d.checked(dst)
	if err != nil {
		return err
	}

	if len(values) != b.length {
		return fmt.Errorf("%w: Metal upload length", backend.ErrInvalidView)
	}

	return nativeUpload(d.handle, b.handle, values)
}

// Download synchronizes and copies a buffer into the caller-owned destination.
func (d *Device) Download(src backend.Buffer, values []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	b, err := d.checked(src)
	if err != nil {
		return err
	}

	if len(values) != b.length {
		return fmt.Errorf("%w: Metal download length", backend.ErrInvalidView)
	}

	return nativeDownload(d.handle, b.handle, values)
}

// Synchronize submits queued work, waits, and returns asynchronous GPU errors.
func (d *Device) Synchronize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return fmt.Errorf("metal: %w", backend.ErrClosed)
	}

	return nativeSync(d.handle)
}

// Close waits for pending work, releases all buffers and native resources, and
// returns a queued execution error if present. Repeated Close calls succeed.
func (d *Device) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return nil
	}

	err := nativeSync(d.handle)
	for b := range d.buffers {
		b.close()
	}

	nativeClose(d.handle)
	d.handle = nil
	d.closed = true

	return err
}

// Stats reports tracked allocations. Call Synchronize to retire completed work
// before measuring a quiescent baseline.
func (d *Device) Stats() Stats {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return Stats{}
	}

	live, peak := nativeStats(d.handle)

	return Stats{LiveBytes: live, PeakBytes: peak, LiveBuffers: len(d.buffers)}
}
