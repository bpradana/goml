// Package webgpu implements goml's Float32 GPU backend using native WebGPU
// and embedded WGSL compute shaders. Open a context explicitly and close tensor
// handles, scopes, and the device to release GPU allocations. CPU applications
// do not need to import this optional module.
package webgpu

import (
	"crypto/rand"
	"fmt"
	"sync"

	"github.com/bpradana/goml/tensor/backend"
)

// Device owns an independent WebGPU execution context and ordered command queue.
// Methods are safe for concurrent use with independent buffers. Callers coordinate
// overlapping writes, model updates, and shutdown with live numerical operations.
type Device struct {
	mu      sync.Mutex
	native  *nativeDevice
	id      string
	name    string
	buffers map[*buffer]struct{}
	closed  bool
}

// Stats describes physical tensor allocations, including closed buffers retained
// by queued commands. Internal parameter/readback buffers and driver workspaces
// are excluded. Synchronize before measuring a quiescent baseline.
type Stats struct {
	LiveBytes, PeakBytes uint64
	LiveBuffers          int
}

// Open selects a high-performance hardware adapter. Software adapters are not
// used as a fallback. Missing GPU support and unsupported builds return
// backend.ErrUnsupported; native initialization and shader failures retain details.
func Open() (*Device, error) {
	return OpenDevice(-1)
}

// OpenDevice selects a native adapter by enumeration index; -1 selects the
// default hardware adapter. Each call creates a distinct execution context.
func OpenDevice(index int) (*Device, error) {
	if index < -1 {
		return nil, fmt.Errorf("webgpu: adapter index must be -1 or nonnegative")
	}

	native, name, err := openNative(index)
	if err != nil {
		return nil, err
	}

	id := make([]byte, 16)
	if _, err := rand.Read(id); err != nil {
		native.close()
		return nil, err
	}

	return &Device{native: native, id: fmt.Sprintf("webgpu:%x", id), name: name, buffers: make(map[*buffer]struct{})}, nil
}

// ID returns this execution context's stable identity.
func (d *Device) ID() string {
	return d.id
}

// Name returns the selected hardware adapter's display name.
func (d *Device) Name() string {
	return d.name
}

// Allocate creates persistent Float32 GPU storage. Empty allocations are valid.
// The physical buffer is at least four bytes to satisfy WebGPU binding rules.
func (d *Device) Allocate(elements int) (backend.Buffer, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if err := d.valid(); err != nil {
		return nil, err
	}

	if elements < 0 || uint64(elements) > uint64(^uint32(0))/4 {
		return nil, fmt.Errorf("webgpu: %w: invalid Float32 allocation size", backend.ErrAllocation)
	}

	handle, err := d.native.allocate(max(4, uint64(elements)*4))
	if err != nil {
		return nil, fmt.Errorf("webgpu: %w: %w", backend.ErrAllocation, err)
	}

	result := &buffer{owner: d, handle: handle, length: elements}
	d.buffers[result] = struct{}{}

	return result, nil
}

// Upload copies exactly one allocation from caller-owned host values. Writes are
// ordered after earlier GPU operations without exposing mapped storage to Go.
func (d *Device) Upload(dst backend.Buffer, values []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	b, err := d.checked(dst)
	if err != nil {
		return err
	}

	if len(values) != b.length {
		return fmt.Errorf("webgpu: upload: %w: length mismatch", backend.ErrInvalidView)
	}

	return d.native.upload(b.handle, values)
}

// Download waits for earlier commands and copies exactly one allocation into
// caller-owned host memory. No borrowed native slice escapes the call.
func (d *Device) Download(src backend.Buffer, values []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	b, err := d.checked(src)
	if err != nil {
		return err
	}

	if len(values) != b.length {
		return fmt.Errorf("webgpu: download: %w: length mismatch", backend.ErrInvalidView)
	}

	return d.native.download(b.handle, values)
}

// Synchronize submits pending commands, waits for completion, and returns GPU
// execution or device-loss errors. Mathematical operations do not wait per call.
func (d *Device) Synchronize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if err := d.valid(); err != nil {
		return err
	}

	return d.native.synchronize()
}

// Close waits for pending commands, invalidates all allocations, and releases the
// execution context. Cleanup still completes after a GPU error. Close is idempotent.
func (d *Device) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed || d.native == nil {
		d.closed = true
		return nil
	}

	err := d.native.synchronize()
	for b := range d.buffers {
		b.close()
	}

	d.native.close()
	d.native = nil
	d.closed = true

	return err
}

// Stats returns tracked tensor allocation counts and the peak byte count.
func (d *Device) Stats() Stats {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed || d.native == nil {
		return Stats{}
	}

	return d.native.stats()
}

func (d *Device) valid() error {
	if d.closed || d.native == nil {
		return fmt.Errorf("webgpu: %w", backend.ErrClosed)
	}

	return d.native.failure.get()
}
