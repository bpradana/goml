package webgpu

import (
	"embed"
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/bpradana/goml/tensor/backend"
)

//go:embed kernels/*.wgsl
var shaders embed.FS

// Every field is four-byte aligned, matching the WGSL storage-buffer structure.
// Copying this structure before encoding ends the borrowed View metadata lifetime.
type kernelParams struct {
	n, rank, op, flags    uint32
	shape, as, bs, cs, ds [8]uint32
	ao, bo, co, offset    uint32
	ints                  [24]uint32
	scalars               [4]float32
}

func (p kernelParams) bytes() ([]byte, error) {
	data := make([]byte, 304)
	_, err := binary.Encode(data, binary.LittleEndian, p)

	return data, err
}

// Native callbacks can run inside a call holding Device.mu, or on another
// thread. Keep their error lock separate to avoid reentrant device locking.
type failureState struct {
	mu  sync.Mutex
	err error
}

func (f *failureState) set(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.err == nil {
		f.err = err
	}
}

func (f *failureState) get() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	return f.err
}

func elements(shape []int) (int, error) {
	v := backend.View{Shape: shape}

	count, err := v.Numel()
	if err != nil {
		return 0, err
	}

	if len(shape) > 8 || uint64(count) > uint64(^uint32(0)) {
		return 0, fmt.Errorf("webgpu: %w: rank or element count exceeds shader limits", backend.ErrUnsupported)
	}

	for _, size := range shape {
		if uint64(size) > uint64(^uint32(0)) {
			return 0, fmt.Errorf("webgpu: %w: dimension exceeds 32 bits", backend.ErrUnsupported)
		}
	}

	return count, nil
}

func (d *Device) view(v backend.View, writable bool) (*buffer, error) {
	if v.Buffer == nil {
		return nil, fmt.Errorf("webgpu: %w: missing view buffer", backend.ErrInvalidView)
	}

	b, err := d.checked(v.Buffer)
	if err != nil {
		return nil, err
	}

	if err := v.ValidateLayout(b.length, writable); err != nil {
		return nil, err
	}

	if _, err := elements(v.Shape); err != nil {
		return nil, err
	}

	if uint64(v.Offset) > uint64(^uint32(0)) {
		return nil, fmt.Errorf("webgpu: %w: offset exceeds 32 bits", backend.ErrUnsupported)
	}

	for _, stride := range v.Strides {
		if uint64(stride) > uint64(^uint32(0)) {
			return nil, fmt.Errorf("webgpu: %w: stride exceeds 32 bits", backend.ErrUnsupported)
		}
	}

	return b, nil
}

func params(dst, a, b, c backend.View) kernelParams {
	count, _ := dst.Numel()

	p := kernelParams{n: uint32(count), rank: uint32(len(dst.Shape)), ao: uint32(a.Offset), bo: uint32(b.Offset), co: uint32(c.Offset), offset: uint32(dst.Offset)}
	for axis, size := range dst.Shape {
		p.shape[axis] = uint32(size)
		p.ds[axis] = uint32(dst.Strides[axis])
	}

	for axis, stride := range a.Strides {
		p.as[axis] = uint32(stride)
	}

	for axis, stride := range b.Strides {
		p.bs[axis] = uint32(stride)
	}

	for axis, stride := range c.Strides {
		p.cs[axis] = uint32(stride)
	}

	return p
}

// bindings checks physical ownership under Device.mu, including noncomparable
// foreign Buffer implementations. Only concrete owned buffer pointers are compared.
func (d *Device) bindings(dst backend.View, sources ...backend.View) ([4]*nativeBuffer, error) {
	var handles [4]*nativeBuffer

	inputs := make([]*buffer, len(sources))
	for i, source := range sources {
		b, err := d.view(source, false)
		if err != nil {
			return handles, err
		}

		inputs[i] = b
		handles[i+1] = b.handle
	}

	output, err := d.view(dst, true)
	if err != nil {
		return handles, err
	}

	for _, input := range inputs {
		if input == output {
			return handles, fmt.Errorf("webgpu: %w: aliased numerical destination", backend.ErrInvalidView)
		}
	}

	handles[0] = output.handle

	return handles, nil
}

func (d *Device) dispatch(name string, p kernelParams, handles [4]*nativeBuffer) error {
	if p.n == 0 {
		return nil
	}

	return d.native.dispatch(name, p, handles, 0, 0)
}
