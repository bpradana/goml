package tensor

import (
	"fmt"
	"runtime"
	"slices"

	internalshape "github.com/bpradana/goml/tensor/internal/shape"
)

func allocateDevice(device Device, scope *Scope, shape []int) (*Tensor, error) {
	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	buffer, err := device.Allocate(n)
	if err != nil {
		return nil, fmt.Errorf("tensor: allocate on %s: %w", device.Name(), err)
	}

	storage := &deviceStorage{device: device, buffer: buffer}

	return attachStorage(&Tensor{shape: slices.Clone(shape), strides: strides, size: n}, storage, scope), nil
}

// To explicitly transfers logical values to device. A nil device means CPU.
// A same-device GPU transfer creates an independent storage-sharing lease;
// CPU-to-CPU transfers copy. Cross-GPU transfers use explicit host staging.
func (t *Tensor) To(device Device) (*Tensor, error) {
	defer runtime.KeepAlive(t)

	if err := valid("transfer", t); err != nil {
		return nil, err
	}

	if device == nil {
		return t.ToCPU()
	}

	if t.gpu != nil && device.ID() == t.Device().ID() {
		return attachStorage(&Tensor{shape: t.Shape(), strides: t.Strides(), offset: t.offset, size: t.size}, t.gpu.storage, nil), nil
	}

	values, err := t.HostData()
	if err != nil {
		return nil, err
	}

	result, err := allocateDevice(device, nil, t.shape)
	if err != nil {
		return nil, err
	}

	if err := device.Upload(result.gpu.storage.buffer, values); err != nil {
		_ = result.Close()
		return nil, fmt.Errorf("tensor: upload: %w", err)
	}

	return result, nil
}

// ToCPU explicitly downloads GPU values, returning independent contiguous CPU
// storage. CPU inputs are copied. Transfer and synchronization failures return errors.
func (t *Tensor) ToCPU() (*Tensor, error) {
	values, err := t.HostData()
	if err != nil {
		return nil, err
	}

	return From(values, t.shape...)
}

// HostData explicitly synchronizes and downloads GPU data in logical order.
// CPU inputs return the same independent copy as Data. No unowned native slice
// is exposed. Closed tensors and native failures return errors.
func (t *Tensor) HostData() ([]float32, error) {
	defer runtime.KeepAlive(t)

	if err := valid("host data", t); err != nil {
		return nil, err
	}

	if t.gpu == nil {
		return t.Data(), nil
	}

	source := t
	if !t.IsContiguous() || t.offset != 0 || t.size != t.gpu.storage.buffer.Len() {
		var err error

		source, err = t.MakeContiguous()
		if err != nil {
			return nil, err
		}

		defer func() { _ = source.Close() }()
	}

	values := make([]float32, t.size)
	if err := source.Device().Download(source.gpu.storage.buffer, values); err != nil {
		return nil, err
	}

	return values, nil
}

// MakeContiguous allocates independent row-major storage on the current device.
// GPU execution is fallible and does not perform a host transfer.
func (t *Tensor) MakeContiguous() (*Tensor, error) {
	defer runtime.KeepAlive(t)

	if err := valid("contiguous", t); err != nil {
		return nil, err
	}

	if t.gpu == nil {
		return t.Contiguous(), nil
	}

	backend, ok := t.Device().(LayoutBackend)
	if !ok {
		return nil, fmt.Errorf("tensor: device lacks materialization")
	}

	out, err := allocatedOn(t.shape, t)
	if err != nil {
		return nil, err
	}

	if err := backend.Copy(out.deviceView(), t.deviceView()); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

// FullLike allocates a tensor on reference's device with the requested shape,
// filled with value. It is useful for device-correct seeds and optimizer state.
func FullLike(reference *Tensor, value float32, shape ...int) (*Tensor, error) {
	if err := valid("full", reference); err != nil {
		return nil, err
	}

	out, err := allocatedOn(shape, reference)
	if err != nil {
		return nil, err
	}

	if out.gpu == nil {
		for i := range out.data {
			out.data[i] = value
		}

		return out, nil
	}

	backend, ok := out.Device().(ArithmeticBackend)
	if !ok {
		_ = out.Close()
		return nil, fmt.Errorf("tensor: device lacks fill")
	}

	if err := backend.Fill(out.gpu.storage.buffer, value); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

// FromLike copies host values into a tensor on reference's device and temporary
// scope. It explicitly uploads GPU constants and validates the exact shape.
func FromLike(reference *Tensor, values []float32, shape ...int) (*Tensor, error) {
	if err := valid("from like", reference); err != nil {
		return nil, err
	}

	if reference.gpu == nil {
		return From(values, shape...)
	}

	output, err := allocatedOn(shape, reference)
	if err != nil {
		return nil, err
	}

	if output.size != len(values) {
		_ = output.Close()
		return nil, fmt.Errorf("tensor: from like: data length does not match shape %v", shape)
	}

	if err := output.Device().Upload(output.gpu.storage.buffer, values); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}
