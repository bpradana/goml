package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
	internalshape "github.com/bpradana/goml/tensor/internal/shape"
)

func allocateDevice(device backend.Device, scope *Scope, shape []int) (*Tensor, error) {
	if device == nil {
		return nil, fmt.Errorf("tensor: allocate: device required")
	}

	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	return allocateLayout(device, scope, shape, n, strides)
}

// allocateLayout consumes already-validated layout metadata so constructors do
// not allocate discarded strides and validate the same shape a second time.
func allocateLayout(device backend.Device, scope *Scope, shape []int, n int, strides []int) (*Tensor, error) {
	if device == nil {
		return nil, fmt.Errorf("tensor: allocate: device required")
	}

	buffer, err := device.Allocate(n)
	if err != nil {
		return nil, fmt.Errorf("tensor: allocate on %s: %w", device.Name(), err)
	}

	return attachStorage(&Tensor{device: device, shape: slices.Clone(shape), strides: strides, size: n}, newStorage(buffer), scope)
}

// FromOn copies host data to a contiguous tensor on device. Shape and element
// count must match. Allocation, upload, and device failures return errors.
func FromOn(device backend.Device, data []float32, shape ...int) (*Tensor, error) {
	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	if n != len(data) {
		return nil, fmt.Errorf("tensor: from: shape %v needs %d elements, got %d", shape, n, len(data))
	}

	output, err := allocateLayout(device, nil, shape, n, strides)
	if err != nil {
		return nil, err
	}

	if err := device.Upload(output.deviceView().Buffer, data); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}

// ZerosOn allocates and explicitly initializes a contiguous zero tensor.
func ZerosOn(device backend.Device, shape ...int) (*Tensor, error) {
	return fullOn(device, nil, 0, shape)
}

// OnesOn allocates a contiguous one-filled tensor on device.
func OnesOn(device backend.Device, shape ...int) (*Tensor, error) {
	return fullOn(device, nil, 1, shape)
}

// ScalarOn allocates a rank-zero tensor on device.
func ScalarOn(device backend.Device, value float32) (*Tensor, error) {
	return FromOn(device, []float32{value})
}

// To copies logical values to independent storage on device. Same-device copies
// execute through CopyBackend; use Retain to share immutable storage. Transfers
// between execution contexts explicitly stage through host memory.
func (t *Tensor) To(device backend.Device) (*Tensor, error) {
	defer runtime.KeepAlive(t)

	if err := valid("transfer", t); err != nil {
		return nil, err
	}

	if device == nil {
		return nil, fmt.Errorf("tensor: transfer: device required; use CPU()")
	}

	if t.device.ID() == device.ID() {
		result, err := t.MakeContiguous()
		if err != nil {
			return nil, err
		}

		result.Keep()

		return result, nil
	}

	values, err := t.HostData()
	if err != nil {
		return nil, err
	}

	return FromOn(device, values, t.shape...)
}

// ToCPU copies logical values into independent contiguous CPU storage.
func (t *Tensor) ToCPU() (*Tensor, error) { return t.To(CPU()) }

// HostData explicitly reads logical values into caller-owned Go memory. It may
// synchronize and download accelerator work. No native memory slice is exposed.
func (t *Tensor) HostData() ([]float32, error) {
	defer runtime.KeepAlive(t)

	if err := valid("host data", t); err != nil {
		return nil, err
	}

	values := make([]float32, t.size)
	if host, ok := t.device.(backend.HostAccess); ok {
		if err := host.Read(t.deviceView(), values); err != nil {
			return nil, err
		}

		return values, nil
	}

	source := t

	view := t.deviceView()
	if !t.IsContiguous() || t.offset != 0 || t.size != view.Buffer.Len() {
		// A host read is an explicit transfer. A minimal backend without Copy
		// can still download physical storage and gather logical view values.
		if _, ok := t.device.(backend.CopyBackend); !ok {
			physical := make([]float32, view.Buffer.Len())
			if err := t.device.Download(view.Buffer, physical); err != nil {
				return nil, err
			}

			for i := range values {
				values[i] = physical[t.index(i)]
			}

			return values, nil
		}

		var err error

		source, err = t.MakeContiguous()
		if err != nil {
			return nil, err
		}
		defer func() { _ = source.Close() }()
	}

	if err := source.device.Download(source.deviceView().Buffer, values); err != nil {
		return nil, err
	}

	return values, nil
}

// MakeContiguous copies logical values into independent contiguous storage on
// this device. Missing copy capability returns ErrUnsupported; no CPU fallback.
func (t *Tensor) MakeContiguous() (*Tensor, error) {
	defer runtime.KeepAlive(t)

	if err := valid("contiguous", t); err != nil {
		return nil, err
	}

	compute, ok := t.device.(backend.CopyBackend)
	if !ok {
		return nil, unsupported("copy", t.device)
	}

	output, err := allocatedOn(t.shape, t)
	if err != nil {
		return nil, err
	}

	if err := compute.Copy(output.deviceView(), t.deviceView()); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}

func fullOn(device backend.Device, scope *Scope, value float32, shape []int) (*Tensor, error) {
	if device == nil {
		return nil, fmt.Errorf("tensor: full: device required")
	}

	compute, ok := device.(backend.FillBackend)
	if !ok {
		return nil, unsupported("fill", device)
	}

	output, err := allocateDevice(device, scope, shape)
	if err != nil {
		return nil, err
	}

	if err := compute.Fill(output.deviceView(), value); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}

// FullLike allocates and fills storage on reference's device and temporary scope.
func FullLike(reference *Tensor, value float32, shape ...int) (*Tensor, error) {
	defer runtime.KeepAlive(reference)

	if err := valid("full", reference); err != nil {
		return nil, err
	}

	return fullOn(reference.device, reference.scope.Load(), value, shape)
}

// FromLike explicitly copies host values to reference's device and scope.
func FromLike(reference *Tensor, values []float32, shape ...int) (*Tensor, error) {
	defer runtime.KeepAlive(reference)

	if err := valid("from like", reference); err != nil {
		return nil, err
	}

	n, strides, err := internalshape.Layout(shape)
	if err != nil {
		return nil, err
	}

	if n != len(values) {
		return nil, fmt.Errorf("tensor: from like: data length does not match shape %v", shape)
	}

	output, err := allocateLayout(reference.device, reference.scope.Load(), shape, n, strides)
	if err != nil {
		return nil, err
	}

	if err := output.device.Upload(output.deviceView().Buffer, values); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}

func unsupported(operation string, device backend.Device) error {
	return fmt.Errorf("tensor: %s on %s: %w", operation, device.Name(), backend.ErrUnsupported)
}
