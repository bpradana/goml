package webgpu

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

// CheckFinite checks logical input values on-device and downloads only one
// four-byte status. NaNs and infinities return numerical errors. Empty views pass.
func (d *Device) CheckFinite(input backend.View) error {
	status, err := d.Allocate(1)
	if err != nil {
		return err
	}
	defer func() { _ = status.Close() }()

	d.mu.Lock()
	defer d.mu.Unlock()

	output := backend.View{Buffer: status, Shape: []int{1}, Strides: []int{1}}

	handles, err := d.bindings(output, input)
	if err != nil {
		return err
	}

	p := params(input, input, backend.View{}, backend.View{})
	if err := d.native.dispatch("finite_values", p, handles, 1, 1); err != nil {
		return err
	}

	var values [1]float32
	if err := d.native.download(handles[0], values[:]); err != nil {
		return err
	}

	if values[0] != 0 {
		return fmt.Errorf("webgpu: input contains a nonfinite value")
	}

	return nil
}

var (
	_ backend.Device           = (*Device)(nil)
	_ backend.Buffer           = (*buffer)(nil)
	_ backend.UnaryBackend     = (*Device)(nil)
	_ backend.BinaryBackend    = (*Device)(nil)
	_ backend.ScalarBackend    = (*Device)(nil)
	_ backend.MatMulBackend    = (*Device)(nil)
	_ backend.ReductionBackend = (*Device)(nil)
	_ backend.CopyBackend      = (*Device)(nil)
	_ backend.FillBackend      = (*Device)(nil)
	_ backend.ConcatBackend    = (*Device)(nil)
	_ backend.FiniteBackend    = (*Device)(nil)
)
