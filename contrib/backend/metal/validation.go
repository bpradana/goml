package metal

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/tensor/backend"
)

// CheckFinite validates on the GPU and downloads only a four-byte status.
func (d *Device) CheckFinite(input backend.View) error {
	status, err := d.Allocate(1)
	if err != nil {
		return err
	}
	defer func() { _ = status.Close() }()

	if err := d.Fill(backend.View{Buffer: status, Shape: []int{1}, Strides: []int{1}}, 0); err != nil {
		return err
	}

	d.mu.Lock()

	source, err := d.view(input)
	if err != nil {
		d.mu.Unlock()
		return err
	}

	target, err := d.checked(status)
	if err != nil {
		d.mu.Unlock()
		return err
	}

	p := viewParams(input.Shape, input, backend.View{})
	err = nativeCompute(d.handle, "finite_values", source.handle, nil, target.handle, p)
	d.mu.Unlock()

	if err != nil {
		return err
	}

	var result [1]float32
	if err := d.Download(status, result[:]); err != nil {
		return err
	}

	if math.Float32bits(result[0]) != 0 {
		return fmt.Errorf("metal: input contains a nonfinite value")
	}

	return nil
}
