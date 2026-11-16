package metal

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
)

func convolutionDimensions(input, weight tensor.DeviceView, kernel, stride, padding int) ([]int, error) {
	shape, err := spatialShape(input.Shape, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	features := uint64(input.Shape[1]) * uint64(kernel)
	if features > uint64(^uint32(0))/uint64(kernel) {
		return nil, fmt.Errorf("metal: convolution: kernel element count overflow")
	}

	features *= uint64(kernel)
	if input.Shape[1] == 0 || len(weight.Shape) != 2 || uint64(weight.Shape[0]) != features || weight.Shape[1] <= 0 {
		return nil, fmt.Errorf("metal: convolution: incompatible weight shape")
	}

	shape[1] = weight.Shape[1]
	if _, err := elements(shape); err != nil {
		return nil, err
	}

	return shape, nil
}

// Conv2D encodes cached MPSGraph convolution on contiguous NCHW inputs.
func (d *Device) Conv2D(dst tensor.Buffer, input, weight tensor.DeviceView, kernel, stride, padding int) error {
	shape, err := convolutionDimensions(input, weight, kernel, stride, padding)
	if err != nil {
		return err
	}

	x, releaseX, err := d.contiguous(input)
	if err != nil {
		return err
	}
	defer releaseX()

	w, releaseW, err := d.contiguous(weight)
	if err != nil {
		return err
	}
	defer releaseW()

	d.mu.Lock()
	defer d.mu.Unlock()

	xb, err := d.view(x)
	if err != nil {
		return err
	}

	wb, err := d.view(w)
	if err != nil {
		return err
	}

	out, err := d.checked(dst)
	if err != nil {
		return err
	}

	count, _ := elements(shape)
	if out.length != count {
		return fmt.Errorf("metal: convolution: output size mismatch")
	}

	if out.length == 0 {
		return nil
	}

	p := viewParams(input.Shape, input, tensor.DeviceView{})
	p.ints = [8]uint32{uint32(weight.Shape[1]), uint32(kernel), uint32(stride), uint32(padding)}

	return nativeConv(d.handle, xb.handle, wb.handle, nil, out.handle, nil, p, false)
}

// Conv2DBackward encodes both MPSGraph convolution derivatives without im2col.
func (d *Device) Conv2DBackward(dx, dw tensor.Buffer, input, weight, gradient tensor.DeviceView, kernel, stride, padding int) error {
	shape, err := convolutionDimensions(input, weight, kernel, stride, padding)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, gradient.Shape) {
		return fmt.Errorf("metal: convolution backward: gradient shape mismatch")
	}

	x, releaseX, err := d.contiguous(input)
	if err != nil {
		return err
	}
	defer releaseX()

	w, releaseW, err := d.contiguous(weight)
	if err != nil {
		return err
	}
	defer releaseW()

	g, releaseG, err := d.contiguous(gradient)
	if err != nil {
		return err
	}
	defer releaseG()

	d.mu.Lock()
	defer d.mu.Unlock()

	xb, err := d.view(x)
	if err != nil {
		return err
	}

	wb, err := d.view(w)
	if err != nil {
		return err
	}

	gb, err := d.view(g)
	if err != nil {
		return err
	}

	out, err := d.checked(dx)
	if err != nil {
		return err
	}

	weights, err := d.checked(dw)
	if err != nil {
		return err
	}

	inputCount, _ := elements(input.Shape)

	weightCount, _ := elements(weight.Shape)
	if out.length != inputCount || weights.length != weightCount {
		return fmt.Errorf("metal: convolution backward: destination size mismatch")
	}

	if inputCount == 0 {
		return nativeCompute(d.handle, "fill_values", nil, nil, weights.handle, kernelParams{n: uint32(weightCount)})
	}

	p := viewParams(input.Shape, input, tensor.DeviceView{})
	p.ints = [8]uint32{uint32(weight.Shape[1]), uint32(kernel), uint32(stride), uint32(padding)}

	return nativeConv(d.handle, xb.handle, wb.handle, gb.handle, out.handle, weights.handle, p, true)
}
