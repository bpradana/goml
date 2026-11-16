package metal

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

func convolutionDimensions(input, weight backend.View, kernel, stride, padding int) ([]int, error) {
	shape, err := spatialShape(input.Shape, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	features := uint64(input.Shape[1]) * uint64(kernel)
	if features > uint64(^uint32(0))/uint64(kernel) {
		return nil, fmt.Errorf("%w: Metal convolution indexing exceeds 32 bits", backend.ErrUnsupported)
	}

	features *= uint64(kernel)
	if input.Shape[1] == 0 || len(weight.Shape) != 2 || uint64(weight.Shape[0]) != features || weight.Shape[1] <= 0 {
		return nil, fmt.Errorf("%w: metal: convolution: incompatible weight shape", backend.ErrInvalidView)
	}

	shape[1] = weight.Shape[1]
	if _, err := elements(shape); err != nil {
		return nil, err
	}

	return shape, nil
}

// Conv2D encodes cached MPSGraph convolution on contiguous NCHW inputs.
func (d *Device) conv2D(dst, input, weight backend.View, options backend.Conv2DOptions) error {
	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

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

	out, err := d.output(dst, input, weight)
	if err != nil {
		return err
	}

	count, _ := elements(shape)
	if out.length != count {
		return fmt.Errorf("%w: metal: convolution: output size mismatch", backend.ErrInvalidView)
	}

	if out.length == 0 {
		return nil
	}

	p := viewParams(input.Shape, input, backend.View{})
	p.ints = [8]uint32{uint32(weight.Shape[1]), uint32(kernel), uint32(stride), uint32(padding)}

	return nativeConv(d.handle, xb.handle, wb.handle, nil, out.handle, nil, p, false)
}

// Conv2DBackward encodes both MPSGraph convolution derivatives without im2col.
func (d *Device) Conv2DBackward(dx, dw, input, weight, gradient backend.View, options backend.Conv2DOptions) error {
	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	shape, err := convolutionDimensions(input, weight, kernel, stride, padding)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, gradient.Shape) {
		return fmt.Errorf("%w: metal: convolution backward: gradient shape mismatch", backend.ErrInvalidView)
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

	out, err := d.output(dx, input, weight, gradient, dw)
	if err != nil {
		return err
	}

	weights, err := d.output(dw, input, weight, gradient, dx)
	if err != nil {
		return err
	}

	inputCount, _ := elements(input.Shape)

	weightCount, _ := elements(weight.Shape)
	if out.length != inputCount || weights.length != weightCount {
		return fmt.Errorf("%w: metal: convolution backward: destination size mismatch", backend.ErrInvalidView)
	}

	if inputCount == 0 {
		return nativeCompute(d.handle, "fill_values", nil, nil, weights.handle, kernelParams{n: uint32(weightCount)})
	}

	p := viewParams(input.Shape, input, backend.View{})
	p.ints = [8]uint32{uint32(weight.Shape[1]), uint32(kernel), uint32(stride), uint32(padding)}

	return nativeConv(d.handle, xb.handle, wb.handle, gb.handle, out.handle, weights.handle, p, true)
}

// Conv2D writes native MPSGraph convolution with optional on-device bias.
func (d *Device) Conv2D(dst, input, weight, bias backend.View, options backend.Conv2DOptions) error {
	if bias.Buffer == nil {
		return d.conv2D(dst, input, weight, options)
	}

	if len(dst.Shape) != 4 || !slices.Equal(bias.Shape, []int{dst.Shape[1]}) {
		return backend.ErrInvalidView
	}

	d.mu.Lock()
	_, err := d.view(bias)
	d.mu.Unlock()

	if err != nil {
		return err
	}

	temporary, release, err := d.temporary(dst.Shape)
	if err != nil {
		return err
	}
	defer release()

	if err := d.conv2D(temporary, input, weight, options); err != nil {
		return err
	}

	bias.Shape = []int{1, dst.Shape[1], 1, 1}
	bias.Strides = []int{0, bias.Strides[0], 0, 0}

	return d.Binary(backend.Add, dst, temporary, bias)
}
