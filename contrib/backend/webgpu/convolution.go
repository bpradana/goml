package webgpu

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

func convolutionShape(input, weight backend.View, options backend.Conv2DOptions) ([]int, error) {
	shape, err := spatialShape(input.Shape, options.Kernel, options.Stride, options.Padding)
	if err != nil {
		return nil, err
	}

	features := uint64(input.Shape[1])
	for range 2 {
		if features > uint64(^uint32(0))/uint64(options.Kernel) {
			return nil, fmt.Errorf("webgpu: %w: convolution feature count exceeds 32 bits", backend.ErrUnsupported)
		}

		features *= uint64(options.Kernel)
	}

	if len(weight.Shape) != 2 || uint64(weight.Shape[0]) != features {
		return nil, fmt.Errorf("webgpu: convolution: %w: weight shape mismatch", backend.ErrInvalidView)
	}

	shape[1] = weight.Shape[1]
	if _, err := elements(shape); err != nil {
		return nil, err
	}

	return shape, nil
}

func convolutionParams(dst, input, weight, third backend.View, outputShape []int, options backend.Conv2DOptions) kernelParams {
	p := params(dst, input, weight, third)
	for axis, size := range input.Shape {
		p.ints[axis] = uint32(size)
	}

	p.ints[4], p.ints[5], p.ints[6] = uint32(outputShape[1]), uint32(outputShape[2]), uint32(outputShape[3])
	p.ints[7], p.ints[8], p.ints[9] = uint32(options.Kernel), uint32(options.Stride), uint32(options.Padding)

	return p
}

// Conv2D writes NCHW cross-correlation using flattened [in*kernel*kernel,out]
// weights and optional [out] bias. WGSL reads GPU buffers directly, including
// strided views; no CPU patch extraction or intermediate transfer occurs.
func (d *Device) Conv2D(dst, input, weight, bias backend.View, options backend.Conv2DOptions) error {
	shape, err := convolutionShape(input, weight, options)
	if err != nil {
		return err
	}

	if !slices.Equal(dst.Shape, shape) || (bias.Buffer != nil && !slices.Equal(bias.Shape, []int{shape[1]})) {
		return fmt.Errorf("webgpu: convolution: %w: output or bias shape mismatch", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	sources := []backend.View{input, weight}
	if bias.Buffer != nil {
		sources = append(sources, bias)
	}

	handles, err := d.bindings(dst, sources...)
	if err != nil {
		return err
	}

	p := convolutionParams(dst, input, weight, bias, shape, options)
	if bias.Buffer != nil {
		p.flags = 1
	}

	return d.dispatch("conv_values", p, handles)
}

// Conv2DBackward writes input and flattened-weight gradients on the GPU. Each
// invocation gathers one derivative, avoiding floating-point atomic writes.
// Destinations must match their respective inputs and must not alias each other.
func (d *Device) Conv2DBackward(dx, dw, input, weight, gradient backend.View, options backend.Conv2DOptions) error {
	shape, err := convolutionShape(input, weight, options)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, gradient.Shape) || !slices.Equal(dx.Shape, input.Shape) || !slices.Equal(dw.Shape, weight.Shape) {
		return fmt.Errorf("webgpu: convolution backward: %w: gradient shape mismatch", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	xHandles, err := d.bindings(dx, input, weight, gradient)
	if err != nil {
		return err
	}

	wHandles, err := d.bindings(dw, input, weight, gradient)
	if err != nil {
		return err
	}

	if xHandles[0] == wHandles[0] {
		return fmt.Errorf("webgpu: convolution backward: %w: destinations alias", backend.ErrInvalidView)
	}

	if err := d.dispatch("conv_input_gradients", convolutionParams(dx, input, weight, gradient, shape, options), xHandles); err != nil {
		return err
	}

	return d.dispatch("conv_weight_gradients", convolutionParams(dw, input, weight, gradient, shape, options), wHandles)
}

var (
	_ backend.ConvolutionBackend         = (*Device)(nil)
	_ backend.ConvolutionBackwardBackend = (*Device)(nil)
)
