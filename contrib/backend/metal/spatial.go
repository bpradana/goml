package metal

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

func spatialShape(shape []int, kernel, stride, padding int) ([]int, error) {
	if _, err := elements(shape); err != nil {
		return nil, err
	}

	if len(shape) != 4 || shape[2] <= 0 || shape[3] <= 0 || kernel <= 0 || stride <= 0 || padding < 0 {
		return nil, fmt.Errorf("%w: Metal spatial geometry", backend.ErrInvalidView)
	}

	const limit = 1<<31 - 1
	if kernel > limit || stride > limit || shape[2] > limit || shape[3] > limit || padding > (limit-shape[2])/2 || padding > (limit-shape[3])/2 {
		return nil, fmt.Errorf("%w: Metal spatial indexing exceeds signed 32 bits", backend.ErrUnsupported)
	}

	height, width := shape[2]+2*padding, shape[3]+2*padding
	if kernel > height || kernel > width {
		return nil, fmt.Errorf("%w: metal: spatial operation: kernel exceeds padded dimensions", backend.ErrInvalidView)
	}

	output := []int{shape[0], shape[1], (height-kernel)/stride + 1, (width-kernel)/stride + 1}
	if _, err := elements(output); err != nil {
		return nil, err
	}

	return output, nil
}

func resizedShape(shape []int, scale int) ([]int, error) {
	if _, err := elements(shape); err != nil {
		return nil, err
	}

	if len(shape) != 4 || scale <= 0 {
		return nil, fmt.Errorf("%w: Metal resize geometry", backend.ErrInvalidView)
	}

	const limit = 1<<31 - 1
	if scale > limit || shape[2] > limit/scale || shape[3] > limit/scale {
		return nil, fmt.Errorf("%w: Metal resize indexing exceeds signed 32 bits", backend.ErrUnsupported)
	}

	output := []int{shape[0], shape[1], shape[2] * scale, shape[3] * scale}
	if _, err := elements(output); err != nil {
		return nil, err
	}

	return output, nil
}

// Pool encodes floor-size NCHW maximum pooling with negative-infinity padding.
func (d *Device) Pool(dst, input backend.View, options backend.Pool2DOptions) error {
	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	shape, err := spatialShape(input.Shape, kernel, stride, padding)
	if err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	source, err := d.view(input)
	if err != nil {
		return err
	}

	output, err := d.output(dst, input)
	if err != nil {
		return err
	}

	count, _ := elements(shape)
	if output.length != count {
		return fmt.Errorf("%w: metal: pool: output size mismatch", backend.ErrInvalidView)
	}

	p := viewParams(input.Shape, input, backend.View{})
	p.n = uint32(output.length)
	height := (input.Shape[2]+2*padding-kernel)/stride + 1
	width := (input.Shape[3]+2*padding-kernel)/stride + 1
	p.ints = [8]uint32{uint32(kernel), uint32(stride), uint32(padding), uint32(height), uint32(width)}

	return nativeCompute(d.handle, "pool_values", source.handle, nil, output.handle, p)
}

// PoolBackward gathers overlapping-window contributions into input gradients.
func (d *Device) PoolBackward(dst, input, gradient backend.View, options backend.Pool2DOptions) error {
	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	shape, err := spatialShape(input.Shape, kernel, stride, padding)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, gradient.Shape) {
		return fmt.Errorf("%w: metal: pool backward: gradient shape mismatch", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	source, err := d.view(input)
	if err != nil {
		return err
	}

	g, err := d.view(gradient)
	if err != nil {
		return err
	}

	output, err := d.output(dst, input, gradient)
	if err != nil {
		return err
	}

	count, _ := elements(input.Shape)
	if output.length != count {
		return fmt.Errorf("%w: metal: pool backward: output size mismatch", backend.ErrInvalidView)
	}

	p := viewParams(input.Shape, input, gradient)
	p.n = uint32(output.length)
	p.ints = [8]uint32{uint32(kernel), uint32(stride), uint32(padding), uint32(gradient.Shape[2]), uint32(gradient.Shape[3])}

	return nativeCompute(d.handle, "pool_gradients", source.handle, g.handle, output.handle, p)
}

func resizeMode(mode backend.ResizeMode) (uint32, error) {
	switch mode {
	case backend.Nearest:
		return 0, nil
	case backend.Bilinear:
		return 1, nil
	case backend.Bicubic:
		return 2, nil
	default:
		return 0, fmt.Errorf("%w: Metal interpolation %d", backend.ErrUnsupported, mode)
	}
}

// Resize encodes integer-scale NCHW nearest, bilinear, or bicubic interpolation.
func (d *Device) Resize(dst, input backend.View, options backend.ResizeOptions) error {
	scale, mode := options.Scale, options.Mode

	shape, err := resizedShape(input.Shape, scale)
	if err != nil {
		return err
	}

	code, err := resizeMode(mode)
	if err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	source, err := d.view(input)
	if err != nil {
		return err
	}

	output, err := d.output(dst, input)
	if err != nil {
		return err
	}

	count, _ := elements(shape)
	if output.length != count {
		return fmt.Errorf("%w: metal: resize: output size mismatch", backend.ErrInvalidView)
	}

	p := viewParams(input.Shape, input, backend.View{})
	p.n = uint32(output.length)
	p.ints = [8]uint32{uint32(input.Shape[2] * scale), uint32(input.Shape[3] * scale), uint32(scale), code}

	return nativeCompute(d.handle, "resize_values", source.handle, nil, output.handle, p)
}

// ResizeBackward gathers interpolation weights without atomic floating-point writes.
func (d *Device) ResizeBackward(dst, gradient backend.View, options backend.ResizeOptions) error {
	shape, scale, mode := dst.Shape, options.Scale, options.Mode

	expected, err := resizedShape(shape, scale)
	if err != nil {
		return err
	}

	if !slices.Equal(expected, gradient.Shape) {
		return fmt.Errorf("%w: metal: resize backward: gradient shape mismatch", backend.ErrInvalidView)
	}

	code, err := resizeMode(mode)
	if err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	g, err := d.view(gradient)
	if err != nil {
		return err
	}

	output, err := d.output(dst, gradient)
	if err != nil {
		return err
	}

	count, _ := elements(shape)
	if output.length != count {
		return fmt.Errorf("%w: metal: resize backward: output size mismatch", backend.ErrInvalidView)
	}

	p := viewParams(gradient.Shape, gradient, backend.View{})

	p.n = uint32(output.length)
	for axis, size := range shape {
		p.shape[axis] = uint32(size)
	}

	p.ints = [8]uint32{uint32(gradient.Shape[2]), uint32(gradient.Shape[3]), uint32(scale), code}

	return nativeCompute(d.handle, "resize_gradients", g.handle, nil, output.handle, p)
}
