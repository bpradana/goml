package webgpu

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
		return nil, fmt.Errorf("webgpu: spatial: %w: invalid geometry", backend.ErrInvalidView)
	}

	const limit = 1<<31 - 1
	if kernel > limit || stride > limit || shape[2] > limit || shape[3] > limit ||
		padding > (limit-shape[2])/2 || padding > (limit-shape[3])/2 {
		return nil, fmt.Errorf("webgpu: %w: spatial coordinates exceed signed 32 bits", backend.ErrUnsupported)
	}

	height, width := shape[2]+2*padding, shape[3]+2*padding
	if kernel > height || kernel > width {
		return nil, fmt.Errorf("webgpu: spatial: %w: kernel exceeds padded dimensions", backend.ErrInvalidView)
	}

	result := []int{shape[0], shape[1], (height-kernel)/stride + 1, (width-kernel)/stride + 1}
	if _, err := elements(result); err != nil {
		return nil, err
	}

	return result, nil
}

func poolParams(dst, input, gradient backend.View, shape []int, options backend.Pool2DOptions) kernelParams {
	p := params(dst, input, gradient, backend.View{})
	p.ints[0], p.ints[1] = uint32(input.Shape[2]), uint32(input.Shape[3])
	p.ints[2], p.ints[3] = uint32(shape[2]), uint32(shape[3])
	p.ints[4], p.ints[5], p.ints[6] = uint32(options.Kernel), uint32(options.Stride), uint32(options.Padding)

	return p
}

// Pool writes floor-sized NCHW maximum pooling with negative-infinity padding.
// Ties and NaNs select the first input in row-major order, matching CPU semantics.
func (d *Device) Pool(dst, input backend.View, options backend.Pool2DOptions) error {
	shape, err := spatialShape(input.Shape, options.Kernel, options.Stride, options.Padding)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, dst.Shape) {
		return fmt.Errorf("webgpu: pool: %w: output shape mismatch", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, input)
	if err != nil {
		return err
	}

	return d.dispatch("pool_values", poolParams(dst, input, backend.View{}, shape, options), handles)
}

// PoolBackward gathers all overlapping-window contributions into input-shaped
// gradients without atomics. Shapes and ownership are validated before encoding.
func (d *Device) PoolBackward(dst, input, gradient backend.View, options backend.Pool2DOptions) error {
	shape, err := spatialShape(input.Shape, options.Kernel, options.Stride, options.Padding)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, gradient.Shape) || !slices.Equal(dst.Shape, input.Shape) {
		return fmt.Errorf("webgpu: pool backward: %w: gradient shape mismatch", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, input, gradient)
	if err != nil {
		return err
	}

	return d.dispatch("pool_gradients", poolParams(dst, input, gradient, shape, options), handles)
}

func resizedShape(shape []int, options backend.ResizeOptions) ([]int, error) {
	if _, err := elements(shape); err != nil {
		return nil, err
	}

	if options.Mode < backend.Nearest || options.Mode > backend.Bicubic {
		return nil, fmt.Errorf("webgpu: %w: interpolation %v", backend.ErrUnsupported, options.Mode)
	}

	if len(shape) != 4 || options.Scale <= 0 {
		return nil, fmt.Errorf("webgpu: resize: %w: invalid geometry", backend.ErrInvalidView)
	}

	// Coordinates include a small interpolation halo in the backward shader.
	const limit = (1 << 29) - 1
	if options.Scale > limit || shape[2] > limit/options.Scale || shape[3] > limit/options.Scale {
		return nil, fmt.Errorf("webgpu: %w: interpolation coordinates exceed shader limits", backend.ErrUnsupported)
	}

	result := []int{shape[0], shape[1], shape[2] * options.Scale, shape[3] * options.Scale}
	if _, err := elements(result); err != nil {
		return nil, err
	}

	return result, nil
}

func resizeParams(dst, src backend.View, inputShape, outputShape []int, options backend.ResizeOptions) kernelParams {
	p := params(dst, src, backend.View{}, backend.View{})
	p.ints[0], p.ints[1] = uint32(inputShape[2]), uint32(inputShape[3])
	p.ints[2], p.ints[3] = uint32(outputShape[2]), uint32(outputShape[3])
	p.ints[4], p.ints[5] = uint32(options.Scale), uint32(options.Mode)

	return p
}

// Resize writes integer-scale NCHW nearest, bilinear, or bicubic interpolation.
// Linear and cubic modes use half-pixel coordinates with align_corners=false;
// bicubic uses coefficient -0.75. Output values remain GPU-resident.
func (d *Device) Resize(dst, input backend.View, options backend.ResizeOptions) error {
	shape, err := resizedShape(input.Shape, options)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, dst.Shape) {
		return fmt.Errorf("webgpu: resize: %w: output shape mismatch", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, input)
	if err != nil {
		return err
	}

	return d.dispatch("resize_values", resizeParams(dst, input, input.Shape, shape, options), handles)
}

// ResizeBackward writes input-shaped interpolation derivatives by gathering a
// bounded neighborhood. Repeated clamped boundary samples accumulate correctly.
func (d *Device) ResizeBackward(dst, gradient backend.View, options backend.ResizeOptions) error {
	shape, err := resizedShape(dst.Shape, options)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, gradient.Shape) {
		return fmt.Errorf("webgpu: resize backward: %w: gradient shape mismatch", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, gradient)
	if err != nil {
		return err
	}

	return d.dispatch("resize_gradients", resizeParams(dst, gradient, dst.Shape, shape, options), handles)
}

var (
	_ backend.PoolBackend           = (*Device)(nil)
	_ backend.PoolBackwardBackend   = (*Device)(nil)
	_ backend.ResizeBackend         = (*Device)(nil)
	_ backend.ResizeBackwardBackend = (*Device)(nil)
)
