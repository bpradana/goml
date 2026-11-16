package tensor

import (
	"fmt"
	"runtime"
	"slices"
)

func gpuConv2D(input, weight, bias *Tensor, kernel, stride, padding int, shape []int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input, weight, bias})

	backend, ok := input.Device().(ConvolutionBackend)
	if !ok {
		return nil, fmt.Errorf("tensor: device lacks convolution")
	}

	out, err := allocatedOn(shape, input, weight)
	if err != nil {
		return nil, err
	}

	if err := backend.Conv2D(out.gpu.storage.buffer, input.deviceView(), weight.deviceView(), kernel, stride, padding); err != nil {
		_ = out.Close()
		return nil, err
	}

	if bias == nil {
		return out, nil
	}

	view, err := bias.Reshape(1, shape[1], 1, 1)
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	defer func() { _ = view.Close(); _ = out.Close() }()

	return Add(out, view)
}

// Conv2DBackward computes NCHW input and flattened kernel derivatives on-device.
// Device implementations use native convolution derivatives, never host im2col.
func Conv2DBackward(input, weight, gradient *Tensor, kernel, stride, padding int) (*Tensor, *Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input, weight, gradient})

	if err := valid("conv2d backward", input, weight, gradient); err != nil {
		return nil, nil, err
	}

	if err := compatible("conv2d backward", input, weight, gradient); err != nil {
		return nil, nil, err
	}

	height, width, features, err := convolutionShape(input.shape, kernel, stride, padding)
	if err != nil {
		return nil, nil, err
	}

	if weight.Rank() != 2 || weight.shape[0] != features || weight.shape[1] <= 0 {
		return nil, nil, fmt.Errorf("tensor: conv2d backward: incompatible weight shape %v", weight.shape)
	}

	if !slices.Equal(gradient.shape, []int{input.shape[0], weight.shape[1], height, width}) {
		return nil, nil, fmt.Errorf("tensor: conv2d backward: incompatible gradient shape %v", gradient.shape)
	}

	if input.gpu == nil {
		return nil, nil, fmt.Errorf("tensor: CPU convolution differentiation uses Im2Col and Col2Im")
	}

	backend, ok := input.Device().(ConvolutionBackend)
	if !ok {
		return nil, nil, fmt.Errorf("tensor: device lacks convolution gradients")
	}

	dx, err := allocatedOn(input.shape, input, gradient)
	if err != nil {
		return nil, nil, err
	}

	dw, err := allocatedOn(weight.shape, gradient, weight)
	if err != nil {
		_ = dx.Close()
		return nil, nil, err
	}

	if err := backend.Conv2DBackward(dx.gpu.storage.buffer, dw.gpu.storage.buffer, input.deviceView(), weight.deviceView(), gradient.deviceView(), kernel, stride, padding); err != nil {
		_ = dx.Close()
		_ = dw.Close()

		return nil, nil, err
	}

	return dx, dw, nil
}
