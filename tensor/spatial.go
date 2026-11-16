package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

func poolShape(input *Tensor, kernel, stride, padding int) ([]int, error) {
	if err := valid("maxpool2d", input); err != nil {
		return nil, err
	}

	if input.Rank() != 4 || kernel <= 0 || stride <= 0 || padding < 0 || padding > kernel/2 {
		return nil, fmt.Errorf("tensor: maxpool2d: NCHW input and valid pooling geometry required")
	}

	geometry := input.Shape()
	geometry[1] = max(1, geometry[1])

	height, width, _, err := convolutionShape(geometry, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	return []int{input.shape[0], input.shape[1], height, width}, nil
}

// MaxPool2D allocates NCHW spatial maxima on the input device. Padding is negative
// infinity. Finite ties select the first source, and the first NaN propagates.
func MaxPool2D(input *Tensor, kernel, stride, padding int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input})

	shape, err := poolShape(input, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	compute, ok := input.device.(backend.PoolBackend)
	if !ok {
		return nil, unsupported("pool", input.device)
	}

	output, err := allocatedOn(shape, input)
	if err != nil {
		return nil, err
	}

	options := backend.Pool2DOptions{Kernel: kernel, Stride: stride, Padding: padding}
	if err := compute.Pool(output.deviceView(), input.deviceView(), options); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}

// MaxPool2DBackward sums gradients into the selected input positions. It validates
// the exact forward shape and device before allocating an independent result.
func MaxPool2DBackward(input, gradient *Tensor, kernel, stride, padding int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input, gradient})

	shape, err := poolShape(input, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	if err := valid("pool backward", gradient); err != nil {
		return nil, err
	}

	if err := compatible("pool backward", input, gradient); err != nil {
		return nil, err
	}

	if !slices.Equal(gradient.shape, shape) {
		return nil, fmt.Errorf("tensor: pool backward: gradient shape %v must match %v", gradient.shape, shape)
	}

	compute, ok := input.device.(backend.PoolBackwardBackend)
	if !ok {
		return nil, unsupported("pool", input.device)
	}

	output, err := allocatedOn(input.shape, gradient, input)
	if err != nil {
		return nil, err
	}

	options := backend.Pool2DOptions{Kernel: kernel, Stride: stride, Padding: padding}
	if err := compute.PoolBackward(output.deviceView(), input.deviceView(), gradient.deviceView(), options); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}
