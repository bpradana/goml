package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// Conv2DBackward allocates input and flattened weight derivatives on the shared
// input device. Gradient shape must match the forward NCHW output exactly. CPU
// uses bounded patch tiles; accelerators execute their native backward capability.
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

	compute, ok := input.device.(backend.ConvolutionBackwardBackend)
	if !ok {
		return nil, nil, unsupported("conv2d backward", input.device)
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

	if err := compute.Conv2DBackward(dx.deviceView(), dw.deviceView(), input.deviceView(), weight.deviceView(), gradient.deviceView(), backend.Conv2DOptions{Kernel: kernel, Stride: stride, Padding: padding}); err != nil {
		_ = dx.Close()
		_ = dw.Close()

		return nil, nil, err
	}

	return dx, dw, nil
}
