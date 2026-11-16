package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
	shapeutil "github.com/bpradana/goml/tensor/internal/shape"
)

func convolutionShape(input []int, kernel, stride, padding int) (int, int, int, error) {
	return shapeutil.Convolution(input, kernel, stride, padding)
}

// Im2Col extracts zero-padded square patches from NCHW input into a row-major
// [batch*outH*outW,channels*kernel*kernel] matrix. Strided inputs are read directly.
// Invalid shapes, configuration, and size overflow return errors.
func Im2Col(input *Tensor, kernel, stride, padding int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input})

	if err := valid("im2col", input); err != nil {
		return nil, err
	}

	outH, outW, features, err := convolutionShape(input.shape, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	rows := input.shape[0] * outH * outW

	compute, ok := input.device.(backend.PatchBackend)
	if !ok {
		return nil, unsupported("im2col", input.device)
	}

	out, err := allocatedOn([]int{rows, features}, input)
	if err != nil {
		return nil, err
	}

	if err := compute.Im2Col(out.deviceView(), input.deviceView(), backend.Conv2DOptions{Kernel: kernel, Stride: stride, Padding: padding}); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

// Col2Im sums a patch matrix into independent NCHW storage of shape, reversing
// Im2Col's extraction. Overlapping patches accumulate in increasing patch order.
// The matrix shape must match the specified convolution exactly.
func Col2Im(patches *Tensor, shape []int, kernel, stride, padding int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{patches})

	if err := valid("col2im", patches); err != nil {
		return nil, err
	}

	if _, _, err := shapeutil.Layout(shape); err != nil {
		return nil, err
	}

	outH, outW, features, err := convolutionShape(shape, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	rows := shape[0] * outH * outW
	if !slices.Equal(patches.shape, []int{rows, features}) {
		return nil, fmt.Errorf("tensor: col2im: incompatible patch shape %v", patches.shape)
	}

	compute, ok := patches.device.(backend.PatchBackend)
	if !ok {
		return nil, unsupported("col2im", patches.device)
	}

	out, err := allocatedOn(shape, patches)
	if err != nil {
		return nil, err
	}

	if err := compute.Col2Im(out.deviceView(), patches.deviceView(), backend.Conv2DOptions{Kernel: kernel, Stride: stride, Padding: padding}); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

// Conv2D computes NCHW cross-correlation using a flattened [channels*kernel*kernel,
// outChannels] weight and optional [outChannels] bias. It returns independent
// contiguous storage, reading arbitrary input and weight strides without copies.
// Temporary patch storage is bounded to 64 spatial positions. Invalid shapes,
// configuration, and size overflow return errors.
func Conv2D(input, weight, bias *Tensor, kernel, stride, padding int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input, weight, bias})

	if err := valid("conv2d", input, weight); err != nil {
		return nil, err
	}

	outH, outW, features, err := convolutionShape(input.shape, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	if weight.Rank() != 2 || weight.shape[0] != features || weight.shape[1] <= 0 {
		return nil, fmt.Errorf("tensor: conv2d: incompatible weight shape %v", weight.shape)
	}

	channels := weight.shape[1]

	if bias != nil {
		if err := valid("conv2d", bias); err != nil {
			return nil, err
		}

		if bias.Rank() != 1 || bias.shape[0] != channels {
			return nil, fmt.Errorf("tensor: conv2d: incompatible bias shape %v", bias.shape)
		}
	}

	if err := compatible("conv2d", input, weight); err != nil {
		return nil, err
	}

	if bias != nil {
		if err := compatible("conv2d", input, bias); err != nil {
			return nil, err
		}
	}

	compute, ok := input.device.(backend.ConvolutionBackend)
	if !ok {
		return nil, unsupported("conv2d", input.device)
	}

	out, err := allocatedOn([]int{input.shape[0], channels, outH, outW}, input, weight)
	if err != nil {
		return nil, err
	}

	var biasView backend.View
	if bias != nil {
		biasView = bias.deviceView()
	}

	if err := compute.Conv2D(out.deviceView(), input.deviceView(), weight.deviceView(), biasView, backend.Conv2DOptions{Kernel: kernel, Stride: stride, Padding: padding}); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}
