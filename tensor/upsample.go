package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

func resizeShape(shape []int, scale int, mode string) ([]int, error) {
	if len(shape) != 4 || scale <= 0 || (mode != "nearest" && mode != "bilinear" && mode != "bicubic") {
		return nil, fmt.Errorf("tensor: upsample2d: NCHW shape, positive scale, and supported mode required")
	}

	maximum := int(^uint(0) >> 1)
	if shape[2] > maximum/scale || shape[3] > maximum/scale {
		return nil, fmt.Errorf("tensor: upsample2d: output dimension overflow")
	}

	return []int{shape[0], shape[1], shape[2] * scale, shape[3] * scale}, nil
}

// Upsample2D allocates integer-scale NCHW interpolation on the input device.
// Modes are nearest, bilinear, and bicubic, matching nn/layer's spatial contract.
func interpolationMode(mode string) backend.ResizeMode {
	switch mode {
	case "nearest":
		return backend.Nearest
	case "bilinear":
		return backend.Bilinear
	case "bicubic":
		return backend.Bicubic
	default:
		return 0
	}
}

// Upsample2D allocates integer-scale NCHW interpolation on the input device.
// Modes are nearest, bilinear, and bicubic, matching nn/layer's spatial contract.
func Upsample2D(input *Tensor, scale int, mode string) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input})

	if err := valid("upsample2d", input); err != nil {
		return nil, err
	}

	shape, err := resizeShape(input.shape, scale, mode)
	if err != nil {
		return nil, err
	}

	compute, ok := input.device.(backend.ResizeBackend)
	if !ok {
		return nil, unsupported("resize", input.device)
	}

	output, err := allocatedOn(shape, input)
	if err != nil {
		return nil, err
	}

	options := backend.ResizeOptions{Scale: scale, Mode: interpolationMode(mode)}
	if err := compute.Resize(output.deviceView(), input.deviceView(), options); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}

// Upsample2DBackward accumulates interpolation weights into an independent input
// gradient of shape. The gradient must match the forward output and device.
func Upsample2DBackward(gradient *Tensor, shape []int, scale int, mode string) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{gradient})

	if err := valid("upsample backward", gradient); err != nil {
		return nil, err
	}

	expected, err := resizeShape(shape, scale, mode)
	if err != nil {
		return nil, err
	}

	if !slices.Equal(expected, gradient.shape) {
		return nil, fmt.Errorf("tensor: upsample backward: incompatible gradient shape %v", gradient.shape)
	}

	compute, ok := gradient.device.(backend.ResizeBackwardBackend)
	if !ok {
		return nil, unsupported("resize", gradient.device)
	}

	output, err := allocatedOn(shape, gradient)
	if err != nil {
		return nil, err
	}

	options := backend.ResizeOptions{Scale: scale, Mode: interpolationMode(mode)}
	if err := compute.ResizeBackward(output.deviceView(), gradient.deviceView(), options); err != nil {
		_ = output.Close()
		return nil, err
	}

	return output, nil
}
