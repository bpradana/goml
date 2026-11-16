package tensor

import (
	"fmt"
	"math"
	"runtime"
	"slices"
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

	output, err := allocatedOn(shape, input)
	if err != nil {
		return nil, err
	}

	if input.gpu != nil {
		backend, ok := input.Device().(SpatialBackend)
		if !ok {
			_ = output.Close()
			return nil, fmt.Errorf("tensor: device lacks pooling")
		}

		if err := backend.Pool(output.gpu.storage.buffer, input.deviceView(), kernel, stride, padding); err != nil {
			_ = output.Close()
			return nil, err
		}

		return output, nil
	}

	if !input.IsContiguous() {
		input = input.Contiguous()
	}

	for index := range output.data {
		best, _ := poolMaximum(input, index, shape[2], shape[3], kernel, stride, padding)
		output.data[index] = best
	}

	return output, nil
}

func poolMaximum(input *Tensor, index, outH, outW, kernel, stride, padding int) (float32, int) {
	plane, pixel := index/(outH*outW), index%(outH*outW)
	row, column := pixel/outW, pixel%outW
	height, width := input.shape[2], input.shape[3]
	best, selected := float32(math.Inf(-1)), -1

	for ky := range kernel {
		for kx := range kernel {
			y, x := row*stride-padding+ky, column*stride-padding+kx
			if y < 0 || y >= height || x < 0 || x >= width {
				continue
			}

			logical := (plane*height+y)*width + x

			value := input.data[input.offset+logical]
			if !math.IsNaN(float64(best)) && (selected < 0 || value > best || math.IsNaN(float64(value))) {
				best, selected = value, logical
			}
		}
	}

	return best, selected
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

	output, err := allocatedOn(input.shape, gradient, input)
	if err != nil {
		return nil, err
	}

	if input.gpu != nil {
		backend, ok := input.Device().(SpatialBackend)
		if !ok {
			_ = output.Close()
			return nil, fmt.Errorf("tensor: device lacks pooling gradients")
		}

		if err := backend.PoolBackward(output.gpu.storage.buffer, input.deviceView(), gradient.deviceView(), kernel, stride, padding); err != nil {
			_ = output.Close()
			return nil, err
		}

		return output, nil
	}

	if !input.IsContiguous() {
		input = input.Contiguous()
	}

	if !gradient.IsContiguous() {
		gradient = gradient.Contiguous()
	}

	for index := range gradient.size {
		_, selected := poolMaximum(input, index, shape[2], shape[3], kernel, stride, padding)
		if selected >= 0 {
			output.data[selected] += gradient.data[gradient.offset+index]
		}
	}

	return output, nil
}
