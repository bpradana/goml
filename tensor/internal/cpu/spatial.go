package cpu

import (
	"math"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/shape"
)

func poolingShape(input *array, options backend.Pool2DOptions) ([]int, error) {
	if input.Rank() != 4 || options.Padding > options.Kernel/2 {
		return nil, backend.ErrInvalidView
	}

	geometry := input.Shape()
	geometry[1] = max(1, geometry[1])
	h, w, _, err := shape.Convolution(geometry, options.Kernel, options.Stride, options.Padding)

	return []int{input.shape[0], input.shape[1], h, w}, err
}

// Pool writes NCHW maxima, selecting the first finite tie or NaN.
func (Device) Pool(dst, src backend.View, options backend.Pool2DOptions) error {
	outputArray, err := writeView(dst, true)
	if err != nil {
		return err
	}

	inputArray, err := readView(src)
	if err != nil {
		return err
	}

	input, output := &inputArray, &outputArray

	shape, err := poolingShape(input, options)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, output.shape) || dst.Buffer == src.Buffer {
		return backend.ErrInvalidView
	}

	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	if !input.IsContiguous() {
		input = input.Contiguous()
	}

	for index := range output.data {
		best, _ := poolMaximum(input, index, shape[2], shape[3], kernel, stride, padding)
		output.data[index] = best
	}

	return nil
}

func poolMaximum(input *array, index, outH, outW, kernel, stride, padding int) (float32, int) {
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
func (Device) PoolBackward(dst, src, grad backend.View, options backend.Pool2DOptions) error {
	outputArray, err := writeView(dst, true)
	if err != nil {
		return err
	}

	inputArray, err := readView(src)
	if err != nil {
		return err
	}

	gradientArray, err := readView(grad)
	if err != nil {
		return err
	}

	input, output, gradient := &inputArray, &outputArray, &gradientArray

	shape, err := poolingShape(input, options)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, gradient.shape) || !slices.Equal(input.shape, output.shape) || dst.Buffer == src.Buffer || dst.Buffer == grad.Buffer {
		return backend.ErrInvalidView
	}

	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	clear(output.data)

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

	return nil
}
