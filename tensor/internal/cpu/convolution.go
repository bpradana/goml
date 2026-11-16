package cpu

import (
	"slices"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/shape"
)

// Im2Col writes zero-padded NCHW patches into independent row-major storage.
func (Device) Im2Col(dst, source backend.View, options backend.Conv2DOptions) error {
	output, err := writeView(dst, true)
	if err != nil {
		return err
	}

	input, err := readView(source)
	if err != nil {
		return err
	}

	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	outH, outW, features, err := shape.Convolution(input.shape, kernel, stride, padding)
	if err != nil {
		return err
	}

	rows := input.shape[0] * outH * outW
	if !slices.Equal(output.shape, []int{rows, features}) || dst.Buffer == source.Buffer {
		return backend.ErrInvalidView
	}

	extractPatches(output.data, &input, kernel, stride, padding, outH, outW, 0, rows)

	return nil
}

func extractPatches(dst []float32, input *array, kernel, stride, padding, outH, outW, start, rows int) {
	features := input.shape[1] * kernel * kernel

	for row := range rows {
		position := start + row
		batch, spatial := position/(outH*outW), position%(outH*outW)
		y, x := spatial/outW*stride-padding, spatial%outW*stride-padding
		patch := dst[row*features : (row+1)*features]
		clear(patch)

		for channel := range input.shape[1] {
			base := input.offset + batch*input.strides[0] + channel*input.strides[1]

			for ky := range kernel {
				iy := y + ky
				if iy < 0 || iy >= input.shape[2] {
					continue
				}

				first, last := max(0, -x), min(kernel, input.shape[3]-x)
				if first >= last {
					continue
				}

				source := base + iy*input.strides[2] + (x+first)*input.strides[3]
				target := (channel*kernel+ky)*kernel + first

				for kx := first; kx < last; kx++ {
					patch[target] = input.data[source]
					source += input.strides[3]
					target++
				}
			}
		}
	}
}

// Col2Im sums a patch matrix into independent NCHW storage of shape, reversing
// Im2Col's extraction. Overlapping patches accumulate in increasing patch order.
// The matrix shape must match the specified convolution exactly.
func (Device) Col2Im(dst, source backend.View, options backend.Conv2DOptions) error {
	target, err := writeView(dst, true)
	if err != nil {
		return err
	}

	sourceArray, err := readView(source)
	if err != nil {
		return err
	}

	out, patches := &target, &sourceArray
	shape := out.shape
	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	outH, outW, features, err := convolutionGeometry(shape, options)
	if err != nil {
		return err
	}

	rows := shape[0] * outH * outW
	if !slices.Equal(patches.shape, []int{rows, features}) || dst.Buffer == source.Buffer {
		return backend.ErrInvalidView
	}

	clear(out.data)

	for row := range rows {
		batch, spatial := row/(outH*outW), row%(outH*outW)
		y, x := spatial/outW*stride-padding, spatial%outW*stride-padding
		base := patches.offset + row*patches.strides[0]

		for channel := range shape[1] {
			for ky := range kernel {
				iy := y + ky
				if iy < 0 || iy >= shape[2] {
					continue
				}

				for kx := max(0, -x); kx < min(kernel, shape[3]-x); kx++ {
					source := base + ((channel*kernel+ky)*kernel+kx)*patches.strides[1]
					target := ((batch*shape[1]+channel)*shape[2]+iy)*shape[3] + x + kx
					out.data[target] += patches.data[source]
				}
			}
		}
	}

	return nil
}

func convolutionGeometry(input []int, options backend.Conv2DOptions) (int, int, int, error) {
	return shape.Convolution(input, options.Kernel, options.Stride, options.Padding)
}

// Conv2D uses bounded patch tiles and fused bias to write NCHW cross-correlation.
func (Device) Conv2D(dst, source, weights, biasView backend.View, options backend.Conv2DOptions) error {
	target, err := writeView(dst, true)
	if err != nil {
		return err
	}

	sourceArray, err := readView(source)
	if err != nil {
		return err
	}

	weightArray, err := readView(weights)
	if err != nil {
		return err
	}

	out, input, weight := &target, &sourceArray, &weightArray
	kernel, stride, padding := options.Kernel, options.Stride, options.Padding

	outH, outW, features, err := shape.Convolution(input.shape, kernel, stride, padding)
	if err != nil {
		return err
	}

	if weight.Rank() != 2 || weight.shape[0] != features || weight.shape[1] <= 0 {
		return backend.ErrInvalidView
	}

	channels := weight.shape[1]
	if !slices.Equal(out.shape, []int{input.shape[0], channels, outH, outW}) || dst.Buffer == source.Buffer || dst.Buffer == weights.Buffer {
		return backend.ErrInvalidView
	}

	var bias *array

	if biasView.Buffer != nil {
		a, err := readView(biasView)
		if err != nil {
			return err
		}

		if !slices.Equal(a.shape, []int{channels}) || dst.Buffer == biasView.Buffer {
			return backend.ErrInvalidView
		}

		bias = &a
	}

	if out.size == 0 {
		return nil
	}

	rows := input.shape[0] * outH * outW
	tile := min(64, rows)
	patches := make([]float32, tile*features)
	products := make([]float32, tile*channels)
	matrix := &array{data: patches, strides: []int{features, 1}}

	for start := 0; start < rows; start += tile {
		count := min(tile, rows-start)
		extractPatches(patches, input, kernel, stride, padding, outH, outW, start, count)
		clear(products)
		matmulAdd(products, matrix, weight, count, features, channels)

		for row := range count {
			position := start + row
			batch, spatial := position/(outH*outW), position%(outH*outW)

			for channel := range channels {
				value := products[row*channels+channel]
				if bias != nil {
					value += bias.data[bias.offset+channel*bias.strides[0]]
				}

				out.data[(batch*channels+channel)*outH*outW+spatial] = value
			}
		}
	}

	return nil
}
