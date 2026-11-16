package cpu

import (
	"slices"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/shape"
)

// Conv2DBackward accumulates input and weight derivatives using bounded patch tiles.
func (Device) Conv2DBackward(dxView, dwView, src, weights, grad backend.View, options backend.Conv2DOptions) error {
	dx, err := writeView(dxView, true)
	if err != nil {
		return err
	}

	dw, err := writeView(dwView, true)
	if err != nil {
		return err
	}

	input, err := readView(src)
	if err != nil {
		return err
	}

	weight, err := readView(weights)
	if err != nil {
		return err
	}

	gradient, err := readView(grad)
	if err != nil {
		return err
	}

	h, w, features, err := shape.Convolution(input.shape, options.Kernel, options.Stride, options.Padding)
	if err != nil {
		return err
	}

	if weight.Rank() != 2 || weight.shape[0] != features || weight.shape[1] <= 0 || !slices.Equal(dx.shape, input.shape) || !slices.Equal(dw.shape, weight.shape) || !slices.Equal(gradient.shape, []int{input.shape[0], weight.shape[1], h, w}) {
		return backend.ErrInvalidView
	}

	for _, output := range []backend.View{dxView, dwView} {
		for _, other := range []backend.View{src, weights, grad} {
			if output.Buffer == other.Buffer {
				return backend.ErrInvalidView
			}
		}
	}

	if dxView.Buffer == dwView.Buffer {
		return backend.ErrInvalidView
	}

	clear(dx.data)
	clear(dw.data)

	rows, channels := input.shape[0]*h*w, weight.shape[1]

	tile := min(64, rows)
	if tile == 0 {
		return nil
	}

	patches := make([]float32, tile*features)
	upstream := make([]float32, tile*channels)
	patchGradient := make([]float32, tile*features)
	transposedWeight := weight
	transposedWeight.shape = []int{channels, features}
	transposedWeight.strides = []int{weight.strides[1], weight.strides[0]}

	for start := 0; start < rows; start += tile {
		count := min(tile, rows-start)
		extractPatches(patches, &input, options.Kernel, options.Stride, options.Padding, h, w, start, count)

		for row := range count {
			position := start + row

			batch, pixel := position/(h*w), position%(h*w)
			for c := range channels {
				upstream[row*channels+c] = gradient.data[gradient.offset+batch*gradient.strides[0]+c*gradient.strides[1]+(pixel/w)*gradient.strides[2]+(pixel%w)*gradient.strides[3]]
			}
		}

		patchTranspose := array{data: patches, strides: []int{1, features}}
		upstreamArray := array{data: upstream, strides: []int{channels, 1}}
		matmulAdd(dw.data, &patchTranspose, &upstreamArray, features, count, channels)
		clear(patchGradient)
		matmulAdd(patchGradient, &upstreamArray, &transposedWeight, count, channels, features)

		for row := range count {
			position := start + row
			batch, pixel := position/(h*w), position%(h*w)

			y, x := (pixel/w)*options.Stride-options.Padding, (pixel%w)*options.Stride-options.Padding
			for c := range input.shape[1] {
				for ky := range options.Kernel {
					iy := y + ky
					if iy < 0 || iy >= input.shape[2] {
						continue
					}

					for kx := max(0, -x); kx < min(options.Kernel, input.shape[3]-x); kx++ {
						feature := (c*options.Kernel+ky)*options.Kernel + kx
						target := ((batch*input.shape[1]+c)*input.shape[2]+iy)*input.shape[3] + x + kx
						dx.data[target] += patchGradient[row*features+feature]
					}
				}
			}
		}
	}

	return nil
}
