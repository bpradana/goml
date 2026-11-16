package tensor

import (
	"fmt"
	"slices"

	internalshape "github.com/bpradana/goml/tensor/internal/shape"
)

func convolutionShape(shape []int, kernel, stride, padding int) (int, int, int, error) {
	if len(shape) != 4 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 || kernel <= 0 || stride <= 0 || padding < 0 {
		return 0, 0, 0, fmt.Errorf("tensor: convolution: expected NCHW input, positive channels, spatial dimensions, kernel and stride, and nonnegative padding")
	}

	maxInt := int(^uint(0) >> 1)
	if padding > (maxInt-shape[2])/2 || padding > (maxInt-shape[3])/2 || kernel > maxInt/shape[1] || kernel > maxInt/(shape[1]*kernel) {
		return 0, 0, 0, fmt.Errorf("tensor: convolution: padded shape or kernel size overflow")
	}

	height, width := shape[2]+2*padding, shape[3]+2*padding
	if kernel > height || kernel > width {
		return 0, 0, 0, fmt.Errorf("tensor: convolution: kernel exceeds padded spatial dimensions")
	}

	outH, outW, features := (height-kernel)/stride+1, (width-kernel)/stride+1, shape[1]*kernel*kernel
	if _, _, err := internalshape.Layout([]int{shape[0], outH, outW, features}); err != nil {
		return 0, 0, 0, err
	}

	return outH, outW, features, nil
}

// Im2Col extracts zero-padded square patches from NCHW input into a row-major
// [batch*outH*outW,channels*kernel*kernel] matrix. Strided inputs are read directly.
// Invalid shapes, configuration, and size overflow return errors.
func Im2Col(input *Tensor, kernel, stride, padding int) (*Tensor, error) {
	if err := valid("im2col", input); err != nil {
		return nil, err
	}

	outH, outW, features, err := convolutionShape(input.shape, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	rows := input.shape[0] * outH * outW

	out, err := allocated([]int{rows, features})
	if err != nil {
		return nil, err
	}

	extractPatches(out.data, input, kernel, stride, padding, outH, outW, 0, rows)

	return out, nil
}

func extractPatches(dst []float32, input *Tensor, kernel, stride, padding, outH, outW, start, rows int) {
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
func Col2Im(patches *Tensor, shape []int, kernel, stride, padding int) (*Tensor, error) {
	if err := valid("col2im", patches); err != nil {
		return nil, err
	}

	if _, _, err := internalshape.Layout(shape); err != nil {
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

	out, err := allocated(shape)
	if err != nil {
		return nil, err
	}

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

	return out, nil
}

// Conv2D computes NCHW cross-correlation using a flattened [channels*kernel*kernel,
// outChannels] weight and optional [outChannels] bias. It returns independent
// contiguous storage, reading arbitrary input and weight strides without copies.
// Temporary patch storage is bounded to 64 spatial positions. Invalid shapes,
// configuration, and size overflow return errors.
func Conv2D(input, weight, bias *Tensor, kernel, stride, padding int) (*Tensor, error) {
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

	out, err := allocated([]int{input.shape[0], channels, outH, outW})
	if err != nil || out.size == 0 {
		return out, err
	}

	rows := input.shape[0] * outH * outW
	tile := min(64, rows)
	patches := make([]float32, tile*features)
	products := make([]float32, tile*channels)
	matrix := &Tensor{data: patches, strides: []int{features, 1}}

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

	return out, nil
}
