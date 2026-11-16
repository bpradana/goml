package tensor

import (
	"fmt"
	"math"
	"runtime"
	"slices"
)

type interpolationPoint struct {
	indices [4]int
	weights [4]float32
	count   int
}

func interpolationAxis(size, scale int, mode string) []interpolationPoint {
	points := make([]interpolationPoint, size*scale)
	for i := range points {
		point := &points[i]
		if mode == "nearest" {
			point.indices[0], point.weights[0], point.count = i/scale, 1, 1
			continue
		}

		position := (float64(i)+0.5)/float64(scale) - 0.5

		base := int(math.Floor(position))
		if mode == "bilinear" {
			fraction := float32(position - float64(base))
			point.count = 2
			point.indices[0], point.indices[1] = max(0, min(size-1, base)), max(0, min(size-1, base+1))
			point.weights[0], point.weights[1] = 1-fraction, fraction

			continue
		}

		point.count = 4

		for j := range 4 {
			source := base + j - 1
			point.indices[j] = max(0, min(size-1, source))
			distance := math.Abs(position - float64(source))

			var weight float64
			if distance <= 1 {
				weight = (1.25*distance-2.25)*distance*distance + 1
			} else if distance < 2 {
				weight = ((-0.75*distance+3.75)*distance-6)*distance + 3
			}

			point.weights[j] = float32(weight)
		}
	}

	return points
}

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
func Upsample2D(input *Tensor, scale int, mode string) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{input})

	if err := valid("upsample2d", input); err != nil {
		return nil, err
	}

	shape, err := resizeShape(input.shape, scale, mode)
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
			return nil, fmt.Errorf("tensor: device lacks interpolation")
		}

		if err := backend.Resize(output.gpu.storage.buffer, input.deviceView(), scale, mode); err != nil {
			_ = output.Close()
			return nil, err
		}

		return output, nil
	}

	if output.size == 0 {
		return output, nil
	}

	if !input.IsContiguous() {
		input = input.Contiguous()
	}

	rows := interpolationAxis(input.shape[2], scale, mode)
	columns := interpolationAxis(input.shape[3], scale, mode)

	for index := range output.data {
		plane, pixel := index/(shape[2]*shape[3]), index%(shape[2]*shape[3])

		row, column := rows[pixel/shape[3]], columns[pixel%shape[3]]
		if mode == "nearest" {
			source := (plane*input.shape[2]+row.indices[0])*input.shape[3] + column.indices[0]
			output.data[index] = input.data[input.offset+source]

			continue
		}

		for y := range row.count {
			for x := range column.count {
				weight := row.weights[y] * column.weights[x]
				if weight != 0 {
					source := (plane*input.shape[2]+row.indices[y])*input.shape[3] + column.indices[x]
					output.data[index] += weight * input.data[input.offset+source]
				}
			}
		}
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

	output, err := allocatedOn(shape, gradient)
	if err != nil {
		return nil, err
	}

	if gradient.gpu != nil {
		backend, ok := gradient.Device().(SpatialBackend)
		if !ok {
			_ = output.Close()
			return nil, fmt.Errorf("tensor: device lacks interpolation gradients")
		}

		if err := backend.ResizeBackward(output.gpu.storage.buffer, gradient.deviceView(), shape, scale, mode); err != nil {
			_ = output.Close()
			return nil, err
		}

		return output, nil
	}

	if gradient.size == 0 {
		return output, nil
	}

	if !gradient.IsContiguous() {
		gradient = gradient.Contiguous()
	}

	rows, columns := interpolationAxis(shape[2], scale, mode), interpolationAxis(shape[3], scale, mode)

	for index := range gradient.size {
		plane, pixel := index/(expected[2]*expected[3]), index%(expected[2]*expected[3])
		row, column := rows[pixel/expected[3]], columns[pixel%expected[3]]
		value := gradient.data[gradient.offset+index]

		for y := range row.count {
			for x := range column.count {
				weight := row.weights[y] * column.weights[x]
				if weight != 0 {
					target := (plane*shape[2]+row.indices[y])*shape[3] + column.indices[x]
					output.data[target] += value * weight
				}
			}
		}
	}

	return output, nil
}
