package cpu

import (
	"math"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

type interpolationPoint struct {
	indices [4]int
	weights [4]float32
	count   int
}

func interpolationAxis(size, scale int, mode backend.ResizeMode) []interpolationPoint {
	points := make([]interpolationPoint, size*scale)
	for i := range points {
		point := &points[i]
		if mode == backend.Nearest {
			point.indices[0], point.weights[0], point.count = i/scale, 1, 1
			continue
		}

		position := (float64(i)+0.5)/float64(scale) - 0.5

		base := int(math.Floor(position))
		if mode == backend.Bilinear {
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

func resizeShape(input []int, options backend.ResizeOptions) ([]int, error) {
	if len(input) != 4 || options.Scale <= 0 || options.Mode < backend.Nearest || options.Mode > backend.Bicubic {
		return nil, backend.ErrUnsupported
	}

	limit := int(^uint(0)>>1) / options.Scale
	if input[2] > limit || input[3] > limit {
		return nil, backend.ErrInvalidView
	}

	return []int{input[0], input[1], input[2] * options.Scale, input[3] * options.Scale}, nil
}

// Resize writes integer-scale NCHW interpolation into a contiguous destination.
func (Device) Resize(dst, src backend.View, options backend.ResizeOptions) error {
	outputArray, err := writeView(dst, true)
	if err != nil {
		return err
	}

	inputArray, err := readView(src)
	if err != nil {
		return err
	}

	input, output := &inputArray, &outputArray

	shape, err := resizeShape(input.shape, options)
	if err != nil {
		return err
	}

	if !slices.Equal(shape, output.shape) || dst.Buffer == src.Buffer {
		return backend.ErrInvalidView
	}

	scale, mode := options.Scale, options.Mode

	clear(output.data)

	if output.size == 0 {
		return nil
	}

	if !input.IsContiguous() {
		input = input.Contiguous()
	}

	rows := interpolationAxis(input.shape[2], scale, mode)
	columns := interpolationAxis(input.shape[3], scale, mode)

	for index := range output.data {
		plane, pixel := index/(shape[2]*shape[3]), index%(shape[2]*shape[3])

		row, column := rows[pixel/shape[3]], columns[pixel%shape[3]]
		if mode == backend.Nearest {
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

	return nil
}

// ResizeBackward sums interpolation derivatives into the original input shape.
func (Device) ResizeBackward(dst, grad backend.View, options backend.ResizeOptions) error {
	outputArray, err := writeView(dst, true)
	if err != nil {
		return err
	}

	gradientArray, err := readView(grad)
	if err != nil {
		return err
	}

	output, gradient := &outputArray, &gradientArray
	shape := output.shape

	expected, err := resizeShape(shape, options)
	if err != nil {
		return err
	}

	if !slices.Equal(expected, gradient.shape) || dst.Buffer == grad.Buffer {
		return backend.ErrInvalidView
	}

	scale, mode := options.Scale, options.Mode

	clear(output.data)

	if gradient.size == 0 {
		return nil
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

	return nil
}
