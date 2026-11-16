package layer

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// UpsampleMode selects the spatial interpolation used by Upsample2D.
type UpsampleMode string

const (
	// Nearest repeats each source pixel without interpolation.
	Nearest UpsampleMode = "nearest"
	// Bilinear interpolates four neighboring pixels using half-pixel centers.
	Bilinear UpsampleMode = "bilinear"
	// Bicubic interpolates sixteen neighbors with cubic coefficient -0.75.
	Bicubic UpsampleMode = "bicubic"
)

// Upsample2D increases both NCHW spatial dimensions by a positive integer scale.
// Bilinear and bicubic use align_corners=false coordinates and replicate edge
// pixels. Bicubic may overshoot the input range. No antialias filter is applied.
type Upsample2D struct {
	scale int
	mode  UpsampleMode
}

// NewUpsample2D creates interpolation with a positive scale and one of Nearest,
// Bilinear, or Bicubic. Unknown modes and nonpositive scales return errors.
func NewUpsample2D(scale int, mode UpsampleMode) (*Upsample2D, error) {
	if scale <= 0 || (mode != Nearest && mode != Bilinear && mode != Bicubic) {
		return nil, fmt.Errorf("nn: upsample2d: positive scale and nearest, bilinear, or bicubic mode required")
	}

	return &Upsample2D{scale: scale, mode: mode}, nil
}

type interpolationPoint struct {
	indices [4]int
	weights [4]float32
	count   int
}

func interpolationAxis(size, scale int, mode UpsampleMode) []interpolationPoint {
	points := make([]interpolationPoint, size*scale)
	for i := range points {
		point := &points[i]
		if mode == Nearest {
			point.indices[0], point.weights[0], point.count = i/scale, 1, 1
			continue
		}

		position := (float64(i)+0.5)/float64(scale) - 0.5

		base := int(math.Floor(position))
		if mode == Bilinear {
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

// Forward allocates [batch,channels,height*scale,width*scale]. Rank must be four.
// Zero dimensions are supported. Invalid inputs, size overflow, and consumed
// graphs return errors. Backward accumulates interpolation weights into source
// pixels. Inference does not construct a graph. NaN and infinity follow weighted
// IEEE arithmetic; zero-weight contributions are skipped.
func (u *Upsample2D) Forward(input *nn.Value) (*nn.Value, error) {
	if u == nil || u.scale <= 0 || input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: upsample2d: uninitialized layer or input")
	}

	shape := input.Data().Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("nn: upsample2d: expected NCHW, got %v", shape)
	}

	maxInt := int(^uint(0) >> 1)
	if shape[2] > maxInt/u.scale || shape[3] > maxInt/u.scale {
		return nil, fmt.Errorf("nn: upsample2d: spatial shape overflow")
	}

	outH, outW := shape[2]*u.scale, shape[3]*u.scale
	outputShape := []int{shape[0], shape[1], outH, outW}

	storage, err := tensor.Zeros(outputShape...)
	if err != nil {
		return nil, err
	}

	values, output := input.Data().Data(), storage.Data()

	var rows, columns []interpolationPoint
	if len(output) > 0 {
		rows = interpolationAxis(shape[2], u.scale, u.mode)
		columns = interpolationAxis(shape[3], u.scale, u.mode)
	}

	for i := range output {
		plane, pixel := i/(outH*outW), i%(outH*outW)

		row, column := rows[pixel/outW], columns[pixel%outW]
		if u.mode == Nearest {
			output[i] = values[(plane*shape[2]+row.indices[0])*shape[3]+column.indices[0]]
			continue
		}

		for y := range row.count {
			for x := range column.count {
				weight := row.weights[y] * column.weights[x]
				if weight != 0 {
					output[i] += weight * values[(plane*shape[2]+row.indices[y])*shape[3]+column.indices[x]]
				}
			}
		}
	}

	data, err := tensor.From(output, outputShape...)
	if err != nil {
		return nil, err
	}

	if !input.RequiresGrad() {
		return nn.NewOperation(data, input, nil)
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		gradient := make([]float32, len(values))

		for i, value := range g.Data() {
			plane, pixel := i/(outH*outW), i%(outH*outW)

			row, column := rows[pixel/outW], columns[pixel%outW]
			for y := range row.count {
				for x := range column.count {
					weight := row.weights[y] * column.weights[x]
					if weight != 0 {
						gradient[(plane*shape[2]+row.indices[y])*shape[3]+column.indices[x]] += value * weight
					}
				}
			}
		}

		return tensor.From(gradient, shape...)
	})
}

// Parameters returns nil because upsampling has no trainable parameters.
func (*Upsample2D) Parameters() []*nn.Parameter {
	return nil
}
