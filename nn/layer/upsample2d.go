package layer

import (
	"fmt"

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

// Forward allocates NCHW interpolation on the input device and records its
// derivative. Invalid geometry, inputs, overflow, and consumed graphs fail.
func (u *Upsample2D) Forward(input *nn.Value) (*nn.Value, error) {
	if u == nil || u.scale <= 0 || input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: upsample2d: uninitialized layer or input")
	}

	data, err := tensor.Upsample2D(input.Data(), u.scale, string(u.mode))
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Upsample2DBackward(g, input.Data().Shape(), u.scale, string(u.mode))
	})
}

// Parameters returns nil because upsampling has no trainable parameters.
func (*Upsample2D) Parameters() []*nn.Parameter {
	return nil
}
