package layer

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// PositionalEncoding adds fixed sinusoidal position information to sequences.
type PositionalEncoding struct{}

// NewPositionalEncoding constructs a stateless sinusoidal encoding layer.
func NewPositionalEncoding() *PositionalEncoding {
	return &PositionalEncoding{}
}

// Forward adds sin(position/10000^(2*i/features)) on even features and cosine on
// odd features. Input must be [batch,time,features] with positive time and features.
// It preserves shape, allocates a result, and leaves the input gradient unchanged.
func (*PositionalEncoding) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: positional encoding: uninitialized input")
	}

	shape := input.Data().Shape()
	if len(shape) != 3 || shape[1] <= 0 || shape[2] <= 0 {
		return nil, fmt.Errorf("nn: positional encoding: expected [batch,time>0,features>0], got %v", shape)
	}

	encoding, err := tensor.Zeros(shape[1], shape[2])
	if err != nil {
		return nil, err
	}

	values, err := encoding.HostData()
	if err != nil {
		return nil, err
	}

	for position := range shape[1] {
		for feature := range shape[2] {
			angle := float64(position) / math.Pow(10000, float64(2*(feature/2))/float64(shape[2]))
			if feature%2 == 0 {
				values[position*shape[2]+feature] = float32(math.Sin(angle))
			} else {
				values[position*shape[2]+feature] = float32(math.Cos(angle))
			}
		}
	}

	encoding, err = tensor.FromLike(input.Data(), values, shape[1], shape[2])
	if err != nil {
		return nil, err
	}

	constant, err := nn.NewValue(encoding, false)
	if err != nil {
		return nil, err
	}

	return nn.Add(input, constant)
}

// Parameters returns nil because positional encoding has no trainable state.
func (*PositionalEncoding) Parameters() []*nn.Parameter {
	return nil
}
