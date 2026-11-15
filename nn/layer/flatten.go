package layer

import (
	"fmt"

	"github.com/bpradana/goml/nn"
)

// Flatten preserves the batch dimension and joins all remaining dimensions.
type Flatten struct{}

// NewFlatten constructs a stateless batch-preserving flatten layer.
func NewFlatten() *Flatten { return &Flatten{} }

// Forward maps [batch,d1,...] to [batch,d1*...], materializing noncontiguous
// inputs. Rank below two, uninitialized inputs, and feature-size overflow fail.
func (*Flatten) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: flatten: uninitialized input")
	}

	shape := input.Data().Shape()
	if len(shape) < 2 {
		return nil, fmt.Errorf("nn: flatten: expected at least two dimensions, got %v", shape)
	}

	features := 1

	for _, dim := range shape[1:] {
		if dim == 0 {
			features = 0
			break
		}
	}

	if features != 0 {
		for _, dim := range shape[1:] {
			if dim > int(^uint(0)>>1)/features {
				return nil, fmt.Errorf("nn: flatten: feature size overflow")
			}

			features *= dim
		}
	}

	if !input.Data().IsContiguous() {
		var err error

		input, err = nn.Contiguous(input)
		if err != nil {
			return nil, err
		}
	}

	return nn.Reshape(input, shape[0], features)
}

// Parameters returns nil because Flatten has no trainable parameters.
func (*Flatten) Parameters() []*nn.Parameter { return nil }
