package activation

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// Softmax normalizes exponentials along one axis without changing shape.
type Softmax struct{ axis int }

// NewSoftmax constructs a stateless softmax layer. Forward validates axis against
// the input rank; axes are nonnegative and the selected dimension must be nonzero.
func NewSoftmax(axis int) *Softmax { return &Softmax{axis: axis} }

// Axis returns the normalization axis used by Forward and architecture snapshots.
func (s *Softmax) Axis() int {
	return s.axis
}

// Forward allocates softmax(input), subtracting each group's maximum for numerical
// stability. Negative infinity has probability zero when another logit is finite.
// NaNs, positive infinity, and all-negative-infinity groups produce NaNs.
// Invalid inputs, axes, empty normalization dimensions, and consumed graphs fail.
func (s *Softmax) Forward(input *nn.Value) (*nn.Value, error) {
	if s == nil || input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn/activation: softmax: uninitialized layer or input")
	}

	shape := input.Data().Shape()
	if s.axis < 0 || s.axis >= len(shape) || shape[s.axis] == 0 {
		return nil, fmt.Errorf("nn/activation: softmax: axis %d must select a nonempty dimension in %v", s.axis, shape)
	}

	values := input.Data().Data()
	probabilities := make([]float32, len(values))

	inner, outer, width := 1, 0, shape[s.axis]
	if len(values) > 0 {
		for _, dim := range shape[s.axis+1:] {
			inner *= dim
		}

		outer = len(values) / (width * inner)
	}

	for row := range outer {
		for column := range inner {
			base := row*width*inner + column

			maximum := math.Inf(-1)
			for j := range width {
				maximum = math.Max(maximum, float64(values[base+j*inner]))
			}

			sum := float64(0)
			for j := range width {
				sum += math.Exp(float64(values[base+j*inner]) - maximum)
			}

			for j := range width {
				probabilities[base+j*inner] = float32(math.Exp(float64(values[base+j*inner])-maximum) / sum)
			}
		}
	}

	data, err := tensor.From(probabilities, shape...)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		gradient := g.Data()

		for row := range outer {
			for column := range inner {
				base := row*width*inner + column
				dot := float64(0)

				for j := range width {
					index := base + j*inner
					dot += float64(gradient[index]) * float64(probabilities[index])
				}

				for j := range width {
					index := base + j*inner
					gradient[index] = probabilities[index] * (gradient[index] - float32(dot))
				}
			}
		}

		return tensor.From(gradient, shape...)
	})
}

// Parameters returns nil because Softmax has no trainable parameters.
func (*Softmax) Parameters() []*nn.Parameter { return nil }
