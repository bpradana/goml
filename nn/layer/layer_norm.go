package layer

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

// LayerNorm normalizes the final dimension, with trainable per-feature scale and
// bias. It uses population variance and has no running statistics.
type LayerNorm struct {
	scale    *nn.Parameter
	bias     *nn.Parameter
	features int
	epsilon  float64
}

// NewLayerNorm initializes scale to one and bias to zero. Features and epsilon
// must be positive, and epsilon must be finite. Invalid arguments return errors.
func NewLayerNorm(features int, epsilon float32) (*LayerNorm, error) {
	if features <= 0 || epsilon <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return nil, fmt.Errorf("nn: layer norm: positive features and finite positive epsilon required")
	}

	ones, err := tensor.Ones(features)
	if err != nil {
		return nil, err
	}

	zeros, err := tensor.Zeros(features)
	if err != nil {
		return nil, err
	}

	scale, err := nn.NewParameter(ones)
	if err != nil {
		return nil, err
	}

	bias, err := nn.NewParameter(zeros)
	if err != nil {
		return nil, err
	}

	return &LayerNorm{scale: scale, bias: bias, features: features, epsilon: float64(epsilon)}, nil
}

// Forward allocates scale*(x-mean)/sqrt(variance+epsilon)+bias, preserving shape.
// The input must have rank at least one and a matching final dimension. Mean,
// variance, and saved normalized values use float64 accumulation. Nonfinite
// inputs propagate NaNs. Inference retains no backward rule.
func (l *LayerNorm) Forward(input *nn.Value) (*nn.Value, error) {
	if l == nil || l.scale == nil || input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: layer norm: uninitialized layer or input")
	}

	shape := input.Data().Shape()
	if len(shape) == 0 || shape[len(shape)-1] != l.features {
		return nil, fmt.Errorf("nn: layer norm: expected final dimension %d, got %v", l.features, shape)
	}

	if input.Data().Device().ID() != l.scale.Value().Data().Device().ID() || input.Data().Device().ID() != l.bias.Value().Data().Device().ID() {
		return nil, fmt.Errorf("nn: layer norm: %w", backend.ErrDeviceMismatch)
	}

	if !input.Data().HostAccessible() {
		return l.deviceForward(input)
	}

	values, err := input.Data().HostData()
	if err != nil {
		return nil, err
	}

	scales, err := l.scale.Value().Data().HostData()
	if err != nil {
		return nil, err
	}

	biases, err := l.bias.Value().Data().HostData()
	if err != nil {
		return nil, err
	}

	normalized := make([]float64, len(values))
	inverse := make([]float64, len(values)/l.features)
	output := make([]float32, len(values))

	for row := range inverse {
		base := row * l.features

		mean := float64(0)
		for j := range l.features {
			mean += float64(values[base+j])
		}

		mean /= float64(l.features)

		variance := float64(0)

		for j := range l.features {
			difference := float64(values[base+j]) - mean
			variance += difference * difference
		}

		inverse[row] = 1 / math.Sqrt(variance/float64(l.features)+l.epsilon)

		for j := range l.features {
			normalized[base+j] = (float64(values[base+j]) - mean) * inverse[row]
			output[base+j] = float32(normalized[base+j]*float64(scales[j]) + float64(biases[j]))
		}
	}

	data, err := tensor.FromLike(input.Data(), output, shape...)
	if err != nil {
		return nil, err
	}

	return nn.NewMultiOperation(data, []*nn.Value{input, l.scale.Value(), l.bias.Value()}, func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		upstream, err := g.HostData()
		if err != nil {
			return nil, err
		}

		dx := make([]float32, len(upstream))
		dScale := make([]float32, l.features)

		dBias := make([]float32, l.features)
		for row, inv := range inverse {
			base := row * l.features
			sum, weighted := float64(0), float64(0)

			for j := range l.features {
				gradient := float64(upstream[base+j]) * float64(scales[j])
				sum += gradient
				weighted += gradient * normalized[base+j]
				dScale[j] += upstream[base+j] * float32(normalized[base+j])
				dBias[j] += upstream[base+j]
			}

			for j := range l.features {
				gradient := float64(upstream[base+j]) * float64(scales[j])
				dx[base+j] = float32(inv * (gradient - (sum+normalized[base+j]*weighted)/float64(l.features)))
			}
		}

		inputGradient, err := tensor.FromLike(input.Data(), dx, shape...)
		if err != nil {
			return nil, err
		}

		scaleGradient, err := tensor.FromLike(input.Data(), dScale, l.features)
		if err != nil {
			return nil, err
		}

		biasGradient, err := tensor.FromLike(input.Data(), dBias, l.features)
		if err != nil {
			return nil, err
		}

		return []*tensor.Tensor{inputGradient, scaleGradient, biasGradient}, nil
	})
}

// Parameters returns scale followed by bias in a fresh slice.
func (l *LayerNorm) Parameters() []*nn.Parameter {
	return []*nn.Parameter{l.scale, l.bias}
}
