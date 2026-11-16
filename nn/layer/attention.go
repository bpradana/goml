package layer

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/tensor"
)

// MultiHeadAttention performs scaled dot-product self-attention over batch-first
// sequences. Heads have equal width; causal mode excludes future positions.
type MultiHeadAttention struct {
	query    *Dense
	key      *Dense
	value    *Dense
	output   *Dense
	features int
	heads    int
	causal   bool
}

// NewMultiHeadAttention creates four trainable Xavier-initialized projections
// with zero biases. Features and heads must be positive, features must be divisible
// by heads, and rng must be nonnil. There is no dropout or padding mask.
func NewMultiHeadAttention(features, heads int, causal bool, rng *rand.Rand) (*MultiHeadAttention, error) {
	if features <= 0 || heads <= 0 || features%heads != 0 {
		return nil, fmt.Errorf("nn: attention: positive features divisible by positive heads required")
	}

	projections := make([]*Dense, 4)
	for i := range projections {
		projection, err := NewDense(features, features, true, rng)
		if err != nil {
			return nil, err
		}

		projections[i] = projection
	}

	return &MultiHeadAttention{
		query: projections[0], key: projections[1], value: projections[2], output: projections[3],
		features: features, heads: heads, causal: causal,
	}, nil
}

// Forward maps [batch,time,features] to the same shape. Batch and time must be
// positive. It computes softmax(QK^T/sqrt(headWidth))V independently per batch and
// head, then concatenates and projects the results. Attention storage grows with
// time squared. Invalid shapes, uninitialized layers, and consumed graphs fail.
func (a *MultiHeadAttention) Forward(input *nn.Value) (*nn.Value, error) {
	if a == nil || a.query == nil {
		return nil, fmt.Errorf("nn: attention: uninitialized layer")
	}

	shape, err := sequenceShape("attention", input, a.features)
	if err != nil {
		return nil, err
	}

	if shape[0] == 0 {
		return nil, fmt.Errorf("nn: attention: batch must be positive")
	}

	query, err := projectSequence(input, a.query)
	if err != nil {
		return nil, err
	}

	key, err := projectSequence(input, a.key)
	if err != nil {
		return nil, err
	}

	value, err := projectSequence(input, a.value)
	if err != nil {
		return nil, err
	}

	batches := make([]*nn.Value, shape[0])
	for batch := range shape[0] {
		heads := make([]*nn.Value, a.heads)
		for head := range a.heads {
			q, err := attentionHead(query, batch, head, a.heads)
			if err != nil {
				return nil, err
			}

			k, err := attentionHead(key, batch, head, a.heads)
			if err != nil {
				return nil, err
			}

			v, err := attentionHead(value, batch, head, a.heads)
			if err != nil {
				return nil, err
			}

			k, err = nn.Transpose(k)
			if err != nil {
				return nil, err
			}

			scores, err := nn.MatMul(q, k)
			if err != nil {
				return nil, err
			}

			scores, err = nn.Scale(scores, float32(1/math.Sqrt(float64(a.features/a.heads))))
			if err != nil {
				return nil, err
			}

			if a.causal {
				scores, err = maskFuture(scores)
				if err != nil {
					return nil, err
				}
			}

			probabilities, err := activation.NewSoftmax(1).Forward(scores)
			if err != nil {
				return nil, err
			}

			heads[head], err = nn.MatMul(probabilities, v)
			if err != nil {
				return nil, err
			}
		}

		joined, err := nn.Concat(1, heads...)
		if err != nil {
			return nil, err
		}

		batches[batch], err = nn.Reshape(joined, 1, shape[1], a.features)
		if err != nil {
			return nil, err
		}
	}

	joined, err := nn.Concat(0, batches...)
	if err != nil {
		return nil, err
	}

	return projectSequence(joined, a.output)
}

func attentionHead(input *nn.Value, batch, head, heads int) (*nn.Value, error) {
	shape := input.Data().Shape()
	width := shape[2] / heads

	value, err := nn.Slice(input, 0, batch, batch+1)
	if err != nil {
		return nil, err
	}

	value, err = nn.Slice(value, 2, head*width, (head+1)*width)
	if err != nil {
		return nil, err
	}

	value, err = nn.Contiguous(value)
	if err != nil {
		return nil, err
	}

	return nn.Reshape(value, shape[1], width)
}

func projectSequence(input *nn.Value, projection *Dense) (*nn.Value, error) {
	shape := input.Data().Shape()

	var err error
	if !input.Data().IsContiguous() {
		input, err = nn.Contiguous(input)
		if err != nil {
			return nil, err
		}
	}

	flat, err := nn.Reshape(input, shape[0]*shape[1], shape[2])
	if err != nil {
		return nil, err
	}

	output, err := projection.Forward(flat)
	if err != nil {
		return nil, err
	}

	return nn.Reshape(output, shape[0], shape[1], output.Data().Shape()[1])
}

func maskFuture(input *nn.Value) (*nn.Value, error) {
	if !input.Data().HostAccessible() {
		data, err := tensor.MaskUpperTriangle(input.Data(), float32(math.Inf(-1)))
		if err != nil {
			return nil, err
		}

		return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
			return tensor.MaskUpperTriangle(g, 0)
		})
	}

	shape := input.Data().Shape()

	values, err := input.Data().HostData()
	if err != nil {
		return nil, err
	}

	for row := range shape[0] {
		for column := row + 1; column < shape[1]; column++ {
			values[row*shape[1]+column] = float32(math.Inf(-1))
		}
	}

	data, err := tensor.FromLike(input.Data(), values, shape...)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		gradient, err := g.HostData()
		if err != nil {
			return nil, err
		}

		for row := range shape[0] {
			for column := row + 1; column < shape[1]; column++ {
				gradient[row*shape[1]+column] = 0
			}
		}

		return tensor.FromLike(input.Data(), gradient, shape...)
	})
}

// Parameters returns query, key, value, and output projection parameters, each
// consisting of weight followed by bias. The returned slice is independent.
func (a *MultiHeadAttention) Parameters() []*nn.Parameter {
	var parameters []*nn.Parameter
	for _, projection := range []*Dense{a.query, a.key, a.value, a.output} {
		parameters = append(parameters, projection.Parameters()...)
	}

	return parameters
}
