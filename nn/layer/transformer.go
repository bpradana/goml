package layer

import (
	"fmt"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
)

// TransformerEncoder is a pre-normalized encoder block with multi-head
// self-attention, residual connections, and a two-projection ReLU feed-forward
// network. It does not include embeddings, positional encoding, or dropout.
type TransformerEncoder struct {
	attention       *MultiHeadAttention
	normAttention   *LayerNorm
	normFeedForward *LayerNorm
	feedForwardIn   *Dense
	feedForwardOut  *Dense
}

// NewTransformerEncoder constructs a bidirectional encoder block. Features must
// be divisible by heads; features, heads, and feedForward must be positive. The
// caller supplies a seeded generator. Normalization uses epsilon 1e-5.
func NewTransformerEncoder(features, heads, feedForward int, rng *rand.Rand) (*TransformerEncoder, error) {
	attention, err := NewMultiHeadAttention(features, heads, false, rng)
	if err != nil {
		return nil, err
	}

	normAttention, err := NewLayerNorm(features, 1e-5)
	if err != nil {
		return nil, err
	}

	normFeedForward, err := NewLayerNorm(features, 1e-5)
	if err != nil {
		return nil, err
	}

	feedForwardIn, err := NewDense(features, feedForward, true, rng)
	if err != nil {
		return nil, err
	}

	feedForwardOut, err := NewDense(feedForward, features, true, rng)
	if err != nil {
		return nil, err
	}

	return &TransformerEncoder{
		attention: attention, normAttention: normAttention, normFeedForward: normFeedForward,
		feedForwardIn: feedForwardIn, feedForwardOut: feedForwardOut,
	}, nil
}

// Forward allocates an encoded [batch,time,features] sequence of the same shape.
// Batch and time must be positive. It records gradients through both residual
// paths during training and propagates inference through all internal layers.
// Invalid inputs, uninitialized layers, and consumed graphs return errors.
func (t *TransformerEncoder) Forward(input *nn.Value) (*nn.Value, error) {
	if t == nil || t.attention == nil {
		return nil, fmt.Errorf("nn: transformer: uninitialized layer")
	}

	normalized, err := t.normAttention.Forward(input)
	if err != nil {
		return nil, err
	}

	attended, err := t.attention.Forward(normalized)
	if err != nil {
		return nil, err
	}

	residual, err := nn.Add(input, attended)
	if err != nil {
		return nil, err
	}

	normalized, err = t.normFeedForward.Forward(residual)
	if err != nil {
		return nil, err
	}

	projected, err := projectSequence(normalized, t.feedForwardIn)
	if err != nil {
		return nil, err
	}

	activated, err := activation.NewReLU().Forward(projected)
	if err != nil {
		return nil, err
	}

	projected, err = projectSequence(activated, t.feedForwardOut)
	if err != nil {
		return nil, err
	}

	return nn.Add(residual, projected)
}

// Parameters returns each sublayer's parameters in forward order, once each.
func (t *TransformerEncoder) Parameters() []*nn.Parameter {
	parameters := t.normAttention.Parameters()
	parameters = append(parameters, t.attention.Parameters()...)
	parameters = append(parameters, t.normFeedForward.Parameters()...)
	parameters = append(parameters, t.feedForwardIn.Parameters()...)

	return append(parameters, t.feedForwardOut.Parameters()...)
}
