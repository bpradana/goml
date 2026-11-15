package layer

import (
	"fmt"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
)

// RNN is a single-layer, unidirectional tanh recurrent network. It has no hidden
// mutable state: Forward starts at zero and ForwardState accepts explicit state.
type RNN struct {
	inputProjection  *Dense
	hiddenProjection *Dense
	in               int
	hidden           int
}

// NewRNN creates h[t]=tanh(x[t]W+h[t-1]U+b) with positive feature sizes and a
// caller-owned seeded generator. Weights use Xavier normal initialization and
// bias starts at zero. Invalid dimensions or a nil generator return an error.
func NewRNN(in, hidden int, rng *rand.Rand) (*RNN, error) {
	inputProjection, err := NewDense(in, hidden, true, rng)
	if err != nil {
		return nil, fmt.Errorf("nn: rnn: %w", err)
	}

	hiddenProjection, err := NewDense(hidden, hidden, false, rng)
	if err != nil {
		return nil, fmt.Errorf("nn: rnn: %w", err)
	}

	return &RNN{inputProjection: inputProjection, hiddenProjection: hiddenProjection, in: in, hidden: hidden}, nil
}

// Forward maps batch-first [batch,time,in] to [batch,time,hidden], starting from
// zero state. Time must be positive. It records backpropagation through all steps
// unless the input is in inference mode. Outputs own independent tensor storage.
func (r *RNN) Forward(input *nn.Value) (*nn.Value, error) {
	output, _, err := r.ForwardState(input, nil)
	return output, err
}

// ForwardState additionally accepts and returns hidden state [batch,hidden]. A
// nil initial state means zero. State is never stored in the layer. Gradients
// flow through supplied tracked state; detach returned state to truncate them.
// Invalid shapes, uninitialized layers, and consumed graphs return errors.
func (r *RNN) ForwardState(input, initial *nn.Value) (*nn.Value, *nn.Value, error) {
	if r == nil || r.inputProjection == nil {
		return nil, nil, fmt.Errorf("nn: rnn: uninitialized layer")
	}

	shape, err := sequenceShape("rnn", input, r.in)
	if err != nil {
		return nil, nil, err
	}

	hidden, err := recurrentState(input, initial, shape[0], r.hidden)
	if err != nil {
		return nil, nil, err
	}

	steps := make([]*nn.Value, shape[1])
	for step := range shape[1] {
		x, err := sequenceStep(input, step)
		if err != nil {
			return nil, nil, err
		}

		projected, err := r.inputProjection.Forward(x)
		if err != nil {
			return nil, nil, err
		}

		recurrent, err := r.hiddenProjection.Forward(hidden)
		if err != nil {
			return nil, nil, err
		}

		hidden, err = nn.Add(projected, recurrent)
		if err != nil {
			return nil, nil, err
		}

		hidden, err = activation.NewTanh().Forward(hidden)
		if err != nil {
			return nil, nil, err
		}

		steps[step], err = nn.Reshape(hidden, shape[0], 1, r.hidden)
		if err != nil {
			return nil, nil, err
		}
	}

	output, err := nn.Concat(1, steps...)
	if err != nil {
		return nil, nil, err
	}

	return output, hidden, nil
}

// Parameters returns input weight, bias, and recurrent weight in that order.
func (r *RNN) Parameters() []*nn.Parameter {
	return append(r.inputProjection.Parameters(), r.hiddenProjection.Parameters()...)
}
