package layer

import (
	"fmt"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/tensor"
)

// LSTM is a single-layer, unidirectional long short-term memory network with
// explicit hidden and cell state. Gate order is input, forget, candidate, output.
type LSTM struct {
	inputProjection  *Dense
	hiddenProjection *Dense
	in               int
	hidden           int
}

// NewLSTM constructs an LSTM with positive feature sizes and a seeded generator.
// Weights use Xavier normal initialization. Forget bias starts at one; other
// biases start at zero. Invalid dimensions, overflow, and nil generators fail.
func NewLSTM(in, hidden int, rng *rand.Rand) (*LSTM, error) {
	if hidden <= 0 || hidden > int(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("nn: lstm: invalid hidden size %d", hidden)
	}

	inputProjection, err := NewDense(in, 4*hidden, true, rng)
	if err != nil {
		return nil, fmt.Errorf("nn: lstm: %w", err)
	}

	hiddenProjection, err := NewDense(hidden, 4*hidden, false, rng)
	if err != nil {
		return nil, fmt.Errorf("nn: lstm: %w", err)
	}

	bias := make([]float32, 4*hidden)
	for i := hidden; i < 2*hidden; i++ {
		bias[i] = 1
	}

	data, err := tensor.From(bias, 4*hidden)
	if err != nil {
		return nil, err
	}

	if err := inputProjection.bias.Update(data); err != nil {
		return nil, err
	}

	return &LSTM{inputProjection: inputProjection, hiddenProjection: hiddenProjection, in: in, hidden: hidden}, nil
}

// Forward maps [batch,time,in] to [batch,time,hidden], starting hidden and cell
// state at zero. Time must be positive. Backward differentiates all time steps;
// inference suppresses graph recording. The layer retains no state between calls.
func (l *LSTM) Forward(input *nn.Value) (*nn.Value, error) {
	output, _, _, err := l.ForwardState(input, nil, nil)
	return output, err
}

// ForwardState accepts optional hidden and cell states [batch,hidden] and returns
// the output sequence and final states. Nil states mean zero. State ownership
// remains with the caller; detach returned states to truncate backpropagation.
// Invalid shapes, uninitialized layers, and consumed graphs return errors.
func (l *LSTM) ForwardState(input, initialHidden, initialCell *nn.Value) (*nn.Value, *nn.Value, *nn.Value, error) {
	if l == nil || l.inputProjection == nil {
		return nil, nil, nil, fmt.Errorf("nn: lstm: uninitialized layer")
	}

	shape, err := sequenceShape("lstm", input, l.in)
	if err != nil {
		return nil, nil, nil, err
	}

	hidden, err := recurrentState(input, initialHidden, shape[0], l.hidden)
	if err != nil {
		return nil, nil, nil, err
	}

	cell, err := recurrentState(input, initialCell, shape[0], l.hidden)
	if err != nil {
		return nil, nil, nil, err
	}

	steps := make([]*nn.Value, shape[1])
	for step := range shape[1] {
		x, err := sequenceStep(input, step)
		if err != nil {
			return nil, nil, nil, err
		}

		projected, err := l.inputProjection.Forward(x)
		if err != nil {
			return nil, nil, nil, err
		}

		recurrent, err := l.hiddenProjection.Forward(hidden)
		if err != nil {
			return nil, nil, nil, err
		}

		gates, err := nn.Add(projected, recurrent)
		if err != nil {
			return nil, nil, nil, err
		}

		parts := make([]*nn.Value, 4)

		for gate := range 4 {
			part, err := nn.Slice(gates, 1, gate*l.hidden, (gate+1)*l.hidden)
			if err != nil {
				return nil, nil, nil, err
			}

			if gate == 2 {
				parts[gate], err = activation.NewTanh().Forward(part)
			} else {
				parts[gate], err = activation.NewSigmoid().Forward(part)
			}

			if err != nil {
				return nil, nil, nil, err
			}
		}

		retained, err := nn.Mul(parts[1], cell)
		if err != nil {
			return nil, nil, nil, err
		}

		added, err := nn.Mul(parts[0], parts[2])
		if err != nil {
			return nil, nil, nil, err
		}

		cell, err = nn.Add(retained, added)
		if err != nil {
			return nil, nil, nil, err
		}

		activated, err := activation.NewTanh().Forward(cell)
		if err != nil {
			return nil, nil, nil, err
		}

		hidden, err = nn.Mul(parts[3], activated)
		if err != nil {
			return nil, nil, nil, err
		}

		steps[step], err = nn.Reshape(hidden, shape[0], 1, l.hidden)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	output, err := nn.Concat(1, steps...)
	if err != nil {
		return nil, nil, nil, err
	}

	return output, hidden, cell, nil
}

// Parameters returns packed input weight, bias, and recurrent weight, in gate
// order input, forget, candidate, output along the final weight dimension.
func (l *LSTM) Parameters() []*nn.Parameter {
	return append(l.inputProjection.Parameters(), l.hiddenProjection.Parameters()...)
}
