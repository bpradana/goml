package layer

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func sequenceShape(name string, input *nn.Value, features int) ([]int, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: %s: uninitialized input", name)
	}

	shape := input.Data().Shape()
	if len(shape) != 3 || shape[1] <= 0 || shape[2] != features {
		return nil, fmt.Errorf("nn: %s: expected [batch,time>0,%d], got %v", name, features, shape)
	}

	return shape, nil
}

func recurrentState(input, state *nn.Value, batch, hidden int) (*nn.Value, error) {
	if state != nil {
		if state.Data() == nil || !slices.Equal(state.Data().Shape(), []int{batch, hidden}) {
			return nil, fmt.Errorf("nn: recurrent state: expected [%d %d]", batch, hidden)
		}

		if input.IsInference() {
			return nn.Inference(state.Data())
		}

		return state, nil
	}

	data, err := tensor.FullLike(input.Data(), 0, batch, hidden)
	if err != nil {
		return nil, err
	}

	if input.IsInference() {
		return nn.Inference(data)
	}

	return nn.NewValue(data, false)
}

func sequenceStep(input *nn.Value, step int) (*nn.Value, error) {
	shape := input.Data().Shape()

	value, err := nn.Slice(input, 1, step, step+1)
	if err != nil {
		return nil, err
	}

	if !value.Data().IsContiguous() {
		value, err = nn.Contiguous(value)
		if err != nil {
			return nil, err
		}
	}

	return nn.Reshape(value, shape[0], shape[2])
}
