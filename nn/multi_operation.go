package nn

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
)

// NewMultiOperation registers an immutable result with multiple differentiable
// inputs. It shares data and copies the input list. Inference on any input
// disables recording. At least one valid, unconsumed input is required.
//
// A tracked operation requires a backward rule returning one gradient per input,
// in input order, each with that input's shape. Repeated inputs accumulate their
// contributions. Save immutable forward tensors rather than mutable parameters.
// Rule errors and invalid gradients fail Backward without committing gradients.
func NewMultiOperation(data *tensor.Tensor, inputs []*Value, backward func(*tensor.Tensor) ([]*tensor.Tensor, error)) (*Value, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("nn: multi operation: at least one input required")
	}

	if err := checkValues("multi operation", inputs...); err != nil {
		return nil, err
	}

	if data == nil || data.IsClosed() || (data.Rank() == 0 && data.Numel() != 1) {
		return nil, fmt.Errorf("nn: multi operation: nil or uninitialized result")
	}

	for _, input := range inputs {
		if !sameDevice(data, input.data) {
			return nil, fmt.Errorf("nn: multi operation: result and input devices must match")
		}
	}

	out := record(data, inputs[0], inputs[1:]...)
	if !out.requires {
		return out, nil
	}

	if backward == nil {
		return nil, fmt.Errorf("nn: multi operation: tracked operation requires a backward rule")
	}

	inputs = out.parents

	shapes := make([][]int, len(inputs))

	sizes := make([]int, len(inputs))
	for i, input := range inputs {
		shapes[i] = input.data.Shape()
		sizes[i] = input.data.Numel()
	}

	out.backward = func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		grads, err := backward(g)
		if err != nil {
			return nil, fmt.Errorf("nn: multi operation: backward: %w", err)
		}

		if len(grads) != len(shapes) {
			return nil, fmt.Errorf("nn: multi operation: expected %d gradients, got %d", len(shapes), len(grads))
		}

		for i, grad := range grads {
			if grad != nil && !sameDevice(grad, inputs[i].data) {
				return nil, fmt.Errorf("nn: multi operation: gradient device must match input")
			}

			if grad == nil || grad.IsClosed() || grad.Numel() != sizes[i] || !slices.Equal(grad.Shape(), shapes[i]) {
				return nil, fmt.Errorf("nn: multi operation: gradient %d must match shape %v", i, shapes[i])
			}
		}

		return grads, nil
	}

	return out, nil
}
