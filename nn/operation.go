package nn

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
)

// NewOperation wraps an already-computed immutable result as an operation with
// one differentiable input. It shares data without copying element storage.
// The result inherits gradient tracking and inference mode from input.
//
// When gradients are tracked, backward must compute the input gradient from the
// supplied output gradient. Its result must match the input's shape exactly.
// Capture immutable forward tensors in backward when the rule needs saved data.
//
// Nil or uninitialized data, invalid or consumed inputs, and a missing tracked
// backward rule return errors. Untracked operations do not retain backward.
// Backward rule errors and gradient shape errors are reported by Backward.
func NewOperation(
	data *tensor.Tensor,
	input *Value,
	backward func(*tensor.Tensor) (*tensor.Tensor, error),
) (*Value, error) {
	if err := checkValues("operation", input); err != nil {
		return nil, err
	}

	if data == nil || data.IsClosed() || (data.Rank() == 0 && data.Numel() != 1) {
		return nil, fmt.Errorf("nn: operation: nil or uninitialized result tensor")
	}

	if !sameDevice(data, input.data) {
		return nil, fmt.Errorf("nn: operation: result device must match input")
	}

	out := record(data, input)
	if !out.requires {
		return out, nil
	}

	if backward == nil {
		return nil, fmt.Errorf("nn: operation: tracked operation requires a backward rule")
	}

	shape, size := input.data.Shape(), input.data.Numel()

	out.backward = func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		gradient, err := backward(g)
		if err != nil {
			return nil, fmt.Errorf("nn: operation: backward: %w", err)
		}

		if gradient != nil && !sameDevice(gradient, input.data) {
			return nil, fmt.Errorf("nn: operation: gradient device must match input")
		}

		if gradient == nil || gradient.IsClosed() || gradient.Numel() != size || !slices.Equal(gradient.Shape(), shape) {
			return nil, fmt.Errorf("nn: operation: backward gradient must match input shape %v", shape)
		}

		return []*tensor.Tensor{gradient}, nil
	}

	return out, nil
}
