package nn

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
)

// Parameter owns a trainable leaf and its gradient. Optimizer updates preserve
// the leaf's identity and replace its immutable tensor data.
type Parameter struct {
	value *Value
}

// NewParameter wraps immutable data in a trainable leaf.
// Nil or uninitialized tensors return an error.
func NewParameter(data *tensor.Tensor) (*Parameter, error) {
	if data != nil && data.Device() != nil {
		owned, err := data.To(data.Device())
		if err != nil {
			return nil, err
		}

		data = owned
	}

	v, err := NewValue(data, true)
	if err != nil {
		return nil, err
	}

	return &Parameter{value: v}, nil
}

// Value returns the trainable leaf used in differentiable computations.
func (p *Parameter) Value() *Value {
	return p.value
}

// ZeroGrad clears this parameter's accumulated gradient.
func (p *Parameter) ZeroGrad() {
	p.value.ZeroGrad()
}

// Update replaces the immutable tensor data while preserving the parameter's
// identity and accumulated gradient. The new shape must match the old shape.
// Optimizers use this boundary; callers must synchronize updates with training.
// GPU parameters own a lease separate from the supplied tensor. Replacement
// releases the old borrowed handle; independent GPU views retain their values.
func (p *Parameter) Update(data *tensor.Tensor) error {
	if p == nil || p.value == nil {
		return fmt.Errorf("nn: parameter: uninitialized parameter")
	}

	if data == nil || data.IsClosed() || !slices.Equal(data.Shape(), p.value.data.Shape()) ||
		data.Numel() != p.value.data.Numel() {
		return fmt.Errorf("nn: parameter: update must preserve shape %v", p.value.data.Shape())
	}

	if !sameDevice(data, p.value.data) {
		return fmt.Errorf("nn: parameter: update must preserve device; use nn.ToDevice")
	}

	if data.Device() != nil {
		owned, err := data.To(data.Device())
		if err != nil {
			return err
		}

		previous := p.value.data
		p.value.data = owned
		_ = previous.Close()
	} else {
		p.value.data = data
	}

	return nil
}
