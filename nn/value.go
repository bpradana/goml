package nn

import (
	"fmt"
	"runtime"

	"github.com/bpradana/goml/tensor"
)

// Value holds an immutable tensor and optional differentiation state.
// Construct it with NewValue or Inference.
type Value struct {
	parameter       *parameterData
	savedParameters *parameterReads
	data            *tensor.Tensor
	grad            *tensor.Tensor
	ownsGrad        bool
	requires        bool
	inference       bool
	consumed        bool
	parents         []*Value
	backward        func(*tensor.Tensor) ([]*tensor.Tensor, error)
}

// NewValue wraps immutable data without copying it. requiresGrad controls
// tracking for this leaf. Nil and uninitialized tensors return an error.
func NewValue(data *tensor.Tensor, requiresGrad bool) (*Value, error) {
	if data == nil || data.IsClosed() || (data.Rank() == 0 && data.Numel() != 1) {
		return nil, fmt.Errorf("nn: value: nil or uninitialized tensor")
	}

	return &Value{data: data, requires: requiresGrad}, nil
}

// Inference wraps data and suppresses graph recording in every operation that
// uses the result, including operations with trainable parameters.
// Nil and uninitialized data return an error.
func Inference(data *tensor.Tensor) (*Value, error) {
	v, err := NewValue(data, false)
	if err != nil {
		return nil, err
	}

	v.inference = true

	return v, nil
}

// Data returns this value's immutable tensor. A parameter handle is borrowed
// until replacement. Use Tensor.Retain to keep an independent handle across
// updates; autograd retains the forward generations needed for backward.
func (v *Value) Data() *tensor.Tensor {
	return v.data
}

// Grad returns the accumulated immutable gradient, or nil before Backward.
func (v *Value) Grad() *tensor.Tensor {
	return v.grad
}

// RequiresGrad reports whether operations record dependencies for this value.
func (v *Value) RequiresGrad() bool {
	return v.requires
}

// IsInference reports whether this value suppresses graph recording. Layers use
// it to initialize explicit recurrent states without creating training graphs.
func (v *Value) IsInference() bool {
	return v.inference
}

// ZeroGrad clears this value's accumulated gradient. GPU leaf gradients own
// independent leases, which are released here; retain a view before clearing.
func (v *Value) ZeroGrad() {
	if v.ownsGrad {
		_ = v.grad.Close()
	}

	v.grad = nil
	v.ownsGrad = false
}

// Detach returns a constant leaf with an independent, unscoped tensor handle
// sharing immutable storage. It retains no graph or parameter references.
// Parameter replacement and closing the source handle do not invalidate it.
// The caller owns closing the returned value's Data after its last use, including
// backward passes that need it. Nil, uninitialized, or closed data return errors.
// Consumed values may be detached; trainable operands can still record a new graph.
func (v *Value) Detach() (*Value, error) {
	if v == nil {
		return nil, fmt.Errorf("nn: detach: nil value")
	}

	data, err := v.data.Retain()
	if err != nil {
		return nil, fmt.Errorf("nn: detach: %w", err)
	}

	return &Value{data: data}, nil
}

func checkValues(op string, values ...*Value) error {
	for _, v := range values {
		if v == nil || v.data == nil || v.data.IsClosed() {
			return fmt.Errorf("nn: %s: nil or uninitialized value", op)
		}

		if v.consumed {
			return fmt.Errorf("nn: %s: graph already consumed; detach or run forward again", op)
		}
	}

	return nil
}

func record(data *tensor.Tensor, first *Value, rest ...*Value) *Value {
	v := &Value{data: data, inference: first.inference, requires: first.requires}

	for _, p := range rest {
		v.inference = v.inference || p.inference
		v.requires = v.requires || p.requires
	}

	if v.inference {
		v.requires = false
	}

	if v.requires {
		v.parents = make([]*Value, len(rest)+1)
		v.parents[0] = first
		copy(v.parents[1:], rest)

		for _, parent := range v.parents {
			if parent.parameter != nil {
				if v.savedParameters == nil {
					v.savedParameters = &parameterReads{}
				}

				parent.parameter.acquire()
				v.savedParameters.values = append(v.savedParameters.values, parent.parameter)
			}
		}

		if v.savedParameters != nil {
			v.savedParameters.cleanup = runtime.AddCleanup(v, func(reads *parameterReads) { reads.release() }, v.savedParameters)
		}
	}

	return v
}

func (v *Value) releaseParameters() {
	if v.savedParameters != nil {
		v.savedParameters.release()
		v.savedParameters.cleanup.Stop()
		v.savedParameters = nil
	}
}
