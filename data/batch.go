package data

import (
	"errors"

	"github.com/bpradana/goml/tensor"
)

// ClassificationSample owns flattened row-major input values, their per-sample
// shape, and a nonnegative integer label. Scalars have an empty shape.
type ClassificationSample struct {
	Input []float32
	Shape []int
	Label int
}

// RegressionSample owns input and numerical target values with independent
// per-sample shapes. Targets may be scalars or fixed-shape arrays.
type RegressionSample struct {
	Input       []float32
	Shape       []int
	Target      []float32
	TargetShape []int
}

// ClassificationBatch owns one CPU input tensor and copied integer labels.
// Its lifetime is independent of the dataset and loader.
type ClassificationBatch struct {
	Inputs *tensor.Tensor
	Labels []int
}

// Close releases the batch input handle. Nil batches and repeated calls are safe.
func (b *ClassificationBatch) Close() error {
	if b == nil || b.Inputs == nil {
		return nil
	}

	return b.Inputs.Close()
}

// RegressionBatch owns independent CPU input and target tensors.
type RegressionBatch struct {
	Inputs  *tensor.Tensor
	Targets *tensor.Tensor
}

// Close attempts both releases even if one fails. Nil batches and repeated calls
// are safe. Closing never invalidates dataset storage or another batch.
func (b *RegressionBatch) Close() error {
	if b == nil {
		return nil
	}

	var inputErr, targetErr error
	if b.Inputs != nil {
		inputErr = b.Inputs.Close()
	}

	if b.Targets != nil {
		targetErr = b.Targets.Close()
	}

	return errors.Join(inputErr, targetErr)
}
