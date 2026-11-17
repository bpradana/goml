package data

import (
	"context"
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
)

// ClassificationDataset stores a host-owned snapshot of input values and labels.
// It does not retain source tensor handles and needs no Close method.
type ClassificationDataset struct {
	values     []float32
	shape      []int
	labels     []int
	sampleSize int
}

// NewClassificationDataset copies a host-accessible tensor once. A leading
// sample dimension must match labels. GPU sources are rejected; explicitly call
// ToCPU first. Subsequent source closure cannot invalidate this dataset.
func NewClassificationDataset(inputs *tensor.Tensor, labels []int) (*ClassificationDataset, error) {
	values, shape, size, err := snapshot(inputs)
	if err != nil {
		return nil, err
	}

	if shape[0] != len(labels) {
		return nil, fmt.Errorf("data: classification: input count %d differs from labels %d", shape[0], len(labels))
	}

	for _, label := range labels {
		if label < 0 {
			return nil, fmt.Errorf("data: classification: labels must be nonnegative")
		}
	}

	return &ClassificationDataset{values: values, shape: shape[1:], labels: slices.Clone(labels), sampleSize: size}, nil
}

func snapshot(input *tensor.Tensor) ([]float32, []int, int, error) {
	if input == nil || input.IsClosed() || !input.HostAccessible() || input.Rank() == 0 {
		return nil, nil, 0, fmt.Errorf("data: dataset requires a live host-accessible tensor with a sample dimension; explicitly transfer accelerators to CPU")
	}

	shape := input.Shape()

	size := 1
	for _, dimension := range shape[1:] {
		if dimension != 0 && size > int(^uint(0)>>1)/dimension {
			return nil, nil, 0, fmt.Errorf("data: sample size overflows address space")
		}

		size *= dimension
	}

	values, err := input.HostData()

	return values, shape, size, err
}

// Len returns the stable sample count.
func (d *ClassificationDataset) Len() int {
	return len(d.labels)
}

// Get copies one sample. Invalid indices and canceled contexts return errors.
func (d *ClassificationDataset) Get(ctx context.Context, index int) (ClassificationSample, error) {
	if err := ctx.Err(); err != nil {
		return ClassificationSample{}, err
	}

	if index < 0 || index >= d.Len() {
		return ClassificationSample{}, fmt.Errorf("data: classification index %d out of range", index)
	}

	return ClassificationSample{Input: slices.Clone(d.values[index*d.sampleSize : (index+1)*d.sampleSize]), Shape: slices.Clone(d.shape), Label: d.labels[index]}, nil
}

// RegressionDataset stores independent input and target snapshots in host memory.
type RegressionDataset struct {
	inputs, targets              []float32
	shape, targetShape           []int
	count, inputSize, targetSize int
}

// NewRegressionDataset copies host-accessible tensors once. Both must have the
// same leading sample dimension. Sources may be noncontiguous and can be closed
// immediately after construction. GPU reads require an explicit ToCPU first.
func NewRegressionDataset(inputs, targets *tensor.Tensor) (*RegressionDataset, error) {
	values, shape, size, err := snapshot(inputs)
	if err != nil {
		return nil, err
	}

	targetValues, targetShape, targetSize, err := snapshot(targets)
	if err != nil {
		return nil, err
	}

	if shape[0] != targetShape[0] {
		return nil, fmt.Errorf("data: regression: input and target sample counts differ")
	}

	return &RegressionDataset{inputs: values, targets: targetValues, shape: shape[1:], targetShape: targetShape[1:], count: shape[0], inputSize: size, targetSize: targetSize}, nil
}

// Len returns the stable sample count.
func (d *RegressionDataset) Len() int {
	return d.count
}

// Get returns independent copies of one input and target sample.
func (d *RegressionDataset) Get(ctx context.Context, index int) (RegressionSample, error) {
	if err := ctx.Err(); err != nil {
		return RegressionSample{}, err
	}

	if index < 0 || index >= d.Len() {
		return RegressionSample{}, fmt.Errorf("data: regression index %d out of range", index)
	}

	return RegressionSample{Input: slices.Clone(d.inputs[index*d.inputSize : (index+1)*d.inputSize]), Shape: slices.Clone(d.shape), Target: slices.Clone(d.targets[index*d.targetSize : (index+1)*d.targetSize]), TargetShape: slices.Clone(d.targetShape)}, nil
}
