package data

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func sampleSize(shape []int, count int) (int, error) {
	n, err := (backend.View{Shape: shape}).Numel()
	if err != nil {
		return 0, fmt.Errorf("data: sample shape %v: %w", shape, err)
	}

	if n != count {
		return 0, fmt.Errorf("data: sample shape %v needs %d values, got %d", shape, n, count)
	}

	return n, nil
}

func batchValues(samples, elements int) ([]float32, error) {
	if elements > int(^uint(0)>>1)/4/samples {
		return nil, fmt.Errorf("data: batch byte count overflows address space")
	}

	return make([]float32, samples*elements), nil
}

// CollateClassification validates equal per-sample shapes and copies values and
// labels into an independent CPU batch. Empty input and malformed samples fail
// before tensor allocation. Output shape is [samples, sample shape...].
func CollateClassification(samples []ClassificationSample) (*ClassificationBatch, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("data: classification collate: empty samples")
	}

	shape := samples[0].Shape

	n, err := sampleSize(shape, len(samples[0].Input))
	if err != nil {
		return nil, err
	}

	for index, sample := range samples {
		if !slices.Equal(shape, sample.Shape) || len(sample.Input) != n || sample.Label < 0 {
			return nil, fmt.Errorf("data: classification sample %d: incompatible shape, length, or label", index)
		}
	}

	values, err := batchValues(len(samples), n)
	if err != nil {
		return nil, err
	}

	labels := make([]int, len(samples))
	for i, sample := range samples {
		copy(values[i*n:], sample.Input)
		labels[i] = sample.Label
	}

	inputs, err := tensor.From(values, append([]int{len(samples)}, shape...)...)
	if err != nil {
		return nil, err
	}

	return &ClassificationBatch{Inputs: inputs, Labels: labels}, nil
}

// CollateRegression copies fixed-shape inputs and targets into two independent
// CPU tensors. Both sample shapes are validated before allocation. On failure
// after the first allocation, its handle is released before returning.
func CollateRegression(samples []RegressionSample) (*RegressionBatch, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("data: regression collate: empty samples")
	}

	shape, targetShape := samples[0].Shape, samples[0].TargetShape

	n, err := sampleSize(shape, len(samples[0].Input))
	if err != nil {
		return nil, err
	}

	targetN, err := sampleSize(targetShape, len(samples[0].Target))
	if err != nil {
		return nil, err
	}

	for index, sample := range samples {
		if !slices.Equal(shape, sample.Shape) || len(sample.Input) != n ||
			!slices.Equal(targetShape, sample.TargetShape) || len(sample.Target) != targetN {
			return nil, fmt.Errorf("data: regression sample %d: incompatible input or target shape", index)
		}
	}

	values, err := batchValues(len(samples), n)
	if err != nil {
		return nil, err
	}

	targetValues, err := batchValues(len(samples), targetN)
	if err != nil {
		return nil, err
	}

	for i, sample := range samples {
		copy(values[i*n:], sample.Input)
		copy(targetValues[i*targetN:], sample.Target)
	}

	inputs, err := tensor.From(values, append([]int{len(samples)}, shape...)...)
	if err != nil {
		return nil, err
	}

	targets, err := tensor.From(targetValues, append([]int{len(samples)}, targetShape...)...)
	if err != nil {
		_ = inputs.Close()
		return nil, err
	}

	return &RegressionBatch{Inputs: inputs, Targets: targets}, nil
}
