package data_test

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
)

func TestCollateClassification_ShapeOrderAndIndependentOwnership(t *testing.T) {
	samples := []data.ClassificationSample{{Input: []float32{1, 2}, Shape: []int{1, 2}, Label: 3}, {Input: []float32{4, 5}, Shape: []int{1, 2}, Label: 7}}

	first, err := data.CollateClassification(samples)
	if err != nil {
		t.Fatal(err)
	}

	second, err := data.CollateClassification(samples)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = second.Close() }()

	samples[0].Input[0] = 99
	first.Labels[0] = 99
	_ = first.Close()

	values, err := second.Inputs.HostData()
	if err != nil || !slices.Equal(values, []float32{1, 2, 4, 5}) || second.Labels[0] != 3 || !slices.Equal(second.Inputs.Shape(), []int{2, 1, 2}) {
		t.Fatalf("bad batch: %v %v", values, err)
	}
}

func TestCollate_RejectsEmptyAndIncompatibleSamples(t *testing.T) {
	for _, samples := range [][]data.ClassificationSample{
		nil,
		{{Input: []float32{1}, Shape: []int{2}}},
		{{Shape: []int{-1}}},
		{{Input: []float32{1}, Shape: []int{1}, Label: -1}},
		{{Input: []float32{1}, Shape: []int{1}}, {Input: []float32{2}, Shape: []int{1, 1}}},
	} {
		if batch, err := data.CollateClassification(samples); err == nil || batch != nil {
			t.Fatal("accepted invalid samples")
		}
	}

	if batch, err := data.CollateRegression([]data.RegressionSample{{Input: []float32{1}, Target: nil}}); err == nil || batch != nil {
		t.Fatal("accepted invalid scalar target")
	}
}

func TestCollateRegression_ScalarTargetsAndEmptySampleDimensions(t *testing.T) {
	batch, err := data.CollateRegression([]data.RegressionSample{{Shape: []int{0}, Target: []float32{2}}, {Shape: []int{0}, Target: []float32{3}}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = batch.Close() }()

	values, _ := batch.Targets.HostData()
	if !slices.Equal(values, []float32{2, 3}) || !slices.Equal(batch.Inputs.Shape(), []int{2, 0}) || !slices.Equal(batch.Targets.Shape(), []int{2}) {
		t.Fatal("bad regression shapes")
	}
}

func BenchmarkCollateClassification(b *testing.B) {
	samples := make([]data.ClassificationSample, 64)
	for i := range samples {
		samples[i] = data.ClassificationSample{Input: make([]float32, 784), Shape: []int{1, 28, 28}, Label: i % 10}
	}

	b.ReportAllocs()

	for b.Loop() {
		batch, err := data.CollateClassification(samples)
		if err != nil {
			b.Fatal(err)
		}

		_ = batch.Close()
	}
}
