package transform_test

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/data/transform"
)

func TestNormalize_ChannelStatisticsAreCopiedAndInputIsUnchanged(t *testing.T) {
	mean, std := []float32{1, 2, 3}, []float32{2, 4, 8}

	apply, err := transform.Normalize(mean, std)
	if err != nil {
		t.Fatal(err)
	}

	mean[0], std[0] = 99, 99
	sample := data.ClassificationSample{Input: []float32{3, 6, 11}, Shape: []int{3, 1, 1}, Label: 7}

	got, err := apply(context.Background(), sample)
	if err != nil || !slices.Equal(got.Input, []float32{1, 1, 1}) || got.Label != 7 {
		t.Fatal(got, err)
	}

	got.Input[0] = 44

	got.Shape[0] = 1
	if sample.Input[0] != 3 || sample.Shape[0] != 3 {
		t.Fatal("transform mutated source")
	}
}

func TestNormalize_RejectsInvalidStatisticsAndImageShape(t *testing.T) {
	for _, std := range [][]float32{nil, {0}, {-1}, {float32(math.Inf(1))}, {float32(math.NaN())}} {
		if _, err := transform.Normalize([]float32{0}, std); err == nil {
			t.Fatal("accepted statistics", std)
		}
	}

	apply, _ := transform.Normalize([]float32{0}, []float32{1})
	if _, err := apply(context.Background(), data.ClassificationSample{Shape: []int{1, 2, 2}, Input: []float32{1}}); err == nil {
		t.Fatal("accepted malformed image")
	}
}
