package activation

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestSoftmax_LargeLogitsAndMaskedNegativeInfinity(t *testing.T) {
	input := testValue(t, []float32{10000, 10001, float32(math.Inf(-1))}, []int{1, 3}, true)

	output, err := NewSoftmax(1).Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	want := []float64{1 / (1 + math.E), math.E / (1 + math.E), 0}
	for i, got := range output.Data().Data() {
		if math.Abs(float64(got)-want[i]) > 1e-6 {
			t.Fatalf("probability %d: got %g, want %g", i, got, want[i])
		}
	}
}

func TestSoftmax_ArbitraryAxisWeightedGradientsMatchFiniteDifferences(t *testing.T) {
	for _, tt := range []struct {
		name string
		axis int
	}{{"Rows", 1}, {"Columns", 0}} {
		t.Run(tt.name, func(t *testing.T) {
			values := []float32{0.3, -0.2, 0.8, 0.1, 0.4, -0.5}
			input := testValue(t, values, []int{2, 3}, true)
			softmax := NewSoftmax(tt.axis)

			output, err := softmax.Forward(input)
			if err != nil {
				t.Fatal(err)
			}

			weights := []float32{1, -2, 3, -4, 5, -6}

			seed, err := tensor.From(weights, 2, 3)
			if err != nil {
				t.Fatal(err)
			}

			if err := output.BackwardWith(seed); err != nil {
				t.Fatal(err)
			}

			for index, got := range input.Grad().Data() {
				objectives := [2]float64{}

				for side, delta := range []float32{-1e-3, 1e-3} {
					perturbed := slices.Clone(values)
					perturbed[index] += delta

					result, err := softmax.Forward(testValue(t, perturbed, []int{2, 3}, false))
					if err != nil {
						t.Fatal(err)
					}

					for i, value := range result.Data().Data() {
						objectives[side] += float64(value) * float64(weights[i])
					}
				}

				want := (objectives[1] - objectives[0]) / 0.002
				if math.Abs(float64(got)-want) > 0.002 {
					t.Fatalf("gradient %d: got %g, finite difference %g", index, got, want)
				}
			}
		})
	}
}

func TestSoftmax_EmptyBatchAndInvalidInputs(t *testing.T) {
	output, err := NewSoftmax(1).Forward(testValue(t, nil, []int{0, 3}, false))
	if err != nil || output.Data().Numel() != 0 {
		t.Fatalf("empty batch: %v %v", output, err)
	}

	for _, tt := range []struct {
		axis  int
		input *nn.Value
	}{{0, nil}, {-1, testValue(t, []float32{1}, []int{1}, false)}, {1, testValue(t, []float32{1}, []int{1}, false)}, {0, testValue(t, nil, []int{0}, false)}} {
		if _, err := NewSoftmax(tt.axis).Forward(tt.input); err == nil {
			t.Fatal("accepted invalid softmax input")
		}
	}

	if len(NewSoftmax(0).Parameters()) != 0 {
		t.Fatal("softmax exposed parameters")
	}
}
