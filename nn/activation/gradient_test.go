package activation

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func testValue(t *testing.T, values []float32, shape []int, tracked bool) *nn.Value {
	t.Helper()

	data, err := tensor.From(values, shape...)
	if err != nil {
		t.Fatal(err)
	}

	value, err := nn.NewValue(data, tracked)
	if err != nil {
		t.Fatal(err)
	}

	return value
}

func checkGradient(t *testing.T, values []float32, forward func(*nn.Value) (*nn.Value, error)) {
	t.Helper()

	input := testValue(t, values, []int{len(values)}, true)

	output, err := forward(input)
	if err != nil {
		t.Fatal(err)
	}

	objective, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	const epsilon float32 = 1e-3

	for i, got := range input.Grad().Data() {
		var objectives [2]float32

		for j, delta := range []float32{-epsilon, epsilon} {
			perturbed := slices.Clone(values)
			perturbed[i] += delta

			result, err := forward(testValue(t, perturbed, []int{len(values)}, false))
			if err != nil {
				t.Fatal(err)
			}

			total, err := tensor.Sum(result.Data())
			if err != nil {
				t.Fatal(err)
			}

			objectives[j] = total.Data()[0]
		}

		want := float64((objectives[1] - objectives[0]) / (2 * epsilon))
		if math.Abs(float64(got)-want) > 0.003+0.015*math.Abs(want) {
			t.Fatalf("gradient at %d: got %g, finite difference %g", i, got, want)
		}
	}
}
