package loss

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

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

func checkLossGradient(
	t *testing.T,
	data []float32,
	shape []int,
	forward func(*nn.Value) (*nn.Value, error),
) {
	t.Helper()

	input := testValue(t, data, shape, true)

	output, err := forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if output.Data().Rank() != 0 {
		t.Fatal("loss did not produce a scalar")
	}

	if err := output.Backward(); err != nil {
		t.Fatal(err)
	}

	gradient := backendtest.HostData(t, input.Grad())

	const epsilon float32 = 1e-3

	for i, got := range gradient {
		var objectives [2]float32

		for j, delta := range []float32{-epsilon, epsilon} {
			perturbed := slices.Clone(data)
			perturbed[i] += delta

			result, err := forward(testValue(t, perturbed, shape, false))
			if err != nil {
				t.Fatal(err)
			}

			objectives[j] = backendtest.HostData(t, result.Data())[0]
		}

		want := float64((objectives[1] - objectives[0]) / (2 * epsilon))
		if math.Abs(float64(got)-want) > 0.003+0.015*math.Abs(want) {
			t.Fatalf("gradient at %d: got %g, finite difference %g", i, got, want)
		}
	}
}
