package nn

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor"
)

func testValue(t *testing.T, data []float32, shape []int, tracked bool) *Value {
	t.Helper()

	array, err := tensor.From(data, shape...)
	if err != nil {
		t.Fatal(err)
	}

	value, err := NewValue(array, tracked)
	if err != nil {
		t.Fatal(err)
	}

	return value
}

func checkGradient(t *testing.T, data []float32, shape []int, forward func(*Value) (*Value, error)) {
	t.Helper()

	input := testValue(t, data, shape, true)

	output, err := forward(input)
	if err != nil {
		t.Fatal(err)
	}

	objective, err := Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	got := input.Grad().Data()

	const epsilon float32 = 1e-3

	for i := range data {
		objectives := [2]float32{}

		for j, delta := range []float32{-epsilon, epsilon} {
			perturbed := slices.Clone(data)
			perturbed[i] += delta

			result, err := forward(testValue(t, perturbed, shape, false))
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
		if math.Abs(float64(got[i])-want) > 0.003+0.015*math.Abs(want) {
			t.Fatalf("gradient at %d: got %g, finite difference %g", i, got[i], want)
		}
	}
}

func TestNewValue_RejectsInvalidTensors(t *testing.T) {
	for _, data := range []*tensor.Tensor{nil, {}} {
		if _, err := NewValue(data, true); err == nil {
			t.Fatal("expected invalid tensor error")
		}
	}
}

func TestValue_IsInferenceDistinguishesConstantAndPredictionInputs(t *testing.T) {
	constant := testValue(t, []float32{1}, nil, false)
	if constant.IsInference() {
		t.Fatal("ordinary constant suppressed future parameter tracking")
	}

	prediction, err := Inference(constant.Data())
	if err != nil {
		t.Fatal(err)
	}

	if !prediction.IsInference() || prediction.RequiresGrad() {
		t.Fatal("inference mode was not exposed correctly")
	}
}

func TestValue_DetachDropsGraphAndSharesImmutableData(t *testing.T) {
	input := testValue(t, []float32{2}, nil, true)

	output, err := Mul(input, input)
	if err != nil {
		t.Fatal(err)
	}

	detached := output.Detach()
	if detached.RequiresGrad() || detached.parents != nil || detached.backward != nil || detached.Data() != output.Data() {
		t.Fatal("detach retained a graph or copied immutable storage")
	}
}

func TestInference_SuppressesParameterTracking(t *testing.T) {
	input, err := Inference(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	weight := testValue(t, []float32{3}, nil, true)

	output, err := Mul(input, weight)
	if err != nil {
		t.Fatal(err)
	}

	if output.RequiresGrad() || output.parents != nil || output.backward != nil {
		t.Fatal("inference constructed a graph")
	}

	if err := output.Backward(); err == nil {
		t.Fatal("inference backward must fail")
	}
}
