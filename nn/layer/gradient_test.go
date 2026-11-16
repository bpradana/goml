package layer

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func layerValue(t *testing.T, values []float32, shape []int, tracked bool) *nn.Value {
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

func setParameter(t *testing.T, parameter *nn.Parameter, values []float32) {
	t.Helper()

	data, err := tensor.From(values, parameter.Value().Data().Shape()...)
	if err != nil {
		t.Fatal(err)
	}

	if err := parameter.Update(data); err != nil {
		t.Fatal(err)
	}
}

func assertDerivative(t *testing.T, name string, index int, got float32, want float64) {
	t.Helper()

	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got)-want) > 0.003+0.02*math.Abs(want) {
		t.Fatalf("%s gradient %d: got %g, finite difference %g", name, index, got, want)
	}
}

// Check a weighted output so normalization and attention derivatives cannot pass
// merely because the sum of every normalized row is constant.
func checkLayerGradients(t *testing.T, stage Layer, values []float32, shape []int) {
	t.Helper()

	input := layerValue(t, values, shape, true)

	output, err := stage.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	weights := make([]float32, output.Data().Numel())
	for i := range weights {
		weights[i] = float32(i%7-3)*0.17 + 0.08
	}

	seed, err := tensor.From(weights, output.Data().Shape()...)
	if err != nil {
		t.Fatal(err)
	}

	if err := output.BackwardWith(seed); err != nil {
		t.Fatal(err)
	}

	evaluate := func(data []float32) float64 {
		array, err := tensor.From(data, shape...)
		if err != nil {
			t.Fatal(err)
		}

		inference, err := nn.Inference(array)
		if err != nil {
			t.Fatal(err)
		}

		result, err := stage.Forward(inference)
		if err != nil {
			t.Fatal(err)
		}

		if result.RequiresGrad() || !result.IsInference() {
			t.Fatal("inference constructed a tracked result")
		}

		sum := float64(0)
		for i, value := range backendtest.HostData(t, result.Data()) {
			sum += float64(value) * float64(weights[i])
		}

		return sum
	}

	const epsilon float32 = 1e-3

	for i, got := range backendtest.HostData(t, input.Grad()) {
		perturbed := slices.Clone(values)
		perturbed[i] += epsilon
		plus := evaluate(perturbed)
		perturbed[i] -= 2 * epsilon
		minus := evaluate(perturbed)
		assertDerivative(t, "input", i, got, (plus-minus)/float64(2*epsilon))
	}

	for index, parameter := range stage.Parameters() {
		if parameter.Value().Grad() == nil {
			t.Fatalf("parameter %d received no gradient", index)
		}

		original := backendtest.HostData(t, parameter.Value().Data())
		for i, got := range backendtest.HostData(t, parameter.Value().Grad()) {
			perturbed := slices.Clone(original)
			perturbed[i] += epsilon
			setParameter(t, parameter, perturbed)

			plus := evaluate(values)
			perturbed[i] -= 2 * epsilon
			setParameter(t, parameter, perturbed)

			minus := evaluate(values)

			setParameter(t, parameter, original)
			assertDerivative(t, fmt.Sprintf("parameter %d", index), i, got, (plus-minus)/float64(2*epsilon))
		}
	}
}
