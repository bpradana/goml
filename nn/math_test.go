package nn

import (
	"math"
	"testing"
)

func TestMath_ExpLogSqrtGradientsMatchIndependentDerivatives(t *testing.T) {
	for _, operation := range []struct {
		name       string
		apply      func(*Value) (*Value, error)
		derivative func(float64) float64
	}{
		{"exp", Exp, math.Exp},
		{"log", Log, func(x float64) float64 { return 1 / x }},
		{"sqrt", Sqrt, func(x float64) float64 { return 0.5 / math.Sqrt(x) }},
	} {
		t.Run(operation.name, func(t *testing.T) {
			input := testValue(t, []float32{0.5, 2}, []int{2}, true)

			output, err := operation.apply(input)
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

			for index, value := range input.Data().Data() {
				got, want := float64(input.Grad().Data()[index]), operation.derivative(float64(value))
				if math.Abs(got-want) > 1e-6 {
					t.Fatalf("element %d: got %g, want %g", index, got, want)
				}
			}

			if _, err := operation.apply(nil); err == nil {
				t.Fatal("nil input accepted")
			}
		})
	}
}
