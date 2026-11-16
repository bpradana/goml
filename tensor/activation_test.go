package tensor

import (
	"math"
	"testing"
)

func TestActivationBackward_CPUFiniteDifferencesAndExactShapes(t *testing.T) {
	input := fromTest(t, []float32{-1.2, -0.3, 0.2, 1.7}, 4)
	gradient := fromTest(t, []float32{1, 1, 1, 1}, 4)

	for _, kind := range []string{"relu", "sigmoid", "tanh", "silu"} {
		t.Run(kind, func(t *testing.T) {
			actual, err := ActivationBackward(kind, input, gradient)
			if err != nil {
				t.Fatal(err)
			}

			for index, value := range input.Data() {
				plus, err := Activation(kind, Scalar(value+0.001))
				if err != nil {
					t.Fatal(err)
				}

				minus, err := Activation(kind, Scalar(value-0.001))
				if err != nil {
					t.Fatal(err)
				}

				want := (plus.Data()[0] - minus.Data()[0]) / 0.002
				if math.Abs(float64(actual.Data()[index]-want)) > 0.0002 {
					t.Fatalf("element %d derivative: got %g, want %g", index, actual.Data()[index], want)
				}
			}
		})
	}

	if _, err := ActivationBackward("relu", input, Scalar(1)); err == nil {
		t.Fatal("broadcast gradient accepted")
	}

	if _, err := Activation("unknown", input); err == nil {
		t.Fatal("unknown activation accepted")
	}

	result, err := ActivationBackward("relu", Scalar(float32(math.NaN())), Scalar(1))
	if err != nil || !math.IsNaN(float64(result.Data()[0])) {
		t.Fatal("ReLU NaN derivative did not propagate", err)
	}
}
