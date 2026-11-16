package activation

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestSiLU_GradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{-10, -2, -0.5, 0, 0.5, 2, 10}, NewSiLU().Forward)
}

func TestSiLU_ForwardPreservesShapeAndSpecialValues(t *testing.T) {
	input := testValue(t, []float32{-1000, 1000, float32(math.Inf(-1)), float32(math.Inf(1)), float32(math.NaN()), 0}, []int{2, 3}, false)

	output, err := NewSiLU().Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	x := output.Data().Data()
	if !slices.Equal(output.Data().Shape(), []int{2, 3}) || x[0] != 0 || x[1] != 1000 || x[2] != 0 || !math.Signbit(float64(x[2])) || !math.IsInf(float64(x[3]), 1) || !math.IsNaN(float64(x[4])) || x[5] != 0 {
		t.Fatal(output.Data())
	}
}

func TestSiLU_InferenceAndInvalidInput(t *testing.T) {
	input, err := nn.Inference(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	stage := NewSiLU()

	output, err := stage.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if output.RequiresGrad() || !output.IsInference() || len(stage.Parameters()) != 0 {
		t.Fatal("unexpected graph or parameters")
	}

	if _, err := stage.Forward(nil); err == nil {
		t.Fatal("expected invalid input error")
	}
}
