package activation

import (
	"math"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestSigmoid_GradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{-2, -0.5, 0.5, 2}, NewSigmoid().Forward)
}

func TestSigmoid_ExtremeInputsRemainFinite(t *testing.T) {
	input := testValue(t, []float32{-1000, 1000}, []int{2}, false)

	output, err := NewSigmoid().Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	values := output.Data().Data()
	if values[0] != 0 || values[1] != 1 || math.IsNaN(float64(values[0])) {
		t.Fatal(values)
	}
}

func TestSigmoid_InferenceDoesNotTrackGradients(t *testing.T) {
	input, err := nn.Inference(tensor.Scalar(0))
	if err != nil {
		t.Fatal(err)
	}

	stage := NewSigmoid()

	output, err := stage.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if output.RequiresGrad() || output.Grad() != nil || len(stage.Parameters()) != 0 {
		t.Fatal("inference recorded gradients or activation exposed parameters")
	}
}

func TestSigmoid_ForwardRejectsNilInput(t *testing.T) {
	if _, err := NewSigmoid().Forward(nil); err == nil {
		t.Fatal("expected invalid input error")
	}
}
