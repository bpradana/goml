package loss

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestMSE_ForwardThroughInterfaceComputesMeanAndBothGradients(t *testing.T) {
	predictionData, err := tensor.From([]float32{1, 3}, 2)
	if err != nil {
		t.Fatal(err)
	}

	targetData, err := tensor.From([]float32{0, 1}, 2)
	if err != nil {
		t.Fatal(err)
	}

	prediction, err := nn.NewValue(predictionData, true)
	if err != nil {
		t.Fatal(err)
	}

	target, err := nn.NewValue(targetData, true)
	if err != nil {
		t.Fatal(err)
	}

	var criterion Loss[*nn.Value] = NewMSE()

	output, err := criterion.Forward(prediction, target)
	if err != nil {
		t.Fatal(err)
	}

	if output.Data().Rank() != 0 || backendtest.HostData(t, output.Data())[0] != 2.5 {
		t.Fatalf("got %v, want scalar 2.5", output.Data())
	}

	if err := output.Backward(); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, prediction.Grad()), []float32{1, 2}) {
		t.Fatalf("prediction gradient: %v", prediction.Grad())
	}

	if !slices.Equal(backendtest.HostData(t, target.Grad()), []float32{-1, -2}) {
		t.Fatalf("target gradient: %v", target.Grad())
	}
}

func TestMSE_ForwardRejectsNilInputs(t *testing.T) {
	var criterion Loss[*nn.Value] = NewMSE()

	if _, err := criterion.Forward(nil, nil); err == nil {
		t.Fatal("expected invalid value error")
	}
}

func TestMSE_ForwardPropagatesInferenceMode(t *testing.T) {
	prediction, err := nn.Inference(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	target, err := nn.NewValue(tensor.Scalar(1), true)
	if err != nil {
		t.Fatal(err)
	}

	output, err := NewMSE().Forward(prediction, target)
	if err != nil {
		t.Fatal(err)
	}

	if output.RequiresGrad() || output.Grad() != nil {
		t.Fatal("inference loss recorded gradients")
	}
}

func TestMSE_GradientMatchesFiniteDifferences(t *testing.T) {
	criterion := NewMSE()
	target := testValue(t, []float32{0.5, 1.5, 2.5}, []int{3}, false)

	checkLossGradient(t, []float32{1, 2, 3}, []int{3}, func(prediction *nn.Value) (*nn.Value, error) {
		return criterion.Forward(prediction, target)
	})
}

func TestMSE_ForwardRejectsShapeMismatch(t *testing.T) {
	prediction := testValue(t, []float32{1, 2}, []int{2}, true)
	target := testValue(t, []float32{1}, nil, false)

	if _, err := NewMSE().Forward(prediction, target); err == nil {
		t.Fatal("MSE must not broadcast targets")
	}
}

func TestMSE_ForwardRejectsEmptyInputs(t *testing.T) {
	empty := testValue(t, nil, []int{0}, true)

	if _, err := NewMSE().Forward(empty, empty); err == nil {
		t.Fatal("expected empty input error")
	}
}
