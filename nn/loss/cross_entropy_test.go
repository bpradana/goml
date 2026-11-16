package loss

import (
	"math"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestCrossEntropy_ForwardThroughInterfacePreservesStableLoss(t *testing.T) {
	data, err := tensor.From([]float32{10000, 9999}, 1, 2)
	if err != nil {
		t.Fatal(err)
	}

	prediction, err := nn.NewValue(data, true)
	if err != nil {
		t.Fatal(err)
	}

	var criterion Loss[[]int] = NewCrossEntropy()

	target := []int{0}

	output, err := criterion.Forward(prediction, target)
	if err != nil {
		t.Fatal(err)
	}

	want := math.Log1p(math.Exp(-1))
	if output.Data().Rank() != 0 || math.Abs(float64(backendtest.HostData(t, output.Data())[0])-want) > 1e-6 {
		t.Fatalf("got %v, want scalar %g", output.Data(), want)
	}

	// Backward must use the labels supplied during Forward.
	target[0] = 1

	if err := output.BackwardWith(tensor.Scalar(2)); err != nil {
		t.Fatal(err)
	}

	for i, gradient := range backendtest.HostData(t, prediction.Grad()) {
		expected := 2 / (1 + math.Exp(1))
		if i == 0 {
			expected = -expected
		}

		if math.Abs(float64(gradient)-expected) > 1e-6 {
			t.Fatalf("gradient %d: got %g, want %g", i, gradient, expected)
		}
	}
}

func TestCrossEntropy_ForwardRejectsTargetsOutsideClasses(t *testing.T) {
	data, err := tensor.Zeros(1, 2)
	if err != nil {
		t.Fatal(err)
	}

	prediction, err := nn.NewValue(data, false)
	if err != nil {
		t.Fatal(err)
	}

	var criterion Loss[[]int] = NewCrossEntropy()

	if _, err := criterion.Forward(prediction, []int{2}); err == nil {
		t.Fatal("expected class index error")
	}
}

func TestCrossEntropy_GradientMatchesFiniteDifferences(t *testing.T) {
	criterion := NewCrossEntropy()

	checkLossGradient(t, []float32{0.2, -0.3, 1.1, 0.5, 0.1, -0.7}, []int{2, 3}, func(prediction *nn.Value) (*nn.Value, error) {
		return criterion.Forward(prediction, []int{2, 0})
	})
}

func TestCrossEntropy_ExtremeLogitsRemainStable(t *testing.T) {
	input := testValue(t, []float32{10000, 9999, -10000, -9999}, []int{2, 2}, true)

	output, err := NewCrossEntropy().Forward(input, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}

	if want := math.Log1p(math.Exp(-1)); math.Abs(float64(backendtest.HostData(t, output.Data())[0])-want) > 1e-6 {
		t.Fatalf("got %v, want %g", output.Data(), want)
	}

	if err := output.Backward(); err != nil {
		t.Fatal(err)
	}

	for _, gradient := range backendtest.HostData(t, input.Grad()) {
		if math.IsNaN(float64(gradient)) || math.IsInf(float64(gradient), 0) {
			t.Fatal("nonfinite gradient")
		}
	}
}

func TestCrossEntropy_ForwardRejectsInvalidTargetsAndLogits(t *testing.T) {
	for _, tt := range []struct {
		name   string
		data   []float32
		shape  []int
		target []int
	}{
		{"TargetOutsideClasses", []float32{1, 2}, []int{1, 2}, []int{2}},
		{"NegativeTarget", []float32{1, 2}, []int{1, 2}, []int{-1}},
		{"MissingTarget", []float32{1, 2}, []int{1, 2}, nil},
		{"EmptyBatch", nil, []int{0, 2}, nil},
		{"EmptyClasses", nil, []int{1, 0}, []int{0}},
		{"InfiniteLogit", []float32{1, float32(math.Inf(1))}, []int{1, 2}, []int{0}},
		{"NaNLogit", []float32{1, float32(math.NaN())}, []int{1, 2}, []int{0}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewCrossEntropy().Forward(testValue(t, tt.data, tt.shape, true), tt.target); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestCrossEntropy_ForwardRejectsConsumedPrediction(t *testing.T) {
	input := testValue(t, []float32{1, 2}, []int{1, 2}, true)

	prediction, err := nn.Mul(input, input)
	if err != nil {
		t.Fatal(err)
	}

	seed, err := tensor.Ones(1, 2)
	if err != nil {
		t.Fatal(err)
	}

	if err := prediction.BackwardWith(seed); err != nil {
		t.Fatal(err)
	}

	if _, err := NewCrossEntropy().Forward(prediction, []int{0}); err == nil {
		t.Fatal("expected consumed graph error")
	}
}
