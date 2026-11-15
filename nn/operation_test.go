package nn

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestNewOperation_UsesOutputSeedAndConsumesGraph(t *testing.T) {
	input := testValue(t, []float32{3}, nil, true)

	output, err := NewOperation(tensor.Scalar(9), input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Scale(g, 6)
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := output.BackwardWith(tensor.Scalar(2)); err != nil {
		t.Fatal(err)
	}

	if got := input.Grad().Data()[0]; got != 12 {
		t.Fatalf("got gradient %g, want 12", got)
	}

	if err := output.Backward(); err == nil {
		t.Fatal("expected consumed graph error")
	}
}

func TestNewOperation_RejectsInvalidConstruction(t *testing.T) {
	tracked := testValue(t, []float32{1}, nil, true)

	for _, tt := range []struct {
		name  string
		data  *tensor.Tensor
		input *Value
	}{
		{"NilResult", nil, tracked},
		{"UninitializedResult", &tensor.Tensor{}, tracked},
		{"NilInput", tensor.Scalar(1), nil},
		{"UninitializedInput", tensor.Scalar(1), &Value{}},
		{"MissingBackwardRule", tensor.Scalar(1), tracked},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewOperation(tt.data, tt.input, nil); err == nil {
				t.Fatal("expected construction error")
			}
		})
	}
}

func TestNewOperation_InferenceRetainsNoBackwardState(t *testing.T) {
	input, err := Inference(tensor.Scalar(1))
	if err != nil {
		t.Fatal(err)
	}

	called := false

	output, err := NewOperation(tensor.Scalar(2), input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		called = true
		return g, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if output.RequiresGrad() || output.parents != nil || output.backward != nil || called {
		t.Fatal("inference recorded or executed a backward rule")
	}
}

func TestNewOperation_RejectsIncorrectGradientShapeWithoutAccumulation(t *testing.T) {
	input := testValue(t, []float32{1}, nil, true)

	output, err := NewOperation(tensor.Scalar(2), input, func(*tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Ones(1)
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := output.Backward(); err == nil {
		t.Fatal("expected gradient shape error")
	}

	if input.Grad() != nil || output.Grad() != nil || output.consumed {
		t.Fatal("failed backward changed gradients or consumed the graph")
	}
}

func TestNewOperation_PropagatesBackwardErrors(t *testing.T) {
	expected := errors.New("gradient calculation failed")
	input := testValue(t, []float32{1}, nil, true)

	output, err := NewOperation(tensor.Scalar(2), input, func(*tensor.Tensor) (*tensor.Tensor, error) {
		return nil, expected
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := output.Backward(); !errors.Is(err, expected) {
		t.Fatalf("got %v, want wrapped error %v", err, expected)
	}
}
