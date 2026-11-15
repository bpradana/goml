package nn

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestNewMultiOperation_RepeatedInputsAccumulateAndInputListIsCopied(t *testing.T) {
	x := testValue(t, []float32{3}, nil, true)
	inputs := []*Value{x, x}

	out, err := NewMultiOperation(tensor.Scalar(9), inputs, func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		first, err := tensor.Scale(g, 2)
		return []*tensor.Tensor{first, g}, err
	})
	if err != nil {
		t.Fatal(err)
	}

	inputs[0] = nil

	if err := out.Backward(); err != nil {
		t.Fatal(err)
	}

	if x.Grad().Data()[0] != 3 {
		t.Fatalf("gradient %v, want 3", x.Grad())
	}

	if _, err := Add(out, x); err == nil {
		t.Fatal("consumed output accepted")
	}
}

func TestNewMultiOperation_InferenceSuppressesTrainableParents(t *testing.T) {
	x, err := Inference(tensor.Scalar(1))
	if err != nil {
		t.Fatal(err)
	}

	w := testValue(t, []float32{2}, nil, true)

	out, err := NewMultiOperation(tensor.Scalar(3), []*Value{x, w}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !out.IsInference() || out.RequiresGrad() || len(out.parents) != 0 || out.backward != nil {
		t.Fatal("inference retained a graph")
	}
}

func TestNewMultiOperation_RejectsInvalidConstruction(t *testing.T) {
	x := testValue(t, []float32{1}, nil, true)
	for _, tt := range []struct {
		name   string
		data   *tensor.Tensor
		inputs []*Value
	}{
		{"NoInputs", tensor.Scalar(1), nil},
		{"NilInput", tensor.Scalar(1), []*Value{nil}},
		{"UninitializedInput", tensor.Scalar(1), []*Value{{}}},
		{"NilResult", nil, []*Value{x}},
		{"UninitializedResult", &tensor.Tensor{}, []*Value{x}},
		{"MissingRule", tensor.Scalar(1), []*Value{x}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewMultiOperation(tt.data, tt.inputs, nil); err == nil {
				t.Fatal("invalid construction accepted")
			}
		})
	}
}

func TestNewMultiOperation_BackwardValidationIsAtomic(t *testing.T) {
	expected := errors.New("rule failure")
	for _, tt := range []struct {
		name      string
		gradients []*tensor.Tensor
		err       error
	}{
		{"Count", nil, nil},
		{"Shape", []*tensor.Tensor{tensor.Scalar(1)}, nil},
		{"NilGradient", []*tensor.Tensor{nil}, nil},
		{"RuleError", nil, expected},
	} {
		t.Run(tt.name, func(t *testing.T) {
			x := testValue(t, []float32{1}, []int{1}, true)

			out, err := NewMultiOperation(tensor.Scalar(1), []*Value{x}, func(*tensor.Tensor) ([]*tensor.Tensor, error) { return tt.gradients, tt.err })
			if err != nil {
				t.Fatal(err)
			}

			err = out.Backward()
			if err == nil || (tt.err != nil && !errors.Is(err, expected)) {
				t.Fatalf("unexpected error %v", err)
			}

			if x.Grad() != nil || out.Grad() != nil || out.consumed {
				t.Fatal("failed rule changed graph or gradients")
			}
		})
	}
}
