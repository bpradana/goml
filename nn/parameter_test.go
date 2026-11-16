package nn

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/tensor"
)

func TestParameter_UpdatePreservesIdentityAndImmutableOldValue(t *testing.T) {
	original := tensor.Scalar(2)

	parameter, err := NewParameter(original)
	if err != nil {
		t.Fatal(err)
	}

	leaf := parameter.Value()
	if err := parameter.Update(tensor.Scalar(3)); err != nil {
		t.Fatal(err)
	}

	if parameter.Value() != leaf || backendtest.HostData(t, leaf.Data())[0] != 3 || backendtest.HostData(t, original)[0] != 2 {
		t.Fatal("update changed identity or mutated immutable storage")
	}
}

func TestParameter_UpdateRejectsShapeChanges(t *testing.T) {
	parameter, err := NewParameter(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	wrong, err := tensor.Zeros(1)
	if err != nil {
		t.Fatal(err)
	}

	if err := parameter.Update(wrong); err == nil {
		t.Fatal("expected shape error")
	}

	if backendtest.HostData(t, parameter.Value().Data())[0] != 2 {
		t.Fatal("invalid update changed parameter")
	}
}

func TestParameter_UpdatePreservesSavedForwardValues(t *testing.T) {
	data, err := tensor.From([]float32{2, 3}, 2)
	if err != nil {
		t.Fatal(err)
	}

	parameter, err := NewParameter(data)
	if err != nil {
		t.Fatal(err)
	}

	squared, err := Mul(parameter.Value(), parameter.Value())
	if err != nil {
		t.Fatal(err)
	}

	objective, err := Sum(squared)
	if err != nil {
		t.Fatal(err)
	}

	replacement, err := tensor.From([]float32{20, 30}, 2)
	if err != nil {
		t.Fatal(err)
	}

	if err := parameter.Update(replacement); err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	if got := backendtest.HostData(t, parameter.Value().Grad()); !slices.Equal(got, []float32{4, 6}) {
		t.Fatalf("gradient %v, want saved-forward derivative [4 6]", got)
	}
}

func TestParameter_DetachedConstantPreservesSavedForwardValue(t *testing.T) {
	parameter, err := NewParameter(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	input, err := NewValue(tensor.Scalar(3), true)
	if err != nil {
		t.Fatal(err)
	}

	product, err := Mul(input, parameter.Value().Detach())
	if err != nil {
		t.Fatal(err)
	}

	if err := parameter.Update(tensor.Scalar(20)); err != nil {
		t.Fatal(err)
	}

	if err := product.Backward(); err != nil {
		t.Fatal(err)
	}

	if got := backendtest.HostData(t, input.Grad())[0]; got != 2 {
		t.Fatalf("got %g, want saved multiplier 2", got)
	}

	if parameter.Value().Grad() != nil {
		t.Fatal("detached parameter received a gradient")
	}
}
