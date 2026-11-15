package nn

import (
	"testing"

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

	if parameter.Value() != leaf || leaf.Data().Data()[0] != 3 || original.Data()[0] != 2 {
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

	if parameter.Value().Data().Data()[0] != 2 {
		t.Fatal("invalid update changed parameter")
	}
}
