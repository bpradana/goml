package nn

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestContiguous_TransposeReshapeAndGradient(t *testing.T) {
	x := testValue(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, true)

	y, err := Transpose(x)
	if err != nil {
		t.Fatal(err)
	}

	y, err = Contiguous(y)
	if err != nil {
		t.Fatal(err)
	}

	y, err = Reshape(y, 6)
	if err != nil {
		t.Fatal(err)
	}

	seed, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 6)
	if err != nil {
		t.Fatal(err)
	}

	if err := y.BackwardWith(seed); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(x.Grad().Data(), []float32{1, 3, 5, 2, 4, 6}) {
		t.Fatal(x.Grad())
	}

	if _, err := Contiguous(y); err == nil {
		t.Fatal("accepted consumed value")
	}

	if _, err := Contiguous(nil); err == nil {
		t.Fatal("accepted nil")
	}
}
