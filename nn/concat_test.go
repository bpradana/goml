package nn

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/tensor"
)

func TestConcat_NoncontiguousInputsAndSeededGradients(t *testing.T) {
	x := testValue(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, true)

	a, err := Slice(x, 1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	b, err := Slice(x, 1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}

	y, err := Concat(1, b, a)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, y.Data()), []float32{2, 3, 1, 5, 6, 4}) {
		t.Fatal(y.Data())
	}

	seed, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	if err := y.BackwardWith(seed); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, x.Grad()), []float32{3, 1, 2, 6, 4, 5}) {
		t.Fatal(x.Grad())
	}
}

func TestConcat_EmptyDimensionsAndInvalidShapes(t *testing.T) {
	x := testValue(t, nil, []int{0, 2}, true)

	y, err := Concat(1, x, x)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(y.Data().Shape(), []int{0, 4}) {
		t.Fatal(y.Data())
	}

	objective, err := Sum(y)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	if x.Grad().Numel() != 0 {
		t.Fatal(x.Grad())
	}

	for _, tt := range []struct {
		name   string
		axis   int
		inputs []*Value
	}{
		{"NoInputs", 0, nil},
		{"Nil", 0, []*Value{nil}},
		{"Axis", 2, []*Value{x}},
		{"Rank", 0, []*Value{x, testValue(t, nil, []int{0}, false)}},
		{"Shape", 0, []*Value{x, testValue(t, nil, []int{0, 3}, false)}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := Concat(tt.axis, tt.inputs...); err == nil {
				t.Fatal("invalid concat accepted")
			}
		})
	}
}

func TestConcat_RejectsDimensionOverflowForLargeEmptyInputs(t *testing.T) {
	input := testValue(t, nil, []int{0, int(^uint(0) >> 1)}, false)
	if _, err := Concat(1, input, input); err == nil {
		t.Fatal("accepted concatenated dimension overflow")
	}
}
