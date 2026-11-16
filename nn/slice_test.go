package nn

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestSlice_NoncontiguousViewScattersGradient(t *testing.T) {
	x := testValue(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, true)

	y, err := Slice(x, 1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, y.Data()), []float32{2, 3, 5, 6}) {
		t.Fatal(y.Data())
	}

	objective, err := Sum(y)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, x.Grad()), []float32{0, 1, 1, 0, 1, 1}) {
		t.Fatal(x.Grad())
	}
}

func TestSlice_EmptyRangeAndInvalidBounds(t *testing.T) {
	x := testValue(t, []float32{1, 2}, []int{2}, true)

	y, err := Slice(x, 0, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	objective, err := Sum(y)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, x.Grad()), []float32{0, 0}) {
		t.Fatal(x.Grad())
	}

	for _, bounds := range [][3]int{{-1, 0, 1}, {1, 0, 1}, {0, -1, 1}, {0, 0, 3}, {0, 2, 1}} {
		if _, err := Slice(x, bounds[0], bounds[1], bounds[2]); err == nil {
			t.Fatalf("accepted %v", bounds)
		}
	}

	if _, err := Slice(nil, 0, 0, 0); err == nil {
		t.Fatal("accepted nil")
	}
}
