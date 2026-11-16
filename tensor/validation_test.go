package tensor

import (
	"math"
	"testing"
)

func TestCheckFinite_RejectsNaNAndInfinityInLogicalViews(t *testing.T) {
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		input := fromTest(t, []float32{1, value}, 2)
		if err := CheckFinite(input); err == nil {
			t.Fatalf("accepted %v", value)
		}

		finite, err := input.Slice(0, 0, 1)
		if err != nil {
			t.Fatal(err)
		}

		if err := CheckFinite(finite); err != nil {
			t.Fatal(err)
		}
	}

	empty, err := Zeros(0)
	if err != nil {
		t.Fatal(err)
	}

	if err := CheckFinite(empty); err != nil {
		t.Fatal(err)
	}
}
