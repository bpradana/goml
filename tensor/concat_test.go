package tensor

import "testing"

func TestConcat_StridedInputsAndEmptyDimensions(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	a, _ = a.Transpose()
	b := fromTest(t, []float32{7, 8, 9}, 3, 1)
	empty, _ := Zeros(3, 0)

	got, err := Concat(1, a, empty, b)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, got, []int{3, 3}, []float32{1, 4, 7, 2, 5, 8, 3, 6, 9})
	assertData(t, a, []int{3, 2}, []float32{1, 4, 2, 5, 3, 6})

	for _, inputs := range [][]*Tensor{nil, {a, Scalar(1)}, {a, b}, {nil}, {{}}} {
		if _, err := Concat(0, inputs...); err == nil {
			t.Fatal("accepted invalid input", inputs)
		}
	}

	for _, axis := range []int{-1, 2} {
		if _, err := Concat(axis, a); err == nil {
			t.Fatal("accepted invalid axis", axis)
		}
	}

	empty, _ = Zeros(0, 2)

	got, err = Concat(0, empty, empty)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, got, []int{0, 2}, nil)
}
