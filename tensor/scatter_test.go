package tensor

import "testing"

func TestScatterSlice_CPUValuesAndMismatchedDimensions(t *testing.T) {
	gradient := fromTest(t, []float32{2, 3}, 2, 1)

	result, err := ScatterSlice(gradient, []int{2, 3}, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, result, []int{2, 3}, []float32{0, 2, 0, 0, 3, 0})

	if _, err := ScatterSlice(gradient, []int{3, 3}, 1, 1); err == nil {
		t.Fatal("mismatched non-axis dimension accepted")
	}

	if _, err := ScatterSlice(gradient, []int{2, 3}, 1, 3); err == nil {
		t.Fatal("slice exceeding output accepted")
	}
}
