package tensor

import "testing"

func TestConv2DBackward_CPUShapesAndValues(t *testing.T) {
	input := fromTest(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)

	weight, err := Ones(4, 2)
	if err != nil {
		t.Fatal(err)
	}

	gradient, err := Ones(1, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	dx, dw, err := Conv2DBackward(input, weight, gradient, 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, dx, []int{1, 1, 3, 3}, []float32{2, 4, 2, 4, 8, 4, 2, 4, 2})
	assertData(t, dw, []int{4, 2}, []float32{12, 12, 16, 16, 24, 24, 28, 28})

	if _, _, err := Conv2DBackward(input, weight, Scalar(1), 2, 1, 0); err == nil {
		t.Fatal("invalid upstream shape accepted")
	}
}
