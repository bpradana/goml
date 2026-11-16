package tensor

import "testing"

func TestMaxPool2D_ForwardAndBackwardShapes(t *testing.T) {
	input := fromTest(t, []float32{1, 2, 3, 4, 9, 6, 7, 8, 5}, 1, 1, 3, 3)

	output, err := MaxPool2D(input, 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, output, []int{1, 1, 2, 2}, []float32{9, 9, 9, 9})
	upstream := fromTest(t, []float32{1, 2, 3, 4}, 1, 1, 2, 2)

	gradient, err := MaxPool2DBackward(input, upstream, 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, gradient, []int{1, 1, 3, 3}, []float32{0, 0, 0, 0, 10, 0, 0, 0, 0})

	if _, err := MaxPool2D(input, 2, 0, 0); err == nil {
		t.Fatal("zero stride accepted")
	}

	if _, err := MaxPool2DBackward(input, Scalar(1), 2, 1, 0); err == nil {
		t.Fatal("wrong upstream shape accepted")
	}
}
