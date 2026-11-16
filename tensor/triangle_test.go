package tensor

import (
	"math"
	"testing"
)

func TestMaskUpperTriangle_RectangularMatrix(t *testing.T) {
	input := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	negativeInfinity := float32(math.Inf(-1))

	output, err := MaskUpperTriangle(input, negativeInfinity)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, output, []int{2, 3}, []float32{1, negativeInfinity, negativeInfinity, 4, 5, negativeInfinity})
	assertData(t, input, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	if _, err := MaskUpperTriangle(Scalar(1), 0); err == nil {
		t.Fatal("scalar accepted as matrix")
	}
}
