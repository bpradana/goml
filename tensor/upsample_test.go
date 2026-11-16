package tensor

import "testing"

func TestUpsample2D_LogicalValuesAndModeValidation(t *testing.T) {
	input := fromTest(t, []float32{1, 2}, 1, 1, 1, 2)

	output, err := Upsample2D(input, 2, "nearest")
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, output, []int{1, 1, 2, 4}, []float32{1, 1, 2, 2, 1, 1, 2, 2})

	gradient, err := Ones(1, 1, 2, 4)
	if err != nil {
		t.Fatal(err)
	}

	dx, err := Upsample2DBackward(gradient, input.Shape(), 2, "nearest")
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, dx, input.Shape(), []float32{4, 4})

	for _, mode := range []string{"unknown", "", "Nearest"} {
		if _, err := Upsample2D(input, 2, mode); err == nil {
			t.Fatalf("mode %q accepted", mode)
		}
	}

	if _, err := Upsample2D(input, 0, "nearest"); err == nil {
		t.Fatal("zero scale accepted")
	}

	if _, err := Upsample2DBackward(Scalar(1), input.Shape(), 2, "nearest"); err == nil {
		t.Fatal("wrong upstream shape accepted")
	}
}
