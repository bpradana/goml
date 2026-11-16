package tensor

import (
	"math"
	"slices"
	"testing"
)

func TestSubScaled_StridedInputsAndImmutableOutput(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	a, _ = a.Transpose()
	b := fromTest(t, []float32{2, 4, 6, 8, 10, 12}, 3, 2)

	got, err := SubScaled(a, b, 0.5)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, got, []int{3, 2}, []float32{0, 2, -1, 1, -2, 0})
	assertData(t, a, []int{3, 2}, []float32{1, 4, 2, 5, 3, 6})
	assertData(t, b, []int{3, 2}, []float32{2, 4, 6, 8, 10, 12})

	if _, err := SubScaled(a, Scalar(1), 1); err == nil {
		t.Fatal("accepted mismatched shapes")
	}
}

func TestAdamUpdate_StridedInputsMatchReference(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4}, 2, 2)
	a, _ = a.Transpose()
	g := fromTest(t, []float32{2, -4, 1, -1}, 2, 2)
	first, second := make([]float32, 4), make([]float32, 4)

	got, err := AdamUpdate(a, g, first, second, 0.01, 0.9, 0.999, 1e-8, 1)
	if err != nil {
		t.Fatal(err)
	}

	for i, want := range []float32{0.99, 3.01, 1.99, 4.01} {
		if math.Abs(float64(got.Data()[i]-want)) > 1e-6 {
			t.Fatal(i, got, want)
		}
	}

	assertData(t, a, []int{2, 2}, []float32{1, 3, 2, 4})
	assertData(t, g, []int{2, 2}, []float32{2, -4, 1, -1})

	saved := slices.Clone(first)

	if _, err := AdamUpdate(a, g, first, second[:3], 0.01, 0.9, 0.999, 1e-8, 2); err == nil || !slices.Equal(first, saved) {
		t.Fatal("invalid moment shape modified state")
	}
}

func TestSubScaled_PreservesSeparateFloat32Rounding(t *testing.T) {
	a := fromTest(t, []float32{1, -2, 1e-30, 0, float32(math.Inf(1)), float32(math.NaN())}, 6)
	b := fromTest(t, []float32{3, 7, -1e-30, float32(math.Copysign(0, -1)), 0, 1}, 6)

	for _, factor := range []float32{0, 0.1, -0.3, float32(math.Inf(1))} {
		scaled, _ := Scale(b, factor)
		want, _ := Sub(a, scaled)

		got, err := SubScaled(a, b, factor)
		if err != nil {
			t.Fatal(err)
		}

		for i, value := range got.Data() {
			expected := want.data[i]
			if math.Float32bits(value) != math.Float32bits(expected) && !(math.IsNaN(float64(value)) && math.IsNaN(float64(expected))) {
				t.Fatal(factor, i, value, expected)
			}
		}
	}
}

func TestAdamUpdate_InvalidHyperparametersLeaveMomentsIntact(t *testing.T) {
	a, _ := Ones(2)
	first, second := []float32{1, 2}, []float32{3, 4}

	for _, configuration := range [][4]float32{{0, 0.9, 0.999, 1e-8}, {0.1, 1, 0.999, 1e-8}, {0.1, 0.9, -1, 1e-8}, {0.1, 0.9, 0.999, 0}, {float32(math.NaN()), 0.9, 0.999, 1e-8}} {
		if _, err := AdamUpdate(a, a, first, second, configuration[0], configuration[1], configuration[2], configuration[3], 1); err == nil {
			t.Fatal(configuration)
		}

		if !slices.Equal(first, []float32{1, 2}) || !slices.Equal(second, []float32{3, 4}) {
			t.Fatal("failed update changed moments")
		}
	}
}
