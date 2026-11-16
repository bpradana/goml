package layer

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestMaxPool2D_PaddedNegativeInputsAndOverlappingGradients(t *testing.T) {
	pool, err := NewMaxPool2D(3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	input := layerValue(t, []float32{-5, -2, -3, -4}, []int{1, 1, 2, 2}, true)

	output, err := pool.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Data(), []float32{-2, -2, -2, -2}) {
		t.Fatal(output.Data())
	}

	seed, _ := tensor.Ones(1, 1, 2, 2)
	if err := output.BackwardWith(seed); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(input.Grad().Data(), []float32{0, 4, 0, 0}) {
		t.Fatal(input.Grad())
	}

	checkLayerGradients(t, pool, []float32{-5, -2, -3, -4}, []int{1, 1, 2, 2})
}

func TestMaxPool2D_TiesInfinityAndNaN(t *testing.T) {
	pool, _ := NewMaxPool2D(2, 2, 0)

	for _, values := range [][]float32{{3, 3, 1, 2}, {float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))}, {1, float32(math.NaN()), float32(math.NaN()), 2}} {
		input := layerValue(t, values, []int{1, 1, 2, 2}, true)

		output, err := pool.Forward(input)
		if err != nil {
			t.Fatal(err)
		}

		seed, _ := tensor.Ones(1, 1, 1, 1)
		if err := output.BackwardWith(seed); err != nil {
			t.Fatal(err)
		}

		want := []float32{1, 0, 0, 0}
		if math.IsNaN(float64(values[1])) {
			want = []float32{0, 1, 0, 0}

			if !math.IsNaN(float64(output.Data().Data()[0])) {
				t.Fatal("NaN did not propagate")
			}
		}

		if !slices.Equal(input.Grad().Data(), want) {
			t.Fatal(input.Grad())
		}
	}
}

func TestMaxPool2D_MultipleChannelsStrideAndEmptyBatch(t *testing.T) {
	pool, _ := NewMaxPool2D(2, 2, 0)
	input := layerValue(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{1, 2, 2, 3}, false)

	output, err := pool.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Shape(), []int{1, 2, 1, 1}) || !slices.Equal(output.Data().Data(), []float32{5, 11}) {
		t.Fatal(output.Data())
	}

	if _, err := pool.Forward(layerValue(t, nil, []int{0, 2, 3, 3}, false)); err != nil {
		t.Fatal(err)
	}
}

func TestMaxPool2D_RejectsInvalidGeometryAndInput(t *testing.T) {
	for _, geometry := range [][3]int{{0, 1, 0}, {2, 0, 0}, {2, 1, -1}, {2, 1, 2}} {
		if _, err := NewMaxPool2D(geometry[0], geometry[1], geometry[2]); err == nil {
			t.Fatal(geometry)
		}
	}

	pool, _ := NewMaxPool2D(3, 1, 0)
	if _, err := pool.Forward(nil); err == nil {
		t.Fatal("nil input accepted")
	}

	for _, shape := range [][]int{{1, 2}, {1, 1, 0, 2}, {1, 1, 2, 2}} {
		data, _ := tensor.Zeros(shape...)
		if _, err := pool.Forward(layerValue(t, data.Data(), shape, false)); err == nil {
			t.Fatal(shape)
		}
	}
}
