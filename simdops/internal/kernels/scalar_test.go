package kernels

import (
	"slices"
	"testing"
)

func TestScalarArithmetic_KnownValues(t *testing.T) {
	a, b := []float32{2, 4, -6}, []float32{1, 2, 3}
	dst := make([]float32, 3)

	tests := []struct {
		name string
		fn   func([]float32, []float32, []float32)
		want []float32
	}{
		{"Add", addScalar, []float32{3, 6, -3}},
		{"Sub", subScalar, []float32{1, 2, -9}},
		{"Mul", mulScalar, []float32{2, 8, -18}},
		{"Div", divScalar, []float32{2, 2, -2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.fn(dst, a, b)

			if !slices.Equal(dst, tt.want) {
				t.Fatalf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestDotScalar_KnownValues(t *testing.T) {
	if got := dotScalar([]float32{2, 4, -6}, []float32{1, 2, 3}); got != -8 {
		t.Fatalf("got %v, want -8", got)
	}
}

func TestSumScalar_KnownValues(t *testing.T) {
	if got := sumScalar([]float32{2, 4, -6}); got != 0 {
		t.Fatalf("got %v, want zero", got)
	}
}

func TestScaleScalar_SupportsExactAliasing(t *testing.T) {
	data := []float32{1, -2, 3}
	scaleScalar(data, data, 2)

	if !slices.Equal(data, []float32{2, -4, 6}) {
		t.Fatal(data)
	}
}

func TestAxpyScalar_SupportsExactAliasing(t *testing.T) {
	data := []float32{1, -2, 3}
	axpyScalar(data, data, data, 2)

	if !slices.Equal(data, []float32{3, -6, 9}) {
		t.Fatal(data)
	}
}
