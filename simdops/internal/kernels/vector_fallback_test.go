//go:build !goexperiment.simd

package kernels

import (
	"slices"
	"testing"
)

func TestFallbackArithmetic_IrregularLengthMatchesScalar(t *testing.T) {
	a, b := []float32{1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1}
	got, want := make([]float32, len(a)), make([]float32, len(a))

	for _, tt := range []struct {
		name      string
		kernel    func([]float32, []float32, []float32)
		reference func([]float32, []float32, []float32)
	}{
		{"Add", Add, addScalar},
		{"Sub", Sub, subScalar},
		{"Mul", Mul, mulScalar},
		{"Div", Div, divScalar},
		{
			"Scale", func(dst, a, _ []float32) { Scale(dst, a, 0.5) },
			func(dst, a, _ []float32) { scaleScalar(dst, a, 0.5) },
		},
		{
			"Axpy", func(dst, a, b []float32) { Axpy(dst, a, b, 0.5) },
			func(dst, a, b []float32) { axpyScalar(dst, a, b, 0.5) },
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tt.kernel(got, a, b)
			tt.reference(want, a, b)

			if !slices.Equal(got, want) {
				t.Fatalf("got %v, want %v", got, want)
			}
		})
	}
}

func TestFallbackReductions_KnownValues(t *testing.T) {
	a, b := []float32{1, 2, 3}, []float32{3, 2, 1}
	if Dot(a, b) != 10 || Sum(a) != 6 || Max(a) != 3 {
		t.Fatal("incorrect fallback reduction")
	}
}
