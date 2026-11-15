//go:build goexperiment.simd

package kernels

import (
	"math"
	"simd"
	"testing"
)

func TestSIMDKernels_SpecialValuesAndScalarTail(t *testing.T) {
	width := (simd.Float32s{}).Len()
	a, b := make([]float32, width+1), make([]float32, width+1)
	special := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Copysign(0, -1)), 1e-38}

	for i := range a {
		a[i] = special[i%len(special)]
		b[i] = 2
	}

	for _, tt := range []struct {
		name string
		fn   func([]float32, []float32, []float32)
		ref  func([]float32, []float32, []float32)
	}{
		{"Add", Add, addScalar},
		{"Sub", Sub, subScalar},
		{"Mul", Mul, mulScalar},
		{"Div", Div, divScalar},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, want := make([]float32, len(a)), make([]float32, len(a))
			tt.fn(got, a, b)
			tt.ref(want, a, b)

			for i, value := range got {
				if math.IsNaN(float64(want[i])) {
					if !math.IsNaN(float64(value)) {
						t.Fatalf("element %d: expected NaN, got %v", i, value)
					}

					continue
				}

				if math.Float32bits(value) != math.Float32bits(want[i]) {
					t.Fatalf("element %d: got %v, want %v", i, value, want[i])
				}
			}
		})
	}
}
