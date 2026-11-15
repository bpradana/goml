package kernels

import (
	"fmt"
	"testing"
)

func BenchmarkAdd(b *testing.B) {
	for _, n := range []int{8, 64, 1024, 1027, 65536, 1048576} {
		a, c, dst := make([]float32, n), make([]float32, n), make([]float32, n)

		for _, scalar := range []bool{false, true} {
			b.Run(fmt.Sprintf("n=%d/scalar=%t", n, scalar), func(b *testing.B) {
				b.SetBytes(int64(n * 12))
				b.ReportAllocs()

				for b.Loop() {
					if scalar {
						addScalar(dst, a, c)
					} else {
						Add(dst, a, c)
					}
				}
			})
		}
	}
}

func BenchmarkDot(b *testing.B) {
	a, c := make([]float32, 65536), make([]float32, 65536)

	b.ReportAllocs()

	for b.Loop() {
		_ = Dot(a, c)
	}
}

func BenchmarkArithmetic(b *testing.B) {
	operations := []struct {
		name      string
		kernel    func([]float32, []float32, []float32)
		reference func([]float32, []float32, []float32)
	}{
		{"Sub", Sub, subScalar},
		{"Mul", Mul, mulScalar},
		{"Div", Div, divScalar},
		{
			"Scale", func(dst, a, _ []float32) { Scale(dst, a, 0.25) },
			func(dst, a, _ []float32) { scaleScalar(dst, a, 0.25) },
		},
		{
			"Axpy", func(dst, a, c []float32) { Axpy(dst, a, c, 0.25) },
			func(dst, a, c []float32) { axpyScalar(dst, a, c, 0.25) },
		},
	}

	for _, operation := range operations {
		for _, size := range []int{8, 64, 1024, 1027, 65536, 1048576} {
			a, c, dst := make([]float32, size), make([]float32, size), make([]float32, size)
			for i := range a {
				a[i] = float32(i%17) + 0.5
				c[i] = float32(i%13) + 0.5
			}

			for _, scalar := range []bool{false, true} {
				b.Run(fmt.Sprintf("%s/n=%d/scalar=%t", operation.name, size, scalar), func(b *testing.B) {
					kernel := operation.kernel
					if scalar {
						kernel = operation.reference
					}

					b.ReportAllocs()

					for b.Loop() {
						kernel(dst, a, c)
					}
				})
			}
		}
	}
}
