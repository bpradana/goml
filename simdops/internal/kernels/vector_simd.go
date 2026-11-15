//go:build goexperiment.simd

package kernels

import "simd"

func Add(dst, a, b []float32) {
	width := (simd.Float32s{}).Len()
	i := 0
	for ; i+width <= len(dst); i += width {
		simd.LoadFloat32s(a[i:]).Add(simd.LoadFloat32s(b[i:])).Store(dst[i:])
	}
	addScalar(dst[i:], a[i:], b[i:])
}

func Sub(dst, a, b []float32) {
	width := (simd.Float32s{}).Len()
	i := 0
	for ; i+width <= len(dst); i += width {
		simd.LoadFloat32s(a[i:]).Sub(simd.LoadFloat32s(b[i:])).Store(dst[i:])
	}
	subScalar(dst[i:], a[i:], b[i:])
}

func Mul(dst, a, b []float32) {
	width := (simd.Float32s{}).Len()
	i := 0
	for ; i+width <= len(dst); i += width {
		simd.LoadFloat32s(a[i:]).Mul(simd.LoadFloat32s(b[i:])).Store(dst[i:])
	}
	mulScalar(dst[i:], a[i:], b[i:])
}

func Div(dst, a, b []float32) {
	width := (simd.Float32s{}).Len()
	i := 0
	for ; i+width <= len(dst); i += width {
		simd.LoadFloat32s(a[i:]).Div(simd.LoadFloat32s(b[i:])).Store(dst[i:])
	}
	divScalar(dst[i:], a[i:], b[i:])
}

func Scale(dst, a []float32, f float32) {
	factor := simd.BroadcastFloat32s(f)
	width := factor.Len()
	i := 0
	for ; i+width <= len(dst); i += width {
		simd.LoadFloat32s(a[i:]).Mul(factor).Store(dst[i:])
	}
	scaleScalar(dst[i:], a[i:], f)
}

func Axpy(dst, x, y []float32, f float32) {
	factor := simd.BroadcastFloat32s(f)
	width := factor.Len()
	i := 0
	for ; i+width <= len(dst); i += width {
		simd.LoadFloat32s(x[i:]).Mul(factor).Add(simd.LoadFloat32s(y[i:])).Store(dst[i:])
	}
	axpyScalar(dst[i:], x[i:], y[i:], f)
}

// Keep reductions in reference order until a faster reduction is measured.
func Dot(a, b []float32) float32 { return dotScalar(a, b) }
func Sum(a []float32) float32    { return sumScalar(a) }
