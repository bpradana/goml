//go:build !goexperiment.simd

package kernels

func Add(dst, a, b []float32) { addScalar(dst, a, b) }
func Sub(dst, a, b []float32) { subScalar(dst, a, b) }
func Mul(dst, a, b []float32) { mulScalar(dst, a, b) }
func Div(dst, a, b []float32) { divScalar(dst, a, b) }

func Scale(dst, a []float32, f float32)   { scaleScalar(dst, a, f) }
func Axpy(dst, x, y []float32, f float32) { axpyScalar(dst, x, y, f) }
func Dot(a, b []float32) float32          { return dotScalar(a, b) }
func Sum(a []float32) float32             { return sumScalar(a) }
