package kernels

import "math"

func addScalar(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subScalar(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulScalar(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divScalar(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleScalar(dst, a []float32, factor float32) {
	for i := range dst {
		dst[i] = a[i] * factor
	}
}

func axpyScalar(dst, x, y []float32, alpha float32) {
	for i := range dst {
		dst[i] = alpha*x[i] + y[i]
	}
}

func dotScalar(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}

	return s
}

func sumScalar(a []float32) float32 {
	var s float32
	for _, v := range a {
		s += v
	}

	return s
}

func Max(a []float32) float32 {
	result := a[0]
	for _, value := range a[1:] {
		result = float32(math.Max(float64(result), float64(value)))
	}

	return result
}
