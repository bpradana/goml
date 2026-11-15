package simdops

import (
	"fmt"

	"github.com/bpradana/goml/simdops/internal/kernels"
)

// Dot returns the dot product of equally sized inputs, or an error on mismatch.
// Empty inputs return zero. No memory is allocated on success.
func Dot(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("simdops: dot: lengths %d and %d differ", len(a), len(b))
	}

	return kernels.Dot(a, b), nil
}

// Sum returns the sum of a, or zero for an empty slice. It does not allocate.
func Sum(a []float32) float32 { return kernels.Sum(a) }

// Max returns the maximum element, propagating NaN and preferring positive zero.
// Empty input returns an error. No memory is allocated on success.
func Max(a []float32) (float32, error) {
	if len(a) == 0 {
		return 0, fmt.Errorf("simdops: max: empty input")
	}

	return kernels.Max(a), nil
}
