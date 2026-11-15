package simdops

import (
	"fmt"

	"github.com/bpradana/goml/simdops/internal/kernels"
)

// Add writes the element-wise sum of a and b to dst.
// All lengths must match; validation errors leave dst unchanged.
// Exact aliasing is supported. NaNs and infinities follow IEEE 754.
func Add(dst, a, b []float32) error {
	if err := checkLengths("add", dst, a, b); err != nil {
		return err
	}

	kernels.Add(dst, a, b)

	return nil
}

// Sub writes the element-wise difference of a and b to dst.
// All lengths must match; validation errors leave dst unchanged.
// Exact aliasing is supported. NaNs and infinities follow IEEE 754.
func Sub(dst, a, b []float32) error {
	if err := checkLengths("sub", dst, a, b); err != nil {
		return err
	}

	kernels.Sub(dst, a, b)

	return nil
}

// Mul writes the element-wise product of a and b to dst.
// All lengths must match; validation errors leave dst unchanged.
// Exact aliasing is supported. NaNs and infinities follow IEEE 754.
func Mul(dst, a, b []float32) error {
	if err := checkLengths("mul", dst, a, b); err != nil {
		return err
	}

	kernels.Mul(dst, a, b)

	return nil
}

// Div writes the element-wise quotient of a and b to dst.
// All lengths must match; validation errors leave dst unchanged.
// Exact aliasing is supported. Division by zero follows IEEE 754.
func Div(dst, a, b []float32) error {
	if err := checkLengths("div", dst, a, b); err != nil {
		return err
	}

	kernels.Div(dst, a, b)

	return nil
}

// Scale writes a*factor to dst. Lengths must match; exact aliasing is supported.
// It allocates no memory. Invalid lengths leave dst unchanged.
func Scale(dst, a []float32, factor float32) error {
	if len(dst) != len(a) {
		return fmt.Errorf("simdops: scale: lengths %d and %d differ", len(dst), len(a))
	}

	kernels.Scale(dst, a, factor)

	return nil
}

// Axpy writes alpha*x+y to dst, without allocations. Lengths must match;
// exact aliasing is supported. Invalid lengths leave dst unchanged.
func Axpy(dst, x, y []float32, alpha float32) error {
	if err := checkLengths("axpy", dst, x, y); err != nil {
		return err
	}

	kernels.Axpy(dst, x, y, alpha)

	return nil
}

func checkLengths(op string, dst, a, b []float32) error {
	if len(dst) != len(a) || len(a) != len(b) {
		return fmt.Errorf("simdops: %s: lengths dst=%d a=%d b=%d differ", op, len(dst), len(a), len(b))
	}

	return nil
}
