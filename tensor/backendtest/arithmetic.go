package backendtest

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunArithmetic checks every arithmetic family claimed by device. At least one
// of UnaryBackend, BinaryBackend, or ScalarBackend must be implemented.
func RunArithmetic(t *testing.T, device backend.Device) {
	t.Helper()

	claimed := false

	if _, ok := device.(backend.UnaryBackend); ok {
		RunUnary(t, device)

		claimed = true
	}

	if _, ok := device.(backend.BinaryBackend); ok {
		RunBinary(t, device)

		claimed = true
	}

	if _, ok := device.(backend.ScalarBackend); ok {
		RunScalar(t, device)

		claimed = true
	}

	if !claimed {
		t.Fatal("arithmetic suite requires an arithmetic capability")
	}
}

// RunUnary checks selected unary operations and unsupported identifiers. With no
// selection it checks every operation, including derivatives. Inference-only
// backends can select their supported forward operations explicitly.
func RunUnary(t *testing.T, device backend.Device, operations ...backend.UnaryOp) {
	t.Helper()

	for _, op := range operations {
		if op < backend.Exp || op > backend.SiLUGrad {
			t.Fatalf("invalid unary suite operation %v", op)
		}
	}

	compute, ok := device.(backend.UnaryBackend)
	if !ok {
		t.Fatal("unary suite requires UnaryBackend")
	}

	values := []float32{-2, -0.5, 0, 0.5, 2}

	for _, test := range []struct {
		op       backend.UnaryOp
		function func(float64) float64
	}{
		{backend.Exp, math.Exp},
		{backend.Log, math.Log},
		{backend.Sqrt, math.Sqrt},
		{backend.ReLU, func(x float64) float64 { return math.Max(0, x) }},
		{backend.Sigmoid, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }},
		{backend.Tanh, math.Tanh},
		{backend.SiLU, func(x float64) float64 { return x / (1 + math.Exp(-x)) }},
		{backend.ReLUGrad, func(x float64) float64 {
			if x > 0 {
				return 1
			}

			return 0
		}},
		{backend.SigmoidGrad, func(x float64) float64 { p := 1 / (1 + math.Exp(-x)); return p * (1 - p) }},
		{backend.TanhGrad, func(x float64) float64 { p := math.Tanh(x); return 1 - p*p }},
		{backend.SiLUGrad, func(x float64) float64 { p := 1 / (1 + math.Exp(-x)); return p + x*p*(1-p) }},
	} {
		if len(operations) > 0 && !slices.Contains(operations, test.op) {
			continue
		}

		t.Run(fmt.Sprintf("Unary/%s", test.op), func(t *testing.T) {
			input, output := view(t, device, values, 5), view(t, device, nil, 5)

			expected := make([]float32, len(values))
			for i, x := range values {
				expected[i] = float32(test.function(float64(x)))
			}

			if err := compute.Unary(test.op, output, input); err != nil {
				t.Fatal(err)
			}

			compare(t, device, output, expected, 3e-5)
		})
	}

	t.Run("Unary/UnknownOperation", func(t *testing.T) {
		input, output := view(t, device, values, 5), view(t, device, nil, 5)
		if err := compute.Unary(255, output, input); !errors.Is(err, backend.ErrUnsupported) {
			t.Fatalf("got %v", err)
		}
	})
}

// RunBinary checks broadcast arithmetic, long tails, empty dimensions, and IEEE
// special values against pure-Go arithmetic rather than another backend.
func RunBinary(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.BinaryBackend)
	if !ok {
		t.Fatal("binary suite requires BinaryBackend")
	}

	for _, size := range []int{0, 3, 1027, 65537} {
		for _, op := range []backend.BinaryOp{backend.Add, backend.Sub, backend.Mul, backend.Div} {
			t.Run(fmt.Sprintf("Binary/%s/Size%d", op, size), func(t *testing.T) {
				values := make([]float32, 2*size)
				for i := range values {
					values[i] = float32(i%19-9) / 4
				}

				a, b, out := view(t, device, values, 2, size), view(t, device, []float32{2, 4}, 2, 1), view(t, device, nil, 2, size)

				expected := make([]float32, len(values))
				for i, x := range values {
					y := float32(2 + 2*(i/max(1, size)))

					switch op {
					case backend.Add:
						expected[i] = x + y
					case backend.Sub:
						expected[i] = x - y
					case backend.Mul:
						expected[i] = x * y
					case backend.Div:
						expected[i] = x / y
					}
				}

				if err := compute.Binary(op, out, a, b); err != nil {
					t.Fatal(err)
				}

				compare(t, device, out, expected, 1e-5)
			})
		}
	}

	t.Run("Binary/SpecialValues", func(t *testing.T) {
		values := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.Copysign(0, -1)), 1e-35, 1e35}
		a, b, out := view(t, device, values, 6), view(t, device, []float32{2}), view(t, device, nil, 6)

		want := make([]float32, len(values))
		for i, x := range values {
			want[i] = x * 2
		}

		if err := compute.Binary(backend.Mul, out, a, b); err != nil {
			t.Fatal(err)
		}

		compare(t, device, out, want, 1e-5)

		actual := make([]float32, len(values))
		if err := device.Download(out.Buffer, actual); err != nil {
			t.Fatal(err)
		}

		if !math.Signbit(float64(actual[3])) {
			t.Fatal("multiplication lost negative zero")
		}
	})
	t.Run("Binary/UnknownOperation", func(t *testing.T) {
		a, b, out := view(t, device, []float32{1}, 1), view(t, device, []float32{2}, 1), view(t, device, nil, 1)
		if err := compute.Binary(255, out, a, b); !errors.Is(err, backend.ErrUnsupported) {
			t.Fatalf("got %v", err)
		}
	})
}

// RunScalar checks scalar operations and causal masking.
func RunScalar(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.ScalarBackend)
	if !ok {
		t.Fatal("scalar suite requires ScalarBackend")
	}

	for _, test := range []struct {
		op    backend.ScalarOp
		value float32
		want  []float32
	}{
		{backend.Scale, 2, []float32{2, 4, 6, 8}},
		{backend.AddScalar, 2, []float32{3, 4, 5, 6}},
		{backend.MaskUpperTriangle, -9, []float32{1, -9, 3, 4}},
	} {
		t.Run(fmt.Sprintf("Scalar/%s", test.op), func(t *testing.T) {
			a, out := view(t, device, []float32{1, 2, 3, 4}, 2, 2), view(t, device, nil, 2, 2)
			if err := compute.Scalar(test.op, out, a, test.value); err != nil {
				t.Fatal(err)
			}

			compare(t, device, out, test.want, 1e-5)
		})
	}
}
