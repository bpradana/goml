package webgpu

import (
	"math"
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestArithmetic_UnaryActivationsAndDerivatives(t *testing.T) {
	backendtest.RunUnary(t, openDevice(t))
}

func TestArithmetic_BinaryBroadcastsTailsAndSpecialValues(t *testing.T) {
	backendtest.RunBinary(t, openDevice(t))
}

func TestArithmetic_ScalarAndCausalMask(t *testing.T) {
	backendtest.RunScalar(t, openDevice(t))
}

func TestArithmetic_IEEEBinaryResultsMatchCPU(t *testing.T) {
	device := openDevice(t)
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	av := []float32{0, 1, -1, inf, inf, inf, -inf, nan, float32(math.Copysign(0, -1))}
	bv := []float32{0, 0, 0, inf, -inf, 0, 2, 1, 3}

	a, err := tensor.From(av, len(av))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = a.Close() }()

	b, err := tensor.From(bv, len(bv))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()

	ga, err := a.To(device)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ga.Close() }()

	gb, err := b.To(device)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = gb.Close() }()

	for _, test := range []struct {
		name string
		run  func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, error)
	}{
		{"Add", tensor.Add}, {"Sub", tensor.Sub}, {"Mul", tensor.Mul}, {"Div", tensor.Div},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.run(ga, gb)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = got.Close() }()

			want, err := test.run(a, b)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = want.Close() }()

			compareTensors(t, got, want, 1e-5)

			actual, expected := backendtest.HostData(t, got), backendtest.HostData(t, want)
			for i, value := range expected {
				if value == 0 && math.Signbit(float64(actual[i])) != math.Signbit(float64(value)) {
					t.Fatalf("element %d lost zero sign", i)
				}
			}
		})
	}
}

func TestArithmetic_UnaryInfinitiesAndSignedZeroMatchCPU(t *testing.T) {
	device := openDevice(t)
	values := []float32{float32(math.Inf(-1)), -2, float32(math.Copysign(0, -1)), 0, 2, float32(math.Inf(1)), float32(math.NaN())}

	execute := func(t *testing.T, context backend.Device, op backend.UnaryOp) []float32 {
		t.Helper()

		input, err := context.Allocate(len(values))
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = input.Close() }()

		output, err := context.Allocate(len(values))
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = output.Close() }()

		if err := context.Upload(input, values); err != nil {
			t.Fatal(err)
		}

		source := backend.View{Buffer: input, Shape: []int{len(values)}, Strides: []int{1}}

		target := backend.View{Buffer: output, Shape: source.Shape, Strides: source.Strides}
		if err := context.(backend.UnaryBackend).Unary(op, target, source); err != nil {
			t.Fatal(err)
		}

		result := make([]float32, len(values))
		if err := context.Download(output, result); err != nil {
			t.Fatal(err)
		}

		return result
	}

	for op := backend.Exp; op <= backend.SiLUGrad; op++ {
		t.Run(op.String(), func(t *testing.T) {
			actual, expected := execute(t, device, op), execute(t, tensor.CPU(), op)
			for i, want := range expected {
				got := actual[i]

				switch {
				case math.IsNaN(float64(want)):
					if !math.IsNaN(float64(got)) {
						t.Fatalf("element %d: got %g, want NaN", i, got)
					}
				case math.IsInf(float64(want), 0):
					if got != want {
						t.Fatalf("element %d: got %g, want %g", i, got, want)
					}
				case want == 0:
					if math.Float32bits(got) != math.Float32bits(want) {
						t.Fatalf("element %d: got zero bits %08x, want %08x", i, math.Float32bits(got), math.Float32bits(want))
					}
				default:
					if difference := math.Abs(float64(got - want)); math.IsNaN(difference) || difference > 3e-5+3e-5*math.Abs(float64(want)) {
						t.Fatalf("element %d: got %g, want %g", i, got, want)
					}
				}
			}
		})
	}
}
