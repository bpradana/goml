//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestConv2D_NativeMPSMatchesCPUReference(t *testing.T) {
	device := openDevice(t)

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = scope.Close() }()

	input, err := tensor.From([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	if err != nil {
		t.Fatal(err)
	}

	weights, err := tensor.Ones(9, 2)
	if err != nil {
		t.Fatal(err)
	}

	expected, err := tensor.Conv2D(input, weights, nil, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	x, err := scope.To(input)
	if err != nil {
		t.Fatal(err)
	}

	w, err := scope.To(weights)
	if err != nil {
		t.Fatal(err)
	}

	output, err := tensor.Conv2D(x, w, nil, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	assertValues(t, output, backendtest.HostData(t, expected))
}

func TestConvolution_Conformance(t *testing.T) {
	device := openDevice(t)
	backendtest.RunConvolution(t, device)
	backendtest.RunConvolutionBackward(t, device)
}

func BenchmarkConvolutionResident(b *testing.B) {
	device, err := metal.Open()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	hostInput, err := tensor.Ones(1, 8, 32, 32)
	if err != nil {
		b.Fatal(err)
	}

	input, err := hostInput.To(device)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = input.Close() }()

	hostWeight, err := tensor.Ones(72, 16)
	if err != nil {
		b.Fatal(err)
	}

	weight, err := hostWeight.To(device)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = weight.Close() }()

	hostGradient, err := tensor.Ones(1, 16, 32, 32)
	if err != nil {
		b.Fatal(err)
	}

	gradient, err := hostGradient.To(device)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = gradient.Close() }()

	warm, err := tensor.Conv2D(input, weight, nil, 3, 1, 1)
	if err != nil {
		b.Fatal(err)
	}

	dx, dw, err := tensor.Conv2DBackward(input, weight, gradient, 3, 1, 1)
	if err != nil {
		b.Fatal(err)
	}

	if err := device.Synchronize(); err != nil {
		b.Fatal(err)
	}

	_ = warm.Close()
	_ = dx.Close()
	_ = dw.Close()

	b.Run("forward", func(b *testing.B) {
		b.ReportAllocs()

		for b.Loop() {
			output, err := tensor.Conv2D(input, weight, nil, 3, 1, 1)
			if err != nil {
				b.Fatal(err)
			}

			if err := device.Synchronize(); err != nil {
				b.Fatal(err)
			}

			if err := output.Close(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("backward", func(b *testing.B) {
		b.ReportAllocs()

		for b.Loop() {
			dx, dw, err := tensor.Conv2DBackward(input, weight, gradient, 3, 1, 1)
			if err != nil {
				b.Fatal(err)
			}

			if err := device.Synchronize(); err != nil {
				b.Fatal(err)
			}

			if err := dx.Close(); err != nil {
				b.Fatal(err)
			}

			if err := dw.Close(); err != nil {
				b.Fatal(err)
			}
		}
	})
}
