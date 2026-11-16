//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestPool_MetalForwardAndOverlappingGradientsMatchCPU(t *testing.T) {
	device := openDevice(t)

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = scope.Close() }()

	values := make([]float32, 48)
	for index := range values {
		values[index] = float32(index%17-8) / 7
	}

	input, err := tensor.From(values, 2, 2, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	expected, err := tensor.MaxPool2D(input, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	seed, err := tensor.Ones(expected.Shape()...)
	if err != nil {
		t.Fatal(err)
	}

	expectedGradient, err := tensor.MaxPool2DBackward(input, seed, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	gpuInput, err := scope.To(input)
	if err != nil {
		t.Fatal(err)
	}

	gpuSeed, err := scope.To(seed)
	if err != nil {
		t.Fatal(err)
	}

	output, err := tensor.MaxPool2D(gpuInput, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	gradient, err := tensor.MaxPool2DBackward(gpuInput, gpuSeed, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	assertValues(t, output, expected.Data())
	assertValues(t, gradient, expectedGradient.Data())
}

func TestUpsample_MetalAllModesAndGradientsMatchCPU(t *testing.T) {
	device := openDevice(t)

	for _, mode := range []string{"nearest", "bilinear", "bicubic"} {
		t.Run(mode, func(t *testing.T) {
			scope, err := tensor.NewScope(device)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = scope.Close() }()

			input, err := tensor.From([]float32{-1, 2, 3, 0.5, 4, -2}, 1, 1, 2, 3)
			if err != nil {
				t.Fatal(err)
			}

			expected, err := tensor.Upsample2D(input, 2, mode)
			if err != nil {
				t.Fatal(err)
			}

			values := make([]float32, expected.Numel())
			for index := range values {
				values[index] = float32(index%7-3) / 5
			}

			seed, err := tensor.From(values, expected.Shape()...)
			if err != nil {
				t.Fatal(err)
			}

			expectedGradient, err := tensor.Upsample2DBackward(seed, input.Shape(), 2, mode)
			if err != nil {
				t.Fatal(err)
			}

			gpuInput, err := scope.To(input)
			if err != nil {
				t.Fatal(err)
			}

			gpuSeed, err := scope.To(seed)
			if err != nil {
				t.Fatal(err)
			}

			output, err := tensor.Upsample2D(gpuInput, 2, mode)
			if err != nil {
				t.Fatal(err)
			}

			gradient, err := tensor.Upsample2DBackward(gpuSeed, input.Shape(), 2, mode)
			if err != nil {
				t.Fatal(err)
			}

			assertValues(t, output, expected.Data())
			assertValues(t, gradient, expectedGradient.Data())
		})
	}
}
