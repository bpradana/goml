//go:build darwin && cgo

package metal_test

import (
	"testing"

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
