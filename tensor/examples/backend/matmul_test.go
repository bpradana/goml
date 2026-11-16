package examplebackend_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/backendtest"
	examplebackend "github.com/bpradana/goml/tensor/examples/backend"
)

func TestMatMul_Conformance(t *testing.T) {
	device, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	backendtest.RunMatMul(t, device)
}

func TestMatMul_PublicTensorIntegrationAndMissingCapability(t *testing.T) {
	device, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	a, err := tensor.FromOn(device, []float32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = a.Close() }()

	result, err := tensor.MatMul(a, a)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = result.Close() }()

	if !slices.Equal(backendtest.HostData(t, result), []float32{7, 10, 15, 22}) {
		t.Fatal("wrong product")
	}

	if _, err := tensor.Add(a, a); !errors.Is(err, backend.ErrUnsupported) {
		t.Fatalf("missing binary capability: %v", err)
	}
}

func TestMatMul_HostReadsDoNotRequireCopyCapability(t *testing.T) {
	device, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	input, err := tensor.FromOn(device, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = input.Close() }()

	transposed, err := input.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = transposed.Close() }()

	if got := backendtest.HostData(t, transposed); !slices.Equal(got, []float32{1, 4, 2, 5, 3, 6}) {
		t.Fatalf("logical values %v", got)
	}
}
