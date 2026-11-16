//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestCompute_BroadcastArithmeticAndReductionsStayOnDevice(t *testing.T) {
	device := openDevice(t)

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = scope.Close() }()

	source, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	a, err := scope.To(source)
	if err != nil {
		t.Fatal(err)
	}

	b, err := scope.To(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	product, err := tensor.Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}

	reduced, err := tensor.Sum(product, 0)
	if err != nil {
		t.Fatal(err)
	}

	if reduced.Device() != device {
		t.Fatal("result left the GPU")
	}

	assertValues(t, reduced, []float32{10, 14, 18})

	if _, err := tensor.Add(a, source); err == nil {
		t.Fatal("mixed CPU and GPU arithmetic accepted")
	}
}

func TestArithmetic_Conformance(t *testing.T) {
	device := openDevice(t)
	backendtest.RunArithmetic(t, device)
}
