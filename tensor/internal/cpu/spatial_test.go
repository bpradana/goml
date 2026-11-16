package cpu_test

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"

	"github.com/bpradana/goml/tensor/backend"
)

func TestPool_ForwardAndOverlappingDerivative(t *testing.T) {
	device := cpu.Device{}
	input := testView(t, []float32{1, 2, 3, 4, 9, 6, 7, 8, 5}, 1, 1, 3, 3)
	output := testView(t, nil, 1, 1, 2, 2)

	options := backend.Pool2DOptions{Kernel: 2, Stride: 1}
	if err := device.Pool(output, input, options); err != nil {
		t.Fatal(err)
	}

	values := make([]float32, 4)
	if err := device.Download(output.Buffer, values); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(values, []float32{9, 9, 9, 9}) {
		t.Fatalf("forward %v", values)
	}

	upstream := testView(t, []float32{1, 2, 3, 4}, 1, 1, 2, 2)

	dx := testView(t, nil, 1, 1, 3, 3)
	if err := device.PoolBackward(dx, input, upstream, options); err != nil {
		t.Fatal(err)
	}

	gradient := make([]float32, 9)
	if err := device.Download(dx.Buffer, gradient); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(gradient, []float32{0, 0, 0, 0, 10, 0, 0, 0, 0}) {
		t.Fatalf("dx %v", gradient)
	}
}

func TestPool_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunPool(t, device)
	backendtest.RunPoolBackward(t, device)
}
