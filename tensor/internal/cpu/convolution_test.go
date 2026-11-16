package cpu_test

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"

	"github.com/bpradana/goml/tensor/backend"
)

func TestConv2D_ForwardAndBothDerivatives(t *testing.T) {
	device := cpu.Device{}
	input := testView(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	weight := testView(t, []float32{1, 1, 1, 1}, 4, 1)
	output := testView(t, nil, 1, 1, 2, 2)

	options := backend.Conv2DOptions{Kernel: 2, Stride: 1}
	if err := device.Conv2D(output, input, weight, backend.View{}, options); err != nil {
		t.Fatal(err)
	}

	values := make([]float32, 4)
	if err := device.Download(output.Buffer, values); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(values, []float32{12, 16, 24, 28}) {
		t.Fatalf("forward %v", values)
	}

	upstream := testView(t, []float32{1, 1, 1, 1}, 1, 1, 2, 2)

	dx, dw := testView(t, nil, 1, 1, 3, 3), testView(t, nil, 4, 1)
	if err := device.Conv2DBackward(dx, dw, input, weight, upstream, options); err != nil {
		t.Fatal(err)
	}

	xvalues := make([]float32, 9)
	if err := device.Download(dx.Buffer, xvalues); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(xvalues, []float32{1, 2, 1, 2, 4, 2, 1, 2, 1}) {
		t.Fatalf("dx %v", xvalues)
	}

	if err := device.Download(dw.Buffer, values); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(values, []float32{12, 16, 24, 28}) {
		t.Fatalf("dw %v", values)
	}
}

func TestConv2D_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunConvolution(t, device)
}

func TestIm2ColAndCol2Im_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunPatch(t, device)
}
