package cpu_test

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

func TestMatMul_RectangularAndStridedInputs(t *testing.T) {
	device := cpu.Device{}
	a := testView(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := testView(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b.Shape = []int{3, 2}
	b.Strides = []int{1, 3}

	out := testView(t, nil, 2, 2)
	if err := device.MatMul(out, a, b); err != nil {
		t.Fatal(err)
	}

	values := make([]float32, 4)
	if err := device.Download(out.Buffer, values); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(values, []float32{14, 32, 32, 77}) {
		t.Fatalf("got %v", values)
	}
}

func TestMatMul_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunMatMul(t, device)
}
