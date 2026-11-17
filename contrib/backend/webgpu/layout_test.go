package webgpu

import (
	"math"
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestLayout_FillAndEmptyInitialization(t *testing.T) {
	backendtest.RunFill(t, openDevice(t))
}

func TestLayout_StridedCopiesScatterAndPendingResources(t *testing.T) {
	backendtest.RunCopy(t, openDevice(t))
}

func TestLayout_Concatenation(t *testing.T) {
	backendtest.RunConcat(t, openDevice(t))
}

func TestCopy_StridedMaterializationPreservesFloat32Bits(t *testing.T) {
	device := openDevice(t)
	bits := []uint32{0x80000000, 0x7fc12345, 0x7fa12345, 0xff800000}

	values := make([]float32, len(bits))
	for i, value := range bits {
		values[i] = math.Float32frombits(value)
	}

	input, err := tensor.FromOn(device, values, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = input.Close() }()

	transposed, err := input.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = transposed.Close() }()

	actual := backendtest.HostData(t, transposed)
	for i, source := range []int{0, 2, 1, 3} {
		if math.Float32bits(actual[i]) != bits[source] {
			t.Fatalf("element %d: bits %08x, want %08x", i, math.Float32bits(actual[i]), bits[source])
		}
	}
}
