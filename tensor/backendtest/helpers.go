package backendtest

import (
	"math"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func view(t *testing.T, device backend.Device, values []float32, shape ...int) backend.View {
	t.Helper()

	strides := make([]int, len(shape))

	count := 1
	for axis := len(shape) - 1; axis >= 0; axis-- {
		strides[axis] = count
		count *= shape[axis]
	}

	buffer, err := device.Allocate(count)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		if err := buffer.Close(); err != nil {
			t.Error(err)
		}
	})

	if values != nil {
		if err := device.Upload(buffer, values); err != nil {
			t.Fatal(err)
		}
	}

	return backend.View{Buffer: buffer, Shape: shape, Strides: strides}
}

func compare(t *testing.T, device backend.Device, output backend.View, want []float32, tolerance float64) {
	t.Helper()

	got := make([]float32, output.Buffer.Len())
	if err := device.Download(output.Buffer, got); err != nil {
		t.Fatal(err)
	}

	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}

	for i, x := range got {
		expected := float64(want[i])
		if math.IsNaN(expected) && math.IsNaN(float64(x)) {
			continue
		}

		if x == want[i] {
			continue
		}

		if math.IsNaN(float64(x)) || math.Abs(float64(x)-expected) > tolerance*(1+math.Abs(expected)) {
			t.Fatalf("element %d: got %g, want %g", i, x, want[i])
		}
	}
}
