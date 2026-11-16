package backendtest

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunConcat checks fused joining, strided inputs, empty allocations, and owners.
func RunConcat(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.ConcatBackend)
	if !ok {
		t.Fatal("concat suite requires ConcatBackend")
	}

	t.Run("Concat/StridedInputs", func(t *testing.T) {
		a := view(t, device, []float32{1, 2, 3, 4}, 2, 2)
		a.Strides = []int{1, 2}

		b, out := view(t, device, []float32{9, 10}, 2, 1), view(t, device, nil, 2, 3)
		if err := compute.Concat(out, 1, []backend.View{a, b}); err != nil {
			t.Fatal(err)
		}

		compare(t, device, out, []float32{1, 3, 9, 2, 4, 10}, 0)
	})
	t.Run("Concat/EmptyForeignBufferRejected", func(t *testing.T) {
		out := view(t, device, nil, 0, 2)

		source := backend.View{Buffer: foreignBuffer{}, Shape: []int{0, 2}, Strides: []int{2, 1}}
		if err := compute.Concat(out, 0, []backend.View{source}); !errors.Is(err, backend.ErrDeviceMismatch) {
			t.Fatalf("got %v", err)
		}
	})
}
