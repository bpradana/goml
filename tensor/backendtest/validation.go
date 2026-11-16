package backendtest

import (
	"math"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunFinite verifies finite-value checks, including logical strided views.
func RunFinite(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.FiniteBackend)
	if !ok {
		t.Fatal("finite suite requires FiniteBackend")
	}

	t.Run("Finite/LogicalViews", func(t *testing.T) {
		input := view(t, device, []float32{3, float32(math.NaN()), 4}, 3)

		input.Shape, input.Strides = []int{2}, []int{2}
		if err := compute.CheckFinite(input); err != nil {
			t.Fatal(err)
		}
	})

	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		t.Run("Finite/RejectNonfinite", func(t *testing.T) {
			input := view(t, device, []float32{1, value}, 2)
			if err := compute.CheckFinite(input); err == nil {
				t.Fatalf("accepted %v", value)
			}
		})
	}
}
