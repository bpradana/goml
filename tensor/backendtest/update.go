package backendtest

import (
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunSubScaled checks fused immutable updates with Float32 product rounding.
func RunSubScaled(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.SubScaledBackend)
	if !ok {
		t.Fatal("subscaled suite requires SubScaledBackend")
	}

	t.Run("SubScaled/ImmutableValues", func(t *testing.T) {
		a, b, out := view(t, device, []float32{1, 2}, 2), view(t, device, []float32{2, 4}, 2), view(t, device, nil, 2)
		if err := compute.SubScaled(out, a, b, 0.25); err != nil {
			t.Fatal(err)
		}

		compare(t, device, out, []float32{0.5, 1}, 0)
		compare(t, device, a, []float32{1, 2}, 0)
		compare(t, device, b, []float32{2, 4}, 0)
	})
}

// RunHostAdam checks two bias-corrected steps with host-owned moment arrays.
func RunHostAdam(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.HostAdamBackend)
	if !ok {
		t.Fatal("host Adam suite requires HostAdamBackend")
	}

	t.Run("HostAdam/TwoReferenceSteps", func(t *testing.T) {
		values, gradient := view(t, device, []float32{1, -2}, 2), view(t, device, []float32{2, -4}, 2)
		first, second := make([]float32, 2), make([]float32, 2)
		options := backend.AdamOptions{Rate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, Step: 1}

		next := view(t, device, nil, 2)
		if err := compute.Adam(next, values, gradient, first, second, options); err != nil {
			t.Fatal(err)
		}

		compare(t, device, next, []float32{0.9, -1.9}, 5e-6)

		if first[0] <= 0 || first[1] >= 0 || second[0] <= 0 || second[1] <= 0 {
			t.Fatal("moments were not updated")
		}

		reversed := view(t, device, []float32{-2, 4}, 2)
		result := view(t, device, nil, 2)

		options.Step = 2
		if err := compute.Adam(result, next, reversed, first, second, options); err != nil {
			t.Fatal(err)
		}

		compare(t, device, result, []float32{0.90526316, -1.90526316}, 5e-6)
		compare(t, device, values, []float32{1, -2}, 0)
	})
}
