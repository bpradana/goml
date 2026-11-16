package backendtest

import (
	"fmt"
	"math"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunReduction checks full, axis, empty, and NaN reductions.
func RunReduction(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.ReductionBackend)
	if !ok {
		t.Fatal("reduction suite requires ReductionBackend")
	}

	for _, test := range []struct {
		op          backend.ReduceOp
		axes, shape []int
		want        []float32
	}{
		{backend.Sum, nil, nil, []float32{21}},
		{backend.Mean, nil, nil, []float32{3.5}},
		{backend.Max, nil, nil, []float32{6}},
		{backend.Sum, []int{0}, []int{3}, []float32{5, 7, 9}},
		{backend.Mean, []int{1}, []int{2}, []float32{2, 5}},
		{backend.Max, []int{1}, []int{2}, []float32{3, 6}},
	} {
		t.Run(fmt.Sprintf("Reduce/%s/Axes%v", test.op, test.axes), func(t *testing.T) {
			input := view(t, device, []float32{1, 2, 3, 4, 5, 6}, 2, 3)

			output := view(t, device, nil, test.shape...)
			if err := compute.Reduce(test.op, output, input, test.axes); err != nil {
				t.Fatal(err)
			}

			compare(t, device, output, test.want, 2e-5)
		})
	}

	t.Run("Reduce/EmptySum", func(t *testing.T) {
		input, output := view(t, device, nil, 0, 3), view(t, device, nil, 3)
		if err := compute.Reduce(backend.Sum, output, input, []int{0}); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{0, 0, 0}, 0)
	})
	t.Run("Reduce/NaNPropagation", func(t *testing.T) {
		input := view(t, device, []float32{1, float32(math.NaN()), 2}, 3)
		for _, op := range []backend.ReduceOp{backend.Sum, backend.Mean, backend.Max} {
			output := view(t, device, nil)
			if err := compute.Reduce(op, output, input, nil); err != nil {
				t.Fatal(err)
			}

			compare(t, device, output, []float32{float32(math.NaN())}, 0)
		}
	})
}
