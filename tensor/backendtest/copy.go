package backendtest

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunCopy checks strided materialization, scatter, physical bounds, overlapping
// output rejection, and retention of source storage after queued copies.
func RunCopy(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.CopyBackend)
	if !ok {
		t.Fatal("copy suite requires CopyBackend")
	}

	t.Run("Copy/ForeignNoncomparableBuffer", func(t *testing.T) {
		foreign := backend.View{Buffer: foreignBuffer{}, Shape: []int{1}, Strides: []int{1}}
		if err := compute.Copy(foreign, foreign); !errors.Is(err, backend.ErrDeviceMismatch) {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("Copy/Transpose", func(t *testing.T) {
		source := view(t, device, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
		source.Shape, source.Strides = []int{3, 2}, []int{1, 3}

		output := view(t, device, nil, 3, 2)
		if err := compute.Copy(output, source); err != nil {
			t.Fatal(err)
		}

		if err := source.Buffer.Close(); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{1, 4, 2, 5, 3, 6}, 0)
	})
	t.Run("Copy/Scatter", func(t *testing.T) {
		source := view(t, device, []float32{1, 2, 3, 4}, 2, 2)
		output := view(t, device, []float32{0, 0, 0, 0, 0, 0}, 2, 3)
		target := output

		target.Shape, target.Offset = []int{2, 2}, 1
		if err := compute.Copy(target, source); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{0, 1, 2, 0, 3, 4}, 0)
	})
	t.Run("Copy/InvalidBoundsAndOverlappingWrites", func(t *testing.T) {
		source, output := view(t, device, []float32{1, 2, 3, 4}, 2, 2), view(t, device, nil, 2, 2)
		invalid := output

		invalid.Strides = []int{8, 1}
		if err := compute.Copy(invalid, source); !errors.Is(err, backend.ErrInvalidView) {
			t.Fatalf("bounds: %v", err)
		}

		invalid.Strides = []int{0, 1}
		if err := compute.Copy(invalid, source); !errors.Is(err, backend.ErrInvalidView) {
			t.Fatalf("overlap: %v", err)
		}
	})
}

// RunFill checks complete contiguous fills and empty initialization.
func RunFill(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.FillBackend)
	if !ok {
		t.Fatal("fill suite requires FillBackend")
	}

	t.Run("Fill/Values", func(t *testing.T) {
		output := view(t, device, nil, 2, 3)
		if err := compute.Fill(output, -2); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{-2, -2, -2, -2, -2, -2}, 0)
	})
	t.Run("Fill/Empty", func(t *testing.T) {
		output := view(t, device, nil, 0, 3)
		if err := compute.Fill(output, 4); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, nil, 0)
	})
}
