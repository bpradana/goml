package backendtest

import (
	"fmt"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunPool checks forward maxima with overlapping windows.
func RunPool(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.PoolBackend)
	if !ok {
		t.Fatal("pool suite requires PoolBackend")
	}

	t.Run("Pool/OverlappingWindows", func(t *testing.T) {
		input := view(t, device, []float32{1, 2, 3, 4, 9, 6, 7, 8, 5}, 1, 1, 3, 3)

		output := view(t, device, nil, 1, 1, 2, 2)
		if err := compute.Pool(output, input, backend.Pool2DOptions{Kernel: 2, Stride: 1}); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{9, 9, 9, 9}, 0)
	})
}

// RunPoolBackward checks accumulation and deterministic finite tie selection.
func RunPoolBackward(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.PoolBackwardBackend)
	if !ok {
		t.Fatal("pool gradient suite requires PoolBackwardBackend")
	}

	t.Run("PoolBackward/OverlappingWindows", func(t *testing.T) {
		input := view(t, device, []float32{1, 2, 3, 4, 9, 6, 7, 8, 5}, 1, 1, 3, 3)
		gradient := view(t, device, []float32{1, 2, 3, 4}, 1, 1, 2, 2)

		output := view(t, device, nil, 1, 1, 3, 3)
		if err := compute.PoolBackward(output, input, gradient, backend.Pool2DOptions{Kernel: 2, Stride: 1}); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{0, 0, 0, 0, 10, 0, 0, 0, 0}, 0)
	})
	t.Run("PoolBackward/FirstTie", func(t *testing.T) {
		input := view(t, device, []float32{2, 2, 2, 2}, 1, 1, 2, 2)
		gradient := view(t, device, []float32{3}, 1, 1, 1, 1)

		output := view(t, device, nil, 1, 1, 2, 2)
		if err := compute.PoolBackward(output, input, gradient, backend.Pool2DOptions{Kernel: 2, Stride: 2}); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{3, 0, 0, 0}, 0)
	})
}

// RunResize checks nearest, bilinear, and bicubic constant preservation, plus
// a nonconstant bilinear reference with half-pixel coordinate semantics.
func RunResize(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.ResizeBackend)
	if !ok {
		t.Fatal("resize suite requires ResizeBackend")
	}

	for _, mode := range []backend.ResizeMode{backend.Nearest, backend.Bilinear, backend.Bicubic} {
		t.Run(fmt.Sprintf("Resize/Constant/%s", mode), func(t *testing.T) {
			input, output := view(t, device, []float32{3, 3, 3, 3}, 1, 1, 2, 2), view(t, device, nil, 1, 1, 4, 4)
			if err := compute.Resize(output, input, backend.ResizeOptions{Scale: 2, Mode: mode}); err != nil {
				t.Fatal(err)
			}

			want := make([]float32, 16)
			for i := range want {
				want[i] = 3
			}

			compare(t, device, output, want, 2e-5)
		})
	}

	t.Run("Resize/BilinearHalfPixel", func(t *testing.T) {
		input, output := view(t, device, []float32{0, 1, 2, 3}, 1, 1, 2, 2), view(t, device, nil, 1, 1, 4, 4)
		if err := compute.Resize(output, input, backend.ResizeOptions{Scale: 2, Mode: backend.Bilinear}); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{0, 0.25, 0.75, 1, 0.5, 0.75, 1.25, 1.5, 1.5, 1.75, 2.25, 2.5, 2, 2.25, 2.75, 3}, 2e-5)
	})
}

// RunResizeBackward checks gradient mass and symmetry for integer interpolation.
func RunResizeBackward(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.ResizeBackwardBackend)
	if !ok {
		t.Fatal("resize gradient suite requires ResizeBackwardBackend")
	}

	for _, mode := range []backend.ResizeMode{backend.Nearest, backend.Bilinear, backend.Bicubic} {
		t.Run(fmt.Sprintf("ResizeBackward/%s", mode), func(t *testing.T) {
			values := make([]float32, 16)
			for i := range values {
				values[i] = 1
			}

			gradient, output := view(t, device, values, 1, 1, 4, 4), view(t, device, nil, 1, 1, 2, 2)
			if err := compute.ResizeBackward(output, gradient, backend.ResizeOptions{Scale: 2, Mode: mode}); err != nil {
				t.Fatal(err)
			}

			compare(t, device, output, []float32{4, 4, 4, 4}, 3e-5)
		})
	}
}
