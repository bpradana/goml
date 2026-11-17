package webgpu

import (
	"fmt"
	"math"
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestPool_OverlappingWindows(t *testing.T) {
	backendtest.RunPool(t, openDevice(t))
}

func TestPoolBackward_AccumulationAndFirstTie(t *testing.T) {
	backendtest.RunPoolBackward(t, openDevice(t))
}

func TestResize_NearestBilinearBicubicAndHalfPixels(t *testing.T) {
	backendtest.RunResize(t, openDevice(t))
}

func TestResizeBackward_MassAndSymmetry(t *testing.T) {
	backendtest.RunResizeBackward(t, openDevice(t))
}

func TestResize_NonconstantEdgesAndGradientsMatchCPU(t *testing.T) {
	device := openDevice(t)

	for _, dimensions := range [][2]int{{1, 1}, {1, 3}, {2, 3}, {3, 2}} {
		for _, scale := range []int{1, 2, 3} {
			for _, mode := range []string{"nearest", "bilinear", "bicubic"} {
				t.Run(fmt.Sprintf("%dx%d/Scale%d/%s", dimensions[0], dimensions[1], scale, mode), func(t *testing.T) {
					shape := []int{2, 2, dimensions[0], dimensions[1]}

					values := make([]float32, 4*dimensions[0]*dimensions[1])
					for i := range values {
						values[i] = float32(i%11-5) / 7
					}

					cpu, err := tensor.From(values, shape...)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = cpu.Close() }()

					gpu, err := cpu.To(device)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = gpu.Close() }()

					want, err := tensor.Upsample2D(cpu, scale, mode)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = want.Close() }()

					got, err := tensor.Upsample2D(gpu, scale, mode)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = got.Close() }()

					compareTensors(t, got, want, 5e-5)

					seeds := make([]float32, want.Numel())
					for i := range seeds {
						seeds[i] = float32(i%13-6) / 9
					}

					gradient, err := tensor.From(seeds, want.Shape()...)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = gradient.Close() }()

					resident, err := gradient.To(device)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = resident.Close() }()

					dx, err := tensor.Upsample2DBackward(resident, shape, scale, mode)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = dx.Close() }()

					expected, err := tensor.Upsample2DBackward(gradient, shape, scale, mode)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = expected.Close() }()

					compareTensors(t, dx, expected, 8e-5)
				})
			}
		}
	}
}

func compareTensors(t *testing.T, actual, expected *tensor.Tensor, tolerance float64) {
	t.Helper()

	a, b := backendtest.HostData(t, actual), backendtest.HostData(t, expected)
	if len(a) != len(b) {
		t.Fatal("tensor lengths differ")
	}

	for i, value := range a {
		difference := math.Abs(float64(value) - float64(b[i]))
		if math.IsNaN(float64(b[i])) {
			if !math.IsNaN(float64(value)) {
				t.Fatalf("element %d: got %g, want NaN", i, value)
			}
		} else if math.IsInf(float64(b[i]), 0) {
			if value != b[i] {
				t.Fatalf("element %d: got %g, want %g", i, value, b[i])
			}
		} else if math.IsNaN(difference) || difference > tolerance+tolerance*math.Abs(float64(b[i])) {
			t.Fatalf("element %d: got %g, want %g", i, value, b[i])
		}
	}
}
