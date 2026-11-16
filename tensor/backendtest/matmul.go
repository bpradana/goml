package backendtest

import (
	"errors"
	"fmt"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunMatMul compares rectangular, irregular, empty, and strided products against
// an independent Float32 scalar reference. MatMul capability is required.
// Optional shapes select hardware-supported contiguous cases; defaults cover
// rectangular, irregular, and empty products. Unsupported strided input is reported.
func RunMatMul(t *testing.T, device backend.Device, shapes ...[3]int) {
	t.Helper()

	compute, ok := device.(backend.MatMulBackend)
	if !ok {
		t.Fatal("MatMul suite requires MatMulBackend")
	}

	if len(shapes) == 0 {
		shapes = [][3]int{{2, 3, 4}, {31, 17, 9}, {65, 33, 67}, {0, 3, 4}, {3, 0, 4}, {3, 4, 0}}
	}

	for _, dimensions := range shapes {
		m, k, n := dimensions[0], dimensions[1], dimensions[2]
		t.Run(fmt.Sprintf("MatMul/%dx%dx%d", m, k, n), func(t *testing.T) {
			av, bv := make([]float32, m*k), make([]float32, k*n)
			for i := range av {
				av[i] = float32(i%11-5) / 8
			}

			for i := range bv {
				bv[i] = float32(i%7-3) / 8
			}

			a, b, out := view(t, device, av, m, k), view(t, device, bv, k, n), view(t, device, nil, m, n)

			want := make([]float32, m*n)
			for row := range m {
				for column := range n {
					for inner := range k {
						want[row*n+column] += av[row*k+inner] * bv[inner*n+column]
					}
				}
			}

			if err := compute.MatMul(out, a, b); err != nil {
				t.Fatal(err)
			}

			compare(t, device, out, want, 2e-4)
		})
	}

	t.Run("MatMul/TransposedInput", func(t *testing.T) {
		a := view(t, device, []float32{1, 2, 3, 4, 5, 6}, 3, 2)
		a.Shape, a.Strides = []int{2, 3}, []int{1, 2}
		b := view(t, device, []float32{1, 0, 0, 1, 1, 1}, 3, 2)

		out := view(t, device, nil, 2, 2)
		if err := compute.MatMul(out, a, b); errors.Is(err, backend.ErrUnsupported) {
			t.Skipf("strided inputs are explicitly unsupported: %v", err)
		} else if err != nil {
			t.Fatal(err)
		}

		compare(t, device, out, []float32{6, 8, 8, 10}, 1e-5)
	})
}
