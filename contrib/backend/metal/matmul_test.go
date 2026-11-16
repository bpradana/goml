//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestMatMul_RectangularIrregularTransposedAndZeroDimensions(t *testing.T) {
	device := openDevice(t)

	for _, dimensions := range [][3]int{{2, 1, 3}, {2, 3, 5}, {17, 31, 13}, {1, 9, 1}, {3, 0, 4}, {0, 3, 4}} {
		m, k, n := dimensions[0], dimensions[1], dimensions[2]
		av, bv := make([]float32, m*k), make([]float32, n*k)

		for i := range av {
			av[i] = float32(i%11-5) / 7
		}

		for i := range bv {
			bv[i] = float32(i%7-3) / 5
		}

		a, err := tensor.From(av, m, k)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.From(bv, n, k)
		if err != nil {
			t.Fatal(err)
		}

		b, err = b.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		expected, err := tensor.MatMul(a, b)
		if err != nil {
			t.Fatal(err)
		}

		scope, err := tensor.NewScope(device)
		if err != nil {
			t.Fatal(err)
		}

		ga, err := scope.To(a)
		if err != nil {
			t.Fatal(err)
		}

		storedB, err := tensor.From(bv, n, k)
		if err != nil {
			t.Fatal(err)
		}

		gb, err := scope.To(storedB)
		if err != nil {
			t.Fatal(err)
		}

		gb, err = gb.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		output, err := tensor.MatMul(ga, gb)
		if err != nil {
			t.Fatal(err)
		}

		assertValues(t, output, backendtest.HostData(t, expected))

		if err := scope.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMatMul_Conformance(t *testing.T) {
	device := openDevice(t)
	backendtest.RunMatMul(t, device)
}
