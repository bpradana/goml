//go:build darwin && cgo

package metal_test

import (
	"fmt"
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"
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

func BenchmarkMatMul(b *testing.B) {
	device, err := metal.Open()
	if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() { _ = device.Close() })

	for _, dimensions := range [][3]int{{8, 8, 8}, {64, 64, 64}, {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}, {127, 65, 193}, {512, 64, 96}, {64, 512, 96}} {
		m, k, n := dimensions[0], dimensions[1], dimensions[2]

		left, err := tensor.Ones(m, k)
		if err != nil {
			b.Fatal(err)
		}

		right, err := tensor.Ones(k, n)
		if err != nil {
			b.Fatal(err)
		}

		b.Run(fmt.Sprintf("%dx%dx%d", m, k, n), func(b *testing.B) {
			for _, mode := range []string{"cpu", "metal_resident", "metal_transfer"} {
				b.Run(mode, func(b *testing.B) {
					a, c := left, right
					if mode == "metal_resident" {
						a, err = left.To(device)
						if err != nil {
							b.Fatal(err)
						}
						defer func() { _ = a.Close() }()

						c, err = right.To(device)
						if err != nil {
							b.Fatal(err)
						}
						defer func() { _ = c.Close() }()
					}

					if mode != "cpu" {
						warmLeft, err := left.To(device)
						if err != nil {
							b.Fatal(err)
						}

						warmRight, err := right.To(device)
						if err != nil {
							b.Fatal(err)
						}

						warmResult, err := tensor.MatMul(warmLeft, warmRight)
						if err != nil {
							b.Fatal(err)
						}

						if err := device.Synchronize(); err != nil {
							b.Fatal(err)
						}

						_ = warmLeft.Close()
						_ = warmRight.Close()
						_ = warmResult.Close()
					}

					b.ReportAllocs()
					b.ResetTimer()

					for range b.N {
						if mode == "metal_transfer" {
							a, err = left.To(device)
							if err != nil {
								b.Fatal(err)
							}

							c, err = right.To(device)
							if err != nil {
								b.Fatal(err)
							}
						}

						result, err := tensor.MatMul(a, c)
						if err != nil {
							b.Fatal(err)
						}

						switch mode {
						case "metal_transfer":
							if _, err := result.HostData(); err != nil {
								b.Fatal(err)
							}

							_ = a.Close()
							_ = c.Close()
						case "metal_resident":
							if err := device.Synchronize(); err != nil {
								b.Fatal(err)
							}
						}

						_ = result.Close()
					}

					b.StopTimer()
					b.ReportMetric(float64(b.N)*2*float64(m)*float64(k)*float64(n)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
				})
			}
		})
	}
}
