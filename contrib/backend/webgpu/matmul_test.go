package webgpu

import (
	"fmt"
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestMatMul_RectangularIrregularEmptyAndStrided(t *testing.T) {
	backendtest.RunMatMul(t, openDevice(t))
}

func BenchmarkMatMul(b *testing.B) {
	device := openDevice(b)

	for _, shape := range [][3]int{{8, 8, 8}, {31, 17, 9}, {128, 128, 128}, {256, 64, 128}, {512, 512, 512}} {
		m, k, n := shape[0], shape[1], shape[2]
		b.Run(fmt.Sprintf("%dx%dx%d", m, k, n), func(b *testing.B) {
			a, err := tensor.Ones(m, k)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = a.Close() }()

			other, err := tensor.Ones(k, n)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = other.Close() }()

			residentA, err := a.To(device)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = residentA.Close() }()

			residentB, err := other.To(device)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = residentB.Close() }()

			b.Run("cpu", func(b *testing.B) {
				b.ReportAllocs()

				for b.Loop() {
					result, err := tensor.MatMul(a, other)
					if err != nil {
						b.Fatal(err)
					}

					if err := result.Close(); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("resident", func(b *testing.B) {
				b.ReportAllocs()

				for b.Loop() {
					result, err := tensor.MatMul(residentA, residentB)
					if err != nil {
						b.Fatal(err)
					}

					if err := device.Synchronize(); err != nil {
						b.Fatal(err)
					}

					if err := result.Close(); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("transfers", func(b *testing.B) {
				b.ReportAllocs()

				for b.Loop() {
					input, err := a.To(device)
					if err != nil {
						b.Fatal(err)
					}

					weight, err := other.To(device)
					if err != nil {
						b.Fatal(err)
					}

					result, err := tensor.MatMul(input, weight)
					if err != nil {
						b.Fatal(err)
					}

					if _, err := result.HostData(); err != nil {
						b.Fatal(err)
					}

					for _, value := range []*tensor.Tensor{result, input, weight} {
						if err := value.Close(); err != nil {
							b.Fatal(err)
						}
					}
				}
			})
		})
	}
}
