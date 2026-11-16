package tensor

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/bpradana/goml/simdops"
)

func BenchmarkMatMul(b *testing.B) {
	for _, dimensions := range [][3]int{{4, 4, 4}, {32, 32, 32}, {128, 16, 32}, {16, 128, 32}, {31, 29, 37}, {128, 128, 128}, {256, 256, 256}, {512, 512, 512}, {1024, 64, 128}, {64, 512, 256}} {
		b.Run(fmt.Sprint(dimensions), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))

			a, err := Rand(rng, dimensions[0], dimensions[1])
			if err != nil {
				b.Fatal(err)
			}

			c, err := Rand(rng, dimensions[1], dimensions[2])
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				if _, err := MatMul(a, c); err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(2*float64(dimensions[0])*float64(dimensions[1])*float64(dimensions[2])*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkMatMulLayouts(b *testing.B) {
	for _, layout := range []string{"TransposeA", "TransposeB", "Both", "Sliced"} {
		b.Run(layout, func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			a, _ := Rand(rng, 129, 129)
			c, _ := Rand(rng, 129, 129)

			switch layout {
			case "TransposeA":
				a, _ = a.Transpose()
			case "TransposeB":
				c, _ = c.Transpose()
			case "Both":
				a, _ = a.Transpose()
				c, _ = c.Transpose()
			case "Sliced":
				a, _ = a.Slice(1, 1, 129)
				c, _ = c.Slice(0, 1, 129)
			}

			b.ReportAllocs()

			for b.Loop() {
				if _, err := MatMul(a, c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	for _, layout := range []string{"Contiguous", "Transposed", "Broadcast", "Column", "Scalar", "Sliced"} {
		b.Run(layout, func(b *testing.B) {
			a, err := Ones(64, 64)
			if err != nil {
				b.Fatal(err)
			}

			c := a

			switch layout {
			case "Transposed":
				a, err = a.Transpose()
			case "Broadcast":
				c, err = Ones(64)
			case "Column":
				c, err = Ones(64, 1)
			case "Scalar":
				c = Scalar(2)
			case "Sliced":
				a, err = a.Slice(1, 1, 63)
				c = a
			}

			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				if _, err := Add(a, c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkData(b *testing.B) {
	for _, layout := range []string{"Contiguous", "Transposed"} {
		b.Run(layout, func(b *testing.B) {
			a, _ := Ones(256, 256)
			if layout == "Transposed" {
				a, _ = a.Transpose()
			}

			b.ReportAllocs()
			b.SetBytes(int64(a.Numel() * 4))

			for b.Loop() {
				_ = a.Data()
			}
		})
	}
}

func BenchmarkSum(b *testing.B) {
	a, err := Ones(64, 64)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()

	for b.Loop() {
		if _, err := Sum(a, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulAxpy(b *testing.B) {
	for _, size := range []int{32, 128, 512} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			a, _ := Randn(rng, size, size)
			c, _ := Randn(rng, size, size)

			b.ReportAllocs()

			for b.Loop() {
				out, err := allocated([]int{size, size})
				if err != nil {
					b.Fatal(err)
				}

				for i := range size {
					row := out.data[i*size : (i+1)*size]
					for p := range size {
						if err := simdops.Axpy(row, c.data[p*size:(p+1)*size], row, a.data[i*size+p]); err != nil {
							b.Fatal(err)
						}
					}
				}
			}
		})
	}
}

// directConvolution is a test-only alternative to patch extraction, retaining
// increasing feature order for each output element.
func directConvolution(input, weight *Tensor, kernel, stride, padding int) (*Tensor, error) {
	outH, outW, _, err := convolutionShape(input.shape, kernel, stride, padding)
	if err != nil {
		return nil, err
	}

	out, err := allocated([]int{input.shape[0], weight.shape[1], outH, outW})
	if err != nil {
		return nil, err
	}

	for batch := range input.shape[0] {
		for channel := range weight.shape[1] {
			for y := range outH {
				for x := range outW {
					var sum float32

					for ic := range input.shape[1] {
						for ky := range kernel {
							for kx := range kernel {
								iy, ix := y*stride-padding+ky, x*stride-padding+kx

								var value float32
								if iy >= 0 && iy < input.shape[2] && ix >= 0 && ix < input.shape[3] {
									value = input.data[input.offset+batch*input.strides[0]+ic*input.strides[1]+iy*input.strides[2]+ix*input.strides[3]]
								}

								feature := (ic*kernel+ky)*kernel + kx
								sum += value * weight.data[weight.offset+feature*weight.strides[0]+channel*weight.strides[1]]
							}
						}
					}

					out.data[((batch*weight.shape[1]+channel)*outH+y)*outW+x] = sum
				}
			}
		}
	}

	return out, nil
}

func BenchmarkConvolutionStrategies(b *testing.B) {
	for _, configuration := range []struct {
		name                    string
		batch, in, out, size    int
		kernel, stride, padding int
	}{
		{"Small", 4, 1, 4, 8, 3, 1, 1},
		{"CNN", 4, 16, 32, 32, 3, 1, 1},
		{"Pointwise", 1, 64, 64, 40, 1, 1, 0},
		{"YOLO320", 1, 3, 16, 320, 3, 2, 1},
	} {
		for _, strategy := range []string{"Direct", "Full", "Tiled"} {
			b.Run(configuration.name+"/"+strategy, func(b *testing.B) {
				rng := rand.New(rand.NewPCG(1, 2))
				input, _ := Randn(rng, configuration.batch, configuration.in, configuration.size, configuration.size)
				weight, _ := Randn(rng, configuration.in*configuration.kernel*configuration.kernel, configuration.out)
				outH, outW, _, _ := convolutionShape(input.shape, configuration.kernel, configuration.stride, configuration.padding)

				b.ReportAllocs()

				for b.Loop() {
					var err error

					switch strategy {
					case "Direct":
						_, err = directConvolution(input, weight, configuration.kernel, configuration.stride, configuration.padding)
					case "Tiled":
						_, err = Conv2D(input, weight, nil, configuration.kernel, configuration.stride, configuration.padding)
					case "Full":
						var patches, product *Tensor

						patches, err = Im2Col(input, configuration.kernel, configuration.stride, configuration.padding)
						if err == nil {
							product, err = MatMul(patches, weight)
						}

						if err == nil {
							product, err = product.Reshape(configuration.batch, outH, outW, configuration.out)
						}

						if err == nil {
							product, err = product.Transpose(0, 3, 1, 2)
						}

						if err == nil {
							_ = product.Contiguous()
						}
					}

					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
