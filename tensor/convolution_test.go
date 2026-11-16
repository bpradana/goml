package tensor

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestIm2Col_ValuesAndOverlappingScatter(t *testing.T) {
	input := fromTest(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)

	patches, err := Im2Col(input, 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, patches, []int{4, 4}, []float32{1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9})

	ones, _ := Ones(4, 4)

	gradient, err := Col2Im(ones, input.Shape(), 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, gradient, input.Shape(), []float32{1, 2, 1, 2, 4, 2, 1, 2, 1})
	assertData(t, input, input.Shape(), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
}

func TestCol2Im_StridedOffsetPatchesAndEmptyBatch(t *testing.T) {
	storage, _ := Randn(rand.New(rand.NewPCG(9, 10)), 9, 33)
	patches, _ := storage.Transpose()
	patches, _ = patches.Slice(0, 1, 33)
	patches, _ = patches.Slice(1, 1, 9)
	shape := []int{2, 2, 3, 3}

	got, err := Col2Im(patches, shape, 2, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	want, err := Col2Im(contiguous(t, patches), shape, 2, 1, 1)
	if err != nil || !slices.Equal(hostData(t, got), hostData(t, want)) {
		t.Fatal("strided scatter differs", got, want, err)
	}

	empty, _ := Zeros(0, 8)

	got, err = Col2Im(empty, []int{0, 2, 3, 3}, 2, 1, 1)
	if err != nil || got.Numel() != 0 {
		t.Fatal("empty scatter", got, err)
	}
}

func TestConv2D_TiledAndStridedMatchDirectReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))

	for _, configuration := range [][5]int{{1, 3, 1, 0, 5}, {2, 3, 2, 1, 9}, {2, 1, 1, 0, 17}, {0, 3, 1, 1, 4}} {
		batch, kernel, stride, padding, size := configuration[0], configuration[1], configuration[2], configuration[3], configuration[4]
		input, _ := Randn(rng, batch, 2, size+1, size+1)
		input, _ = input.Transpose(0, 1, 3, 2)
		input, _ = input.Slice(2, 1, size+1)
		input, _ = input.Slice(3, 1, size+1)
		weight, _ := Randn(rng, 5, 2*kernel*kernel)
		weight, _ = weight.Transpose()
		bias, _ := Randn(rng, 5)

		for _, bias := range []*Tensor{nil, bias} {
			got, err := Conv2D(input, weight, bias, kernel, stride, padding)
			if err != nil {
				t.Fatal(err)
			}

			outSize := (size+2*padding-kernel)/stride + 1
			if !slices.Equal(got.Shape(), []int{batch, 5, outSize, outSize}) || !got.IsContiguous() {
				t.Fatal(got)
			}

			for b := range batch {
				for oc := range 5 {
					for y := range outSize {
						for x := range outSize {
							var want float64

							for ic := range 2 {
								for ky := range kernel {
									for kx := range kernel {
										iy, ix := y*stride-padding+ky, x*stride-padding+kx
										if iy >= 0 && iy < size && ix >= 0 && ix < size {
											v, _ := input.At(b, ic, iy, ix)
											w, _ := weight.At((ic*kernel+ky)*kernel+kx, oc)
											want += float64(v) * float64(w)
										}
									}
								}
							}

							if bias != nil {
								v, _ := bias.At(oc)
								want += float64(v)
							}

							v, _ := got.At(b, oc, y, x)
							if math.Abs(float64(v)-want) > 1e-5 {
								t.Fatalf("configuration %v [%d,%d,%d,%d]: got %g want %g", configuration, b, oc, y, x, v, want)
							}
						}
					}
				}
			}
		}
	}
}

func TestConvolution_RejectsInvalidShapesAndConfiguration(t *testing.T) {
	input, _ := Ones(1, 2, 3, 3)
	weight, _ := Ones(18, 4)
	bias, _ := Ones(3)

	for _, configuration := range [][3]int{{0, 1, 0}, {3, 0, 0}, {3, 1, -1}, {5, 1, 0}, {3, 1, int(^uint(0) >> 1)}} {
		if _, err := Im2Col(input, configuration[0], configuration[1], configuration[2]); err == nil {
			t.Fatal(configuration)
		}
	}

	for _, invalid := range []*Tensor{nil, {}, Scalar(1), weight} {
		if _, err := Im2Col(invalid, 3, 1, 1); err == nil {
			t.Fatal("accepted invalid input")
		}
	}

	if _, err := Conv2D(input, weight, bias, 3, 1, 1); err == nil {
		t.Fatal("accepted wrong bias")
	}

	if _, err := Conv2D(input, weight, nil, 1, 1, 0); err == nil {
		t.Fatal("accepted wrong weight")
	}

	if _, err := Col2Im(weight, input.Shape(), 3, 1, 1); err == nil {
		t.Fatal("accepted wrong patch matrix")
	}

	if _, err := Col2Im(weight, []int{1, -1, 3, 3}, 3, 1, 1); err == nil {
		t.Fatal("accepted negative shape")
	}
}
