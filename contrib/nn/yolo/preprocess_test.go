package yolo

import (
	"image"
	"image/color"
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestPrepare_RGBLetterboxAndNonzeroBounds(t *testing.T) {
	source := image.NewNRGBA(image.Rect(3, 7, 7, 9))

	for y := 7; y < 9; y++ {
		for x := 3; x < 7; x++ {
			source.SetNRGBA(x, y, color.NRGBA{R: 255, G: 128, B: 64, A: 255})
		}
	}

	data, transform, err := prepare(source, 32)
	if err != nil {
		t.Fatal(err)
	}

	if transform != (letterbox{width: 4, height: 2, resizedWidth: 32, resizedHeight: 16, left: 0, top: 8}) || !slices.Equal(data.Shape(), []int{1, 3, 32, 32}) {
		t.Fatal(transform, data.Shape())
	}

	values := backendtest.HostData(t, data)
	for channel, want := range []float32{1, 128.0 / 255, 64.0 / 255} {
		if math.Abs(float64(values[(channel*32+8)*32]-want)) > 1e-6 || values[channel*32*32] != float32(114.0/255) {
			t.Fatal("RGB normalization or padding mismatch")
		}
	}
}

func TestPrepare_BilinearPixelCentersAndOddPadding(t *testing.T) {
	source := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	source.SetNRGBA(0, 0, color.NRGBA{A: 255})
	source.SetNRGBA(1, 0, color.NRGBA{R: 255, A: 255})

	data, transform, err := prepare(source, 32)
	if err != nil {
		t.Fatal(err)
	}

	row := backendtest.HostData(t, data)[transform.top*32 : (transform.top+1)*32]
	for _, test := range []struct {
		index int
		want  float32
	}{{0, 0}, {7, 0}, {8, 8.0 / 255}, {15, 120.0 / 255}, {16, 135.0 / 255}, {31, 1}} {
		if math.Abs(float64(row[test.index]-test.want)) > 1e-6 {
			t.Fatalf("column %d: got %g want %g", test.index, row[test.index], test.want)
		}
	}

	_, transform, err = prepare(image.NewNRGBA(image.Rect(0, 0, 3, 2)), 32)
	if err != nil || transform.resizedHeight != 21 || transform.top != 5 {
		t.Fatal(transform, err)
	}
}

func TestPrepare_RejectsEmptyImagesAndInvalidSizes(t *testing.T) {
	if _, _, err := prepare(nil, 32); err == nil {
		t.Fatal("nil image accepted")
	}

	for _, bounds := range []image.Rectangle{image.Rect(0, 0, 0, 1), image.Rect(0, 0, 1, 0)} {
		if _, _, err := prepare(image.NewNRGBA(bounds), 32); err == nil {
			t.Fatal("empty image accepted")
		}
	}

	for _, size := range []int{0, 31, 33, 1312} {
		if _, _, err := prepare(image.NewGray(image.Rect(0, 0, 1, 1)), size); err == nil {
			t.Fatal(size)
		}
	}
}
