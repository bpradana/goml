package transform_test

import (
	"context"
	"errors"
	"image"
	"image/color"
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/data/transform"
)

func TestImageConversion_NonzeroBoundsCHWAndHWC(t *testing.T) {
	img := image.NewNRGBA(image.Rect(3, 5, 5, 6))
	img.SetNRGBA(3, 5, color.NRGBA{R: 255, A: 255})
	img.SetNRGBA(4, 5, color.NRGBA{G: 255, A: 255})

	chw, err := transform.ImageToCHW(context.Background(), img, 3)
	if err != nil || !slices.Equal(chw.Input, []float32{1, 0, 0, 1, 0, 0}) || !slices.Equal(chw.Shape, []int{3, 1, 2}) {
		t.Fatal(chw, err)
	}

	hwc, err := transform.ToFloat32(context.Background(), img)
	if err != nil || !slices.Equal(hwc, []float32{1, 0, 0, 0, 1, 0}) {
		t.Fatal(hwc, err)
	}

	gray, err := transform.ImageToCHW(context.Background(), img, 1)
	if err != nil || gray.Input[0] != 0.299 || gray.Input[1] != 0.587 {
		t.Fatal(gray, err)
	}
}

func TestResize_HalfPixelBilinearAndCenterCrop(t *testing.T) {
	sample := data.ClassificationSample{Input: []float32{0, 1, 2, 3}, Shape: []int{1, 2, 2}, Label: 4}
	resize, _ := transform.Resize(3, 3)

	got, err := resize(context.Background(), sample)
	if err != nil || !slices.Equal(got.Input, []float32{0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3}) || got.Label != 4 {
		t.Fatal(got, err)
	}

	crop, _ := transform.CenterCrop(1, 1)

	center, err := crop(context.Background(), got)
	if err != nil || center.Input[0] != 1.5 || center.Label != 4 {
		t.Fatal(center, err)
	}

	if !slices.Equal(sample.Input, []float32{0, 1, 2, 3}) {
		t.Fatal("resize mutated sample")
	}

	tooLarge, _ := transform.CenterCrop(9, 9)
	if _, err := tooLarge(context.Background(), sample); err == nil {
		t.Fatal("oversized crop accepted")
	}

	if _, err := transform.Resize(0, 1); err == nil {
		t.Fatal("invalid resize accepted")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := resize(ctx, sample); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

type plainImage struct{ image.Image }

func TestImageConversion_Preserves16BitValuesOnFastAndFallbackPaths(t *testing.T) {
	source := image.NewRGBA64(image.Rect(0, 0, 1, 1))
	source.SetRGBA64(0, 0, color.RGBA64{R: 12345, G: 23456, B: 34567, A: 65535})

	for _, input := range []image.Image{source, plainImage{source}} {
		sample, err := transform.ImageToCHW(context.Background(), input, 3)
		if err != nil || !slices.Equal(sample.Input, []float32{float32(12345) / 65535, float32(23456) / 65535, float32(34567) / 65535}) {
			t.Fatal(sample, err)
		}
	}
}
