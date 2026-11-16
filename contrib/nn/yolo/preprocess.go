package yolo

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/bpradana/goml/tensor"
)

type letterbox struct {
	width, height               int
	resizedWidth, resizedHeight int
	left, top                   int
}

func prepare(source image.Image, size int) (*tensor.Tensor, letterbox, error) {
	var transform letterbox
	if source == nil || !validSize(size) {
		return nil, transform, fmt.Errorf("yolo: preprocess: image and size multiple of 32 in [32,1280] required")
	}

	bounds := source.Bounds()

	width, height := bounds.Dx(), bounds.Dy()
	if width <= 0 || height <= 0 || width > 65536 || height > 65536 {
		return nil, transform, fmt.Errorf("yolo: preprocess: image dimensions must be in [1,65536]")
	}

	ratio := math.Min(float64(size)/float64(width), float64(size)/float64(height))
	resizedWidth := max(1, int(math.Round(float64(width)*ratio)))
	resizedHeight := max(1, int(math.Round(float64(height)*ratio)))
	transform = letterbox{width: width, height: height, resizedWidth: resizedWidth, resizedHeight: resizedHeight, left: (size - resizedWidth) / 2, top: (size - resizedHeight) / 2}

	values := make([]float32, 3*size*size)
	for i := range values {
		values[i] = 114.0 / 255
	}

	for row := range resizedHeight {
		y := (float64(row)+0.5)*float64(height)/float64(resizedHeight) - 0.5
		baseY := int(math.Floor(y))
		fy := y - float64(baseY)

		for column := range resizedWidth {
			x := (float64(column)+0.5)*float64(width)/float64(resizedWidth) - 0.5
			baseX := int(math.Floor(x))
			fx := x - float64(baseX)

			var channels [3]float64

			for dy := range 2 {
				for dx := range 2 {
					sx, sy := max(0, min(width-1, baseX+dx)), max(0, min(height-1, baseY+dy))
					pixel := color.NRGBAModel.Convert(source.At(bounds.Min.X+sx, bounds.Min.Y+sy)).(color.NRGBA)

					weightX, weightY := 1-fx, 1-fy
					if dx == 1 {
						weightX = fx
					}

					if dy == 1 {
						weightY = fy
					}

					weight := weightX * weightY
					channels[0] += weight * float64(pixel.R)
					channels[1] += weight * float64(pixel.G)
					channels[2] += weight * float64(pixel.B)
				}
			}

			for channel, value := range channels {
				values[(channel*size+row+transform.top)*size+column+transform.left] = float32(math.Round(value)) / 255
			}
		}
	}

	data, err := tensor.From(values, 1, 3, size, size)

	return data, transform, err
}
