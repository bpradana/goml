package transform

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/tensor/backend"
)

func validateImage(sample data.ClassificationSample) error {
	count, err := (backend.View{Shape: sample.Shape}).Numel()
	if err != nil || len(sample.Shape) != 3 || count != len(sample.Input) || count == 0 {
		return fmt.Errorf("transform: expected a nonempty CHW image, got %v with %d values", sample.Shape, len(sample.Input))
	}

	return nil
}

func imageElements(height, width, channels int) (int, error) {
	if height <= 0 || width <= 0 || channels <= 0 || height > int(^uint(0)>>1)/4/channels/width {
		return 0, fmt.Errorf("transform: invalid or overflowing image dimensions")
	}

	return height * width * channels, nil
}

// ToFloat32 converts an image to interleaved RGB HWC values in [0,1]. Alpha is
// composited onto black using image.Color.RGBA's premultiplied components.
// Bounds origins may be nonzero. It allocates an independent slice.
func ToFloat32(ctx context.Context, source image.Image) ([]float32, error) {
	return imageValues(ctx, source, 3, false)
}

// ImageToCHW converts an image to independent [channels,height,width] values in
// [0,1], with channels 3 (RGB) or 1 (BT.601 grayscale). Label starts at zero.
// Alpha is composited onto black. Conversion does not resize the image.
func ImageToCHW(ctx context.Context, source image.Image, channels int) (data.ClassificationSample, error) {
	values, err := imageValues(ctx, source, channels, true)
	if err != nil {
		return data.ClassificationSample{}, err
	}

	bounds := source.Bounds()

	return data.ClassificationSample{Input: values, Shape: []int{channels, bounds.Dy(), bounds.Dx()}}, nil
}

func imageValues(ctx context.Context, source image.Image, channels int, chw bool) ([]float32, error) {
	if source == nil || (channels != 1 && channels != 3) {
		return nil, fmt.Errorf("transform: image and channel count 1 or 3 required")
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	bounds := source.Bounds()

	n, err := imageElements(bounds.Dy(), bounds.Dx(), channels)
	if err != nil {
		return nil, err
	}

	values := make([]float32, n)

	area := bounds.Dx() * bounds.Dy()

	readPixel := func(x, y int) color.RGBA64 {
		r, g, b, a := source.At(x, y).RGBA()
		return color.RGBA64{R: uint16(r), G: uint16(g), B: uint16(b), A: uint16(a)}
	}
	if source, ok := source.(image.RGBA64Image); ok {
		// Standard decoders avoid boxing a Color interface for each pixel.
		// Both paths preserve 16-bit component precision.
		readPixel = source.RGBA64At
	}

	for y := range bounds.Dy() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		for x := range bounds.Dx() {
			pixel := readPixel(bounds.Min.X+x, bounds.Min.Y+y)
			rgb := [3]float32{float32(pixel.R) / 65535, float32(pixel.G) / 65535, float32(pixel.B) / 65535}

			index := y*bounds.Dx() + x
			if channels == 1 {
				values[index] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
				continue
			}

			for channel, value := range rgb {
				if chw {
					values[channel*area+index] = value
				} else {
					values[index*3+channel] = value
				}
			}
		}
	}

	return values, nil
}

// Resize returns a copying CHW bilinear resize using half-pixel coordinates and
// clamped edge pixels. It preserves labels and does not change channel count.
func Resize(height, width int) (Func[data.ClassificationSample], error) {
	if _, err := imageElements(height, width, 1); err != nil {
		return nil, err
	}

	return func(ctx context.Context, sample data.ClassificationSample) (data.ClassificationSample, error) {
		if err := ctx.Err(); err != nil {
			return data.ClassificationSample{}, err
		}

		if err := validateImage(sample); err != nil {
			return data.ClassificationSample{}, err
		}

		channels, ih, iw := sample.Shape[0], sample.Shape[1], sample.Shape[2]

		n, err := imageElements(height, width, channels)
		if err != nil {
			return data.ClassificationSample{}, err
		}

		output := make([]float32, n)

		for y := range height {
			if err := ctx.Err(); err != nil {
				return data.ClassificationSample{}, err
			}

			sy := (float64(y)+0.5)*float64(ih)/float64(height) - 0.5
			y0 := int(math.Floor(sy))
			fy := float32(sy - float64(y0))

			for x := range width {
				sx := (float64(x)+0.5)*float64(iw)/float64(width) - 0.5
				x0 := int(math.Floor(sx))
				fx := float32(sx - float64(x0))

				for channel := range channels {
					base := channel * ih * iw
					a := sample.Input[base+max(0, min(ih-1, y0))*iw+max(0, min(iw-1, x0))]
					b := sample.Input[base+max(0, min(ih-1, y0))*iw+max(0, min(iw-1, x0+1))]
					c := sample.Input[base+max(0, min(ih-1, y0+1))*iw+max(0, min(iw-1, x0))]
					d := sample.Input[base+max(0, min(ih-1, y0+1))*iw+max(0, min(iw-1, x0+1))]
					output[(channel*height+y)*width+x] = (a*(1-fx)+b*fx)*(1-fy) + (c*(1-fx)+d*fx)*fy
				}
			}
		}

		return data.ClassificationSample{Input: output, Shape: []int{channels, height, width}, Label: sample.Label}, nil
	}, nil
}

// CenterCrop copies a centered CHW rectangle. Oversized crops return errors;
// there is no implicit padding. Odd excess pixels remain on the bottom/right.
func CenterCrop(height, width int) (Func[data.ClassificationSample], error) {
	if _, err := imageElements(height, width, 1); err != nil {
		return nil, err
	}

	return func(ctx context.Context, sample data.ClassificationSample) (data.ClassificationSample, error) {
		if err := ctx.Err(); err != nil {
			return data.ClassificationSample{}, err
		}

		if err := validateImage(sample); err != nil {
			return data.ClassificationSample{}, err
		}

		channels, ih, iw := sample.Shape[0], sample.Shape[1], sample.Shape[2]
		if height > ih || width > iw {
			return data.ClassificationSample{}, fmt.Errorf("transform: crop %dx%d exceeds image %dx%d", height, width, ih, iw)
		}

		output := make([]float32, channels*height*width)
		for channel := range channels {
			for y := range height {
				if err := ctx.Err(); err != nil {
					return data.ClassificationSample{}, err
				}

				start := (channel*ih+(ih-height)/2+y)*iw + (iw-width)/2
				copy(output[(channel*height+y)*width:], sample.Input[start:start+width])
			}
		}

		return data.ClassificationSample{Input: output, Shape: []int{channels, height, width}, Label: sample.Label}, nil
	}, nil
}
