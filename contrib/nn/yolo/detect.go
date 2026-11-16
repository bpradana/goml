package yolo

import (
	"fmt"
	"image"
	"math"
	"slices"

	"github.com/bpradana/goml/tensor"
)

// Box contains pixel coordinates of the top-left and bottom-right corners.
type Box struct {
	Left   float32 `json:"left"`
	Top    float32 `json:"top"`
	Right  float32 `json:"right"`
	Bottom float32 `json:"bottom"`
}

// Detection describes one retained object in original image coordinates.
type Detection struct {
	Box        Box     `json:"box"`
	ClassID    int     `json:"class_id"`
	Label      string  `json:"label"`
	Confidence float32 `json:"confidence"`
}

// Options controls square input size, confidence filtering, and class-aware NMS.
// Size must be a multiple of 32 in [32,1280], Confidence and IoU in [0,1], and
// MaxDetections in [1,10000]. Use DefaultOptions for conventional defaults.
type Options struct {
	Size          int
	Confidence    float32
	IoU           float32
	MaxDetections int
}

// DefaultOptions returns size 640, confidence 0.25, IoU 0.45, and at most 300 boxes.
func DefaultOptions() Options {
	return Options{Size: 640, Confidence: 0.25, IoU: 0.45, MaxDetections: 300}
}

func (o Options) validate() error {
	if !validSize(o.Size) || math.IsNaN(float64(o.Confidence)) || o.Confidence < 0 || o.Confidence > 1 || math.IsNaN(float64(o.IoU)) || o.IoU < 0 || o.IoU > 1 || o.MaxDetections < 1 || o.MaxDetections > 10000 {
		return fmt.Errorf("yolo: detect: invalid size, confidence, IoU, or maximum detections")
	}

	return nil
}

// Detect allocates and runs RGB bilinear resizing, symmetric letterboxing with
// value 114, normalization by 255, inference, DFL decoding, and class-aware NMS.
// Each anchor contributes its highest-probability class; YOLOv8 has no separate
// objectness score. At most 30,000 confidence-sorted candidates enter NMS. Returned
// boxes are clipped to the original image and sorted by descending confidence;
// ties preserve anchor order. Empty results are a non-nil empty slice.
//
// Nonzero image bounds are supported. Alpha is ignored after conversion to
// unpremultiplied RGB. Resize rounds to 8-bit pixels and maps each axis using its
// actual rounded resize scale. This may differ slightly from OpenCV preprocessing.
// Nil/empty images, invalid options, and nonfinite model outputs return errors.
func (m *Model) Detect(source image.Image, options Options) ([]Detection, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}

	input, transform, err := prepare(source, options.Size)
	if err != nil {
		return nil, err
	}

	heads, err := m.Forward(input)
	if err != nil {
		return nil, err
	}

	candidates, err := decode(heads, m.classes, options)
	if err != nil {
		return nil, err
	}

	selected := suppress(candidates, options.IoU, options.MaxDetections)

	result := make([]Detection, 0, len(selected))
	for _, detection := range selected {
		box := &detection.Box
		scaleX := float32(transform.width) / float32(transform.resizedWidth)
		scaleY := float32(transform.height) / float32(transform.resizedHeight)
		box.Left = max(0, min(float32(transform.width), (box.Left-float32(transform.left))*scaleX))
		box.Right = max(0, min(float32(transform.width), (box.Right-float32(transform.left))*scaleX))
		box.Top = max(0, min(float32(transform.height), (box.Top-float32(transform.top))*scaleY))

		box.Bottom = max(0, min(float32(transform.height), (box.Bottom-float32(transform.top))*scaleY))
		if box.Right > box.Left && box.Bottom > box.Top {
			result = append(result, detection)
		}
	}

	return result, nil
}

func decode(heads [3]*tensor.Tensor, classes []string, options Options) ([]Detection, error) {
	candidates := make([]Detection, 0)

	for scale, head := range heads {
		stride := 8 << scale

		side := options.Size / stride
		if head == nil || !slices.Equal(head.Shape(), []int{1, 64 + len(classes), side, side}) {
			return nil, fmt.Errorf("yolo: decode: invalid head %d shape", scale)
		}

		values := head.Data()
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("yolo: decode: nonfinite head %d", scale)
			}
		}

		area := side * side
		for anchor := range area {
			classID, best := 0, values[64*area+anchor]
			for class := 1; class < len(classes); class++ {
				if score := values[(64+class)*area+anchor]; score > best {
					classID, best = class, score
				}
			}

			probability := float32(1 / (1 + math.Exp(-float64(best))))
			if probability <= options.Confidence {
				continue
			}

			var distances [4]float32

			for axis := range 4 {
				maximum := values[(axis*16)*area+anchor]
				for bin := 1; bin < 16; bin++ {
					maximum = max(maximum, values[(axis*16+bin)*area+anchor])
				}

				total, weighted := float64(0), float64(0)

				for bin := range 16 {
					weight := math.Exp(float64(values[(axis*16+bin)*area+anchor] - maximum))
					total += weight
					weighted += float64(bin) * weight
				}

				distances[axis] = float32(weighted/total) * float32(stride)
			}

			centerX := (float32(anchor%side) + 0.5) * float32(stride)
			centerY := (float32(anchor/side) + 0.5) * float32(stride)
			candidates = append(candidates, Detection{
				Box:     Box{Left: centerX - distances[0], Top: centerY - distances[1], Right: centerX + distances[2], Bottom: centerY + distances[3]},
				ClassID: classID, Label: classes[classID], Confidence: probability,
			})
		}
	}

	return candidates, nil
}
