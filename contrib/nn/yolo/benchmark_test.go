package yolo

import (
	"fmt"
	"image"
	"os"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
)

func BenchmarkPrepare(b *testing.B) {
	source := image.NewNRGBA(image.Rect(0, 0, 640, 480))

	b.ReportAllocs()

	for b.Loop() {
		if _, _, err := prepare(source, 320); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNMS(b *testing.B) {
	candidates := make([]Detection, 1000)
	for index := range candidates {
		x := float32(index%50) * 8
		y := float32(index/50) * 8
		candidates[index] = Detection{Box: Box{x, y, x + 16, y + 16}, ClassID: index % 5, Confidence: 0.8}
	}

	b.ReportAllocs()

	for b.Loop() {
		if len(suppress(candidates, 0.45, 300)) == 0 {
			b.Fatal("expected detections")
		}
	}
}

func BenchmarkModelForward(b *testing.B) {
	path := os.Getenv("GOML_YOLO_MODEL")
	if path == "" {
		b.Skip("set GOML_YOLO_MODEL to benchmark a local pretrained model")
	}

	model, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}

	for _, size := range []int{320, 640} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			input, err := tensor.Ones(1, 3, size, size)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				if _, err := model.Forward(input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkModelDetect(b *testing.B) {
	path := os.Getenv("GOML_YOLO_MODEL")
	if path == "" {
		b.Skip("set GOML_YOLO_MODEL to benchmark a local pretrained model")
	}

	model, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}

	source := image.NewNRGBA(image.Rect(0, 0, 640, 480))

	for _, size := range []int{320, 640} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			options := DefaultOptions()
			options.Size = size

			b.ReportAllocs()

			for b.Loop() {
				if _, err := model.Detect(source, options); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

type benchmarkStage struct {
	layer.Layer
	name  string
	input *nn.Value
}

func (s *benchmarkStage) Forward(input *nn.Value) (*nn.Value, error) {
	s.input = input
	return s.Layer.Forward(input)
}

// Capture actual upstream activations once, then benchmark each numerical stage
// independently. These warm-input timings must not be summed as model latency.
func BenchmarkModelStages(b *testing.B) {
	path := os.Getenv("GOML_YOLO_MODEL")
	if path == "" {
		b.Skip("set GOML_YOLO_MODEL to profile local pretrained stages")
	}

	for _, size := range []int{320, 640} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			model, err := Load(path)
			if err != nil {
				b.Fatal(err)
			}

			var stages []*benchmarkStage

			for index, stage := range model.stages {
				switch index {
				case 11, 14, 17, 20:
					continue
				}

				name := fmt.Sprintf("Stage%02d", index)
				probe := &benchmarkStage{Layer: stage, name: name}
				model.stages[index] = probe
				stages = append(stages, probe)
			}

			for scale := range 3 {
				for _, head := range []struct {
					name   string
					stages *[3]*layer.Sequential
				}{{"Regression", &model.regression}, {"Classification", &model.classification}} {
					name := fmt.Sprintf("%s%d", head.name, scale)
					probe := &benchmarkStage{Layer: head.stages[scale], name: name}

					head.stages[scale], err = layer.NewSequential(probe)
					if err != nil {
						b.Fatal(err)
					}

					stages = append(stages, probe)
				}
			}

			input, _ := tensor.Ones(1, 3, size, size)
			if _, err := model.Forward(input); err != nil {
				b.Fatal(err)
			}

			for _, stage := range stages {
				b.Run(stage.name, func(b *testing.B) {
					b.ReportAllocs()

					for b.Loop() {
						if _, err := stage.Layer.Forward(stage.input); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}
