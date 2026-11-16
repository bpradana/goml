package yolo

import (
	"image"
	"os"
	"testing"

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

	input, err := tensor.Ones(1, 3, 320, 320)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()

	for b.Loop() {
		if _, err := model.Forward(input); err != nil {
			b.Fatal(err)
		}
	}
}
