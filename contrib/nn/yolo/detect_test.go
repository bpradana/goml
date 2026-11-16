package yolo

import (
	"image"
	"math"
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestPostprocess_RestoresCoordinatesAndRejectsInvalidMetadata(t *testing.T) {
	model := &Model{classes: []string{"object"}}
	options := DefaultOptions()
	options.Size = 32
	heads := emptyHeads(t, 1, 32)
	values := heads[0].Data()
	values[64*16] = 100
	heads[0], _ = tensor.From(values, 1, 65, 4, 4)

	detections, err := model.Postprocess(heads, image.Pt(64, 32), options)
	if err != nil {
		t.Fatal(err)
	}

	if len(detections) != 1 || detections[0].Box != (Box{0, 0, 64, 32}) {
		t.Fatal("letterbox coordinates were not restored", detections)
	}

	if _, err := model.Postprocess(heads, image.Pt(0, 32), options); err == nil {
		t.Fatal("invalid original dimensions accepted")
	}

	if _, err := (*Model)(nil).Postprocess(heads, image.Pt(64, 32), options); err == nil {
		t.Fatal("uninitialized model accepted")
	}
}

func emptyHeads(t *testing.T, classes, size int) [3]*tensor.Tensor {
	t.Helper()

	var heads [3]*tensor.Tensor
	for scale := range heads {
		side := size / (8 << scale)

		values := make([]float32, (64+classes)*side*side)
		for i := 64 * side * side; i < len(values); i++ {
			values[i] = -100
		}

		var err error

		heads[scale], err = tensor.From(values, 1, 64+classes, side, side)
		if err != nil {
			t.Fatal(err)
		}
	}

	return heads
}

func TestDecode_UniformDistributionAndHighestClassProbability(t *testing.T) {
	options := DefaultOptions()
	options.Size = 32
	heads := emptyHeads(t, 2, 32)
	values := heads[0].Data()
	values[64*16] = 0
	values[65*16] = float32(math.Log(3))
	heads[0], _ = tensor.From(values, 1, 66, 4, 4)

	got, err := decode(heads, []string{"first", "second"}, options)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != 1 || got[0].ClassID != 1 || got[0].Label != "second" || math.Abs(float64(got[0].Confidence-0.75)) > 1e-6 || got[0].Box != (Box{-56, -56, 64, 64}) {
		t.Fatal(got)
	}
}

func TestDecode_StableDFLForLargeLogitsAndAllStrides(t *testing.T) {
	options := DefaultOptions()
	options.Size = 32

	heads := emptyHeads(t, 1, 32)
	for scale, head := range heads {
		side := 32 / (8 << scale)
		area := side * side
		values := head.Data()
		values[64*area] = 1000

		for axis := range 4 {
			for bin := range 16 {
				values[(axis*16+bin)*area] = -1000
			}

			values[(axis*16+2)*area] = 1000
		}

		heads[scale], _ = tensor.From(values, 1, 65, side, side)
	}

	detections, err := decode(heads, []string{"object"}, options)
	if err != nil || len(detections) != 3 {
		t.Fatal(detections, err)
	}

	for scale, got := range detections {
		stride := float32(int(8) << scale)
		if got.Box != (Box{-1.5 * stride, -1.5 * stride, 2.5 * stride, 2.5 * stride}) || got.Confidence != 1 {
			t.Fatal(scale, got)
		}
	}
}

func TestDecode_RejectsMalformedAndNonfiniteHeads(t *testing.T) {
	options := DefaultOptions()
	options.Size = 32
	heads := emptyHeads(t, 1, 32)

	heads[1] = nil
	if _, err := decode(heads, []string{"object"}, options); err == nil {
		t.Fatal("nil head accepted")
	}

	heads = emptyHeads(t, 1, 32)
	values := heads[0].Data()
	values[0] = float32(math.NaN())

	heads[0], _ = tensor.From(values, 1, 65, 4, 4)
	if _, err := decode(heads, []string{"object"}, options); err == nil {
		t.Fatal("NaN head accepted")
	}
}

func TestOptions_RejectsInvalidValues(t *testing.T) {
	defaults := DefaultOptions()

	for _, options := range []Options{
		{},
		{Size: 33, Confidence: 0.25, IoU: 0.45, MaxDetections: 300},
		{Size: 32, Confidence: -1, IoU: 0.45, MaxDetections: 300},
		{Size: 32, Confidence: float32(math.NaN()), IoU: 0.45, MaxDetections: 300},
		{Size: 32, Confidence: 0.25, IoU: 2, MaxDetections: 300},
		{Size: 32, Confidence: 0.25, IoU: 0.45, MaxDetections: 0},
	} {
		if options.validate() == nil {
			t.Fatal(options)
		}
	}

	if err := defaults.validate(); err != nil {
		t.Fatal(err)
	}
}
