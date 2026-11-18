package main

import (
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/tensor"
)

func TestCNNGradientsAndPortableCheckpoint(t *testing.T) {
	model, err := newModel(42)
	if err != nil {
		t.Fatal(err)
	}
	defer closeModel(model)

	pixels := make([]float32, 2*3072)
	for index := range pixels {
		pixels[index] = float32(index%251)/125 - 1
	}

	input, err := tensor.From(pixels, 2, 3, 32, 32)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = input.Close() }()

	value, err := nn.NewValue(input, false)
	if err != nil {
		t.Fatal(err)
	}

	logits, err := model.Forward(value)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(logits.Data().Shape(), []int{2, 10}) {
		t.Fatal("wrong output shape", logits.Data().Shape())
	}

	objective, err := loss.NewCrossEntropy().Forward(logits, []int{0, 9})
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	for _, parameter := range model.Parameters() {
		if parameter.Value().Grad() == nil {
			t.Fatal("missing gradient")
		}

		gradient, err := parameter.Value().Grad().HostData()
		if err != nil {
			t.Fatal(err)
		}

		for _, value := range gradient {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("nonfinite gradient")
			}
		}

		parameter.ZeroGrad()
	}

	path := filepath.Join(t.TempDir(), "model.goml")
	if err := checkpoint.Save(path, model); err != nil {
		t.Fatal(err)
	}

	restored, err := checkpoint.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer closeModel(restored)

	a, err := model.Predict(input)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = a.Close() }()

	b, err := restored.Predict(input)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = b.Close() }()

	av, err := a.HostData()
	if err != nil {
		t.Fatal(err)
	}

	if slices.Min(av) == slices.Max(av) {
		t.Fatal("checkpoint reference logits must vary with nonzero pixels")
	}

	bv, err := b.HostData()
	if err != nil || !slices.Equal(av, bv) {
		t.Fatal("checkpoint changed predictions", err)
	}
}
