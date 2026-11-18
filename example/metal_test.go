//go:build metal && darwin && cgo

package main

import (
	"bytes"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func TestMetalCNNParityGradientsAndCleanup(t *testing.T) {
	device, err := metal.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	cpu := cnnStep(t, tensor.CPU())

	gpu := cnnStep(t, device)
	for index := range cpu {
		if math.Abs(float64(cpu[index]-gpu[index])) > 0.001 {
			t.Fatalf("CPU/Metal score %d differs: %g vs %g", index, cpu[index], gpu[index])
		}
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBytes != 0 {
		t.Fatal("Metal tensors leaked after step", stats)
	}

	config := trainConfig{Epochs: 2, BatchSize: 3, Seed: 42, TrainPerClass: 1, ValidationPerClass: 1}
	if err := fitModel(t.Context(), fixtureRecords(3), filepath.Join(t.TempDir(), "run"), config, device, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBytes != 0 {
		t.Fatal("Metal tensors leaked after Fit and validation", stats)
	}
}

func cnnStep(t *testing.T, device backend.Device) []float32 {
	t.Helper()

	model, err := newModel(42)
	if err != nil {
		t.Fatal(err)
	}
	defer closeModel(model)

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		t.Fatal(err)
	}

	before, err := model.Parameters()[0].Value().Data().HostData()
	if err != nil {
		t.Fatal(err)
	}

	records := fixtureRecords(3)

	norm, err := fitNormalization(records, []int{0, 29})
	if err != nil {
		t.Fatal(err)
	}

	dataset := &imageDataset{records: records, indices: []int{0, 29}, normalization: norm}

	samples := make([]data.ClassificationSample, 2)
	for index := range samples {
		samples[index], err = dataset.Get(t.Context(), index)
		if err != nil {
			t.Fatal(err)
		}
	}

	batch, err := data.CollateClassification(samples)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = batch.Close() }()

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = scope.Close() }()

	input, err := scope.To(batch.Inputs)
	if err != nil {
		t.Fatal(err)
	}

	value, err := nn.NewValue(input, false)
	if err != nil {
		t.Fatal(err)
	}

	logits, err := model.Forward(value)
	if err != nil {
		t.Fatal(err)
	}

	scores, err := logits.Data().HostData()
	if err != nil {
		t.Fatal(err)
	}

	objective, err := loss.NewCrossEntropy().Forward(logits, batch.Labels)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	for _, parameter := range model.Parameters() {
		if parameter.Value().Grad() == nil || parameter.Value().Grad().Device().ID() != device.ID() {
			t.Fatal("missing gradient on selected device")
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
	}

	opt, err := optimizer.NewAdam(model.Parameters(), 0.001, 0.9, 0.999, 1e-8)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = opt.Close() }()

	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}

	opt.ZeroGrad()

	after, err := model.Parameters()[0].Value().Data().HostData()
	if err != nil || slices.Equal(before, after) {
		t.Fatal("optimizer did not update convolution weights", err)
	}

	return scores
}
