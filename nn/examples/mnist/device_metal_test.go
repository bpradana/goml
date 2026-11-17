//go:build metal && darwin && cgo

package main

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/trainer"
)

func TestDevice_MetalBatchesOutliveCPUCleanupAndTrainBothModels(t *testing.T) {
	for _, name := range []string{"dense", "cnn"} {
		t.Run(name, func(t *testing.T) {
			device, err := openDevice("metal")
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = device.Close() }()

			model, err := newModel(name, rand.New(rand.NewPCG(1, 2)))
			if err != nil {
				t.Fatal(err)
			}

			if err := nn.ToDevice(model.Parameters(), device); err != nil {
				t.Fatal(err)
			}

			batch, err := data.CollateClassification([]data.ClassificationSample{{Input: make([]float32, 784), Shape: []int{1, 28, 28}, Label: 1}})
			if err != nil {
				t.Fatal(err)
			}

			gpu, err := batch.Inputs.To(device)
			if err != nil {
				t.Fatal(err)
			}

			_ = batch.Close()

			values, err := gpu.HostData()
			if err != nil || len(values) != 784 {
				t.Fatal("batch transfer lost storage", err)
			}

			if _, err := data.NewClassificationDataset(gpu, []int{1}); err == nil {
				t.Fatal("implicitly downloaded a GPU dataset")
			}

			batch, err = data.CollateClassification([]data.ClassificationSample{{Input: values, Shape: []int{1, 28, 28}, Label: 1}})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = batch.Close() }()

			opt, err := optimizer.NewAdam(model.Parameters(), 0.001, 0.9, 0.999, 1e-8)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = opt.Close() }()

			dataset, err := data.NewClassificationDataset(batch.Inputs, batch.Labels)
			if err != nil {
				t.Fatal(err)
			}

			loader, err := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: 1})
			if err != nil {
				t.Fatal(err)
			}

			runner, err := trainer.NewClassifier(model, opt, device, 10)
			if err != nil {
				t.Fatal(err)
			}

			if _, err := runner.Fit(context.Background(), loader, nil, trainer.FitOptions{Epochs: 1}); err != nil {
				t.Fatal(err)
			}

			if err := device.Synchronize(); err != nil {
				t.Fatal(err)
			}

			_ = gpu.Close()
		})
	}
}

func BenchmarkBatchPreparation_MetalTransferInclusive(b *testing.B) {
	device, err := openDevice("metal")
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	samples := make([]data.ClassificationSample, 64)
	for i := range samples {
		samples[i] = data.ClassificationSample{Input: make([]float32, 784), Shape: []int{1, 28, 28}, Label: i % 10}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		batch, err := data.CollateClassification(samples)
		if err != nil {
			b.Fatal(err)
		}

		gpu, err := batch.Inputs.To(device)
		_ = batch.Close()

		if err != nil {
			b.Fatal(err)
		}

		if err := device.Synchronize(); err != nil {
			b.Fatal(err)
		}

		_ = gpu.Close()
	}
}

func TestTrainer_MetalParityConvergenceAndStableMemory(t *testing.T) {
	device, err := metal.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	checkTrainerDevice(t, device, func() (uint64, uint64) { stats := device.Stats(); return stats.LiveBytes, stats.PeakBytes })
}
