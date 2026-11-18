package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

func TestTrainingWritesReloadableRunAndPredictionCLI(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "run")
	records := fixtureRecords(3)
	config := trainConfig{Epochs: 1, BatchSize: 3, Seed: 42, TrainPerClass: 1, ValidationPerClass: 1, Patience: 5}

	var output bytes.Buffer
	if err := fitModel(t.Context(), records, directory, config, tensor.CPU(), &output); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(output.String(), "epoch=1") {
		t.Fatal("missing epoch progress", output.String())
	}

	metadata, err := readMetadata(directory)
	if err != nil {
		t.Fatal(err)
	}

	if len(metadata.TrainIndices) != 10 || len(metadata.ValidationIndices) != 10 {
		t.Fatal("wrong saved split sizes")
	}

	if metadata.Settings.LearningRate != 0.001 || metadata.Settings.Beta1 != 0.9 || metadata.Settings.Beta2 != 0.999 || metadata.Settings.Epsilon != 1e-8 || metadata.Settings.Patience != 5 || metadata.Settings.MinDelta != 0.001 || metadata.Settings.Loss != "cross_entropy" || metadata.Settings.Selection != "minimum_validation_loss" {
		t.Fatal("missing effective training settings", metadata.Settings)
	}

	model, err := checkpoint.Load(filepath.Join(directory, "best.goml"))
	if err != nil {
		t.Fatal(err)
	}
	defer closeModel(model)

	dataset := &imageDataset{records: records, indices: metadata.ValidationIndices, normalization: metadata.Normalization}

	loader, err := classificationLoader(dataset, 3, false, 42)
	if err != nil {
		t.Fatal(err)
	}

	report, err := trainer.EvaluateClassification(t.Context(), model, tensor.CPU(), 10, loader)
	if err != nil || report.Samples != 10 || report.Batches != 4 {
		t.Fatal("partial batch evaluation failed", report, err)
	}

	if err := fitModel(t.Context(), records, directory, config, tensor.CPU(), &output); err == nil {
		t.Fatal("overwrote an existing run")
	}

	imagePath := filepath.Join(t.TempDir(), "image.png")

	file, err := os.Create(imagePath)
	if err != nil {
		t.Fatal(err)
	}

	if err := png.Encode(file, image.NewRGBA(image.Rect(0, 0, 40, 24))); err != nil {
		t.Fatal(err)
	}

	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	// The CLI uses Metal; CPU exercises the identical preprocessing/prediction path.
	if err := predictImage(context.Background(), directory, imagePath, tensor.CPU(), &output); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(output.String(), "probability=") {
		t.Fatal("missing ranked probabilities", output.String())
	}

	if err := run(t.Context(), []string{"predict", "-run", directory}, &output); err == nil {
		t.Fatal("accepted prediction without an image")
	}
}

func TestCLIRejectsInvalidTrainingOptionsBeforeOpeningMetal(t *testing.T) {
	for _, args := range [][]string{
		nil,
		{"unknown"},
		{"train", "-epochs", "0"},
		{"train", "-batch", "0"},
		{"train", "-patience", "-1"},
		{"train", "-train-per-class", "5000"},
		{"train", "unexpected"},
		{"evaluate", "-split", "train"},
	} {
		if err := run(t.Context(), args, &bytes.Buffer{}); err == nil {
			t.Fatal("accepted invalid CLI options", args)
		}
	}
}

func TestMetadataRejectsInvalidNormalization(t *testing.T) {
	directory := t.TempDir()
	metadata := runMetadata{
		Version: 1, Model: "cifar10-cnn-16-32-64", Classes: classNames,
		Config:        trainConfig{Epochs: 1, BatchSize: 1, TrainPerClass: 1, ValidationPerClass: 1},
		Normalization: normalization{Std: [3]float64{0, 1, 1}},
		TrainIndices:  []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ValidationIndices: []int{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
	}

	encoded, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(directory, "metadata.json"), encoded, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := readMetadata(directory); err == nil || !strings.Contains(err.Error(), "normalization") {
		t.Fatal("accepted invalid preprocessing metadata")
	}
}

type cancelAfterEpoch struct {
	cancel context.CancelFunc
}

func (w cancelAfterEpoch) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte("epoch=1")) {
		w.cancel()
	}

	return len(p), nil
}

func TestCanceledTrainingPreservesCompletedEpochAndReason(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	directory := filepath.Join(t.TempDir(), "run")
	config := trainConfig{Epochs: 3, BatchSize: 3, Seed: 42, TrainPerClass: 1, ValidationPerClass: 1}

	err := fitModel(ctx, fixtureRecords(3), directory, config, tensor.CPU(), cancelAfterEpoch{cancel: cancel})
	if !errors.Is(err, context.Canceled) {
		t.Fatal("missing cancellation error", err)
	}

	encoded, err := os.ReadFile(filepath.Join(directory, "history.json"))
	if err != nil {
		t.Fatal(err)
	}

	var history trainingHistory
	if err := json.Unmarshal(encoded, &history); err != nil {
		t.Fatal(err)
	}

	if len(history.Epochs) != 1 || history.StopReason != "canceled" {
		t.Fatal("lost completed epoch or cancellation reason", history)
	}

	if _, err := os.Stat(filepath.Join(directory, "best.goml")); err != nil {
		t.Fatal("lost best checkpoint on cancellation", err)
	}
}

func TestZeroPatienceCompletesRequestedEpochsAndSavesFinalModel(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "run")

	config := trainConfig{Epochs: 2, BatchSize: 3, Seed: 42, TrainPerClass: 1, ValidationPerClass: 1, Patience: 0}
	if err := fitModel(t.Context(), fixtureRecords(3), directory, config, tensor.CPU(), &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}

	metadata, err := readMetadata(directory)
	if err != nil || metadata.Settings.Patience != 0 {
		t.Fatal("early stopping was not disabled", metadata.Settings, err)
	}

	encoded, err := os.ReadFile(filepath.Join(directory, "history.json"))
	if err != nil {
		t.Fatal(err)
	}

	var history trainingHistory
	if err := json.Unmarshal(encoded, &history); err != nil {
		t.Fatal(err)
	}

	if len(history.Epochs) != 2 || history.StopReason != "completed" {
		t.Fatal("did not complete requested epochs", history)
	}

	model, err := checkpoint.Load(filepath.Join(directory, "final.goml"))
	if err != nil {
		t.Fatal("missing final-epoch checkpoint", err)
	}

	closeModel(model)
}
