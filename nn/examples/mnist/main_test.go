package main

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/tensor"
)

func writeIDX(t testing.TB, header []uint32, data []byte) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "fixture.gz")

	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	writer := gzip.NewWriter(file)
	if err := binary.Write(writer, binary.BigEndian, header); err != nil {
		t.Fatal(err)
	}

	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}

	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	return path
}

func TestRun_TrainsOnLocalIDXFixtures(t *testing.T) {
	pixels := make([]byte, 3*784)
	for i := 784; i < len(pixels); i++ {
		pixels[i] = 255
	}

	images := writeIDX(t, []uint32{2051, 3, 28, 28}, pixels)
	labels := writeIDX(t, []uint32{2049, 3}, []byte{0, 1, 2})

	for _, model := range []string{"dense", "cnn"} {
		t.Run(model, func(t *testing.T) {
			if err := run(runOptions{imagesPath: images, labelsPath: labels, modelName: model, limit: 3, epochs: 2, batch: 2}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRun_RejectsUnknownModelBeforeOpeningDataset(t *testing.T) {
	err := run(runOptions{imagesPath: "missing-images.gz", labelsPath: "missing-labels.gz", modelName: "unknown", limit: 1, epochs: 1, batch: 1})
	if err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("expected model validation error, got %v", err)
	}
}

func TestRun_RejectsInvalidTrainingLimits(t *testing.T) {
	for _, arguments := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {-1, 1, 1}} {
		if err := run(runOptions{modelName: "cnn", limit: arguments[0], epochs: arguments[1], batch: arguments[2]}); err == nil || !strings.Contains(err.Error(), "must be positive") {
			t.Fatalf("accepted invalid arguments %v: %v", arguments, err)
		}
	}
}

func TestRun_SavesLoadsAndEvaluatesBothMNISTArchitectures(t *testing.T) {
	images := writeIDX(t, []uint32{2051, 3, 28, 28}, make([]byte, 3*784))

	labels := writeIDX(t, []uint32{2049, 3}, []byte{0, 1, 2})
	for _, name := range []string{"dense", "cnn"} {
		t.Run(name, func(t *testing.T) {
			directory := t.TempDir()
			modelPath := filepath.Join(directory, "model.goml")
			weightsPath := filepath.Join(directory, "weights.goml")

			options := runOptions{imagesPath: images, labelsPath: labels, modelName: name, limit: 3, epochs: 1, batch: 2, saveModelPath: modelPath, saveWeightsPath: weightsPath}
			if err := run(options); err != nil {
				t.Fatal(err)
			}

			model, err := checkpoint.Load(modelPath)
			if err != nil {
				t.Fatal(err)
			}

			weightsModel, err := newModel(name, rand.New(rand.NewPCG(9, 10)))
			if err != nil {
				t.Fatal(err)
			}

			if err := checkpoint.LoadWeights(weightsPath, weightsModel.Parameters()); err != nil {
				t.Fatal(err)
			}

			input, err := tensor.Ones(1, 1, 28, 28)
			if err != nil {
				t.Fatal(err)
			}

			want, err := model.Predict(input)
			if err != nil {
				t.Fatal(err)
			}

			got, err := weightsModel.Predict(input)
			if err != nil || !slices.Equal(backendtest.HostData(t, got), backendtest.HostData(t, want)) {
				t.Fatalf("saved weights and full model differ: %v", err)
			}

			before, err := os.ReadFile(modelPath)
			if err != nil {
				t.Fatal(err)
			}

			options.loadModelPath = modelPath
			options.modelName = "ignored-for-loaded-architecture"
			options.predict = true

			options.epochs = 0
			if err := run(options); err != nil {
				t.Fatal(err)
			}

			after, err := os.ReadFile(modelPath)
			if err != nil || !slices.Equal(before, after) {
				t.Fatal("inference rewrote checkpoint")
			}

			options.predict = false
			options.epochs = 1
			options.saveModelPath = ""

			options.saveWeightsPath = ""
			if err := run(options); err != nil {
				t.Fatal(err)
			}

			options.loadModelPath = ""
			options.loadWeightsPath = weightsPath

			options.modelName = name
			if err := run(options); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRun_RejectsConflictingCheckpointOptions(t *testing.T) {
	for _, options := range []runOptions{
		{limit: 1, epochs: 1, batch: 1, loadModelPath: "model", loadWeightsPath: "weights"},
		{limit: 1, batch: 1, predict: true},
		{limit: 1, epochs: 1, batch: 1, saveModelPath: "same", saveWeightsPath: "./same"},
	} {
		if err := run(options); err == nil {
			t.Fatal("accepted conflicting checkpoint options")
		}
	}
}

func TestSelectedDataset_ExplicitHeldOutSplitAndLimit(t *testing.T) {
	directory := t.TempDir()
	for _, prefix := range []string{"train", "t10k"} {
		images := writeIDX(t, []uint32{2051, 3, 28, 28}, make([]byte, 3*784))

		labels := writeIDX(t, []uint32{2049, 3}, []byte{2, 3, 4})
		if err := os.Rename(images, filepath.Join(directory, prefix+"-images-idx3-ubyte.gz")); err != nil {
			t.Fatal(err)
		}

		if err := os.Rename(labels, filepath.Join(directory, prefix+"-labels-idx1-ubyte.gz")); err != nil {
			t.Fatal(err)
		}
	}

	options := runOptions{directory: directory, split: "test", limit: 3, batch: 2, epochs: 1, modelName: "dense", evaluate: true, shuffle: true, seed: 42}
	if err := run(options); err != nil {
		t.Fatal(err)
	}

	for _, split := range []string{"invalid", "train"} {
		options.split, options.limit = split, 4
		if _, err := selectedDataset(context.Background(), options); err == nil {
			t.Fatal("accepted invalid split or excessive limit")
		}
	}
}

func BenchmarkTrainingEndToEnd(b *testing.B) {
	images := writeIDX(b, []uint32{2051, 128, 28, 28}, make([]byte, 128*784))
	labels := writeIDX(b, []uint32{2049, 128}, make([]byte, 128))
	options := runOptions{imagesPath: images, labelsPath: labels, modelName: "dense", limit: 128, epochs: 1, batch: 32}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		if err := run(options); err != nil {
			b.Fatal(err)
		}
	}
}
