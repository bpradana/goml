package main

import (
	"compress/gzip"
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

func writeIDX(t *testing.T, header []uint32, data []byte) string {
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

func TestLoad_AcceptsValidImageAndLabelIDX(t *testing.T) {
	for _, tt := range []struct {
		name   string
		header []uint32
		size   int
	}{
		{"Images", []uint32{2051, 1, 28, 28}, 784},
		{"Labels", []uint32{2049, 1}, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := writeIDX(t, tt.header, make([]byte, tt.size))
			header, data, err := load(path, len(tt.header))

			if err != nil || len(header) != len(tt.header) || len(data) != tt.size {
				t.Fatalf("unexpected IDX result: %v %d %v", header, len(data), err)
			}
		})
	}
}

func TestLoad_RejectsMalformedIDX(t *testing.T) {
	for _, tt := range []struct {
		name   string
		header []uint32
		size   int
	}{
		{"BadImageMagic", []uint32{42, 1, 28, 28}, 784},
		{"BadDimensions", []uint32{2051, 1, 1, 784}, 784},
		{"ZeroItems", []uint32{2049, 0}, 0},
		{"TruncatedData", []uint32{2051, 1, 28, 28}, 10},
		{"ExtraData", []uint32{2049, 1}, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := writeIDX(t, tt.header, make([]byte, tt.size))
			if _, _, err := load(path, len(tt.header)); err == nil {
				t.Fatal("expected IDX validation error")
			}
		})
	}
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
