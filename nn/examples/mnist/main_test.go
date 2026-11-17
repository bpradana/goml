package main

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/trainer"
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
		{limit: 1, epochs: 1, batch: 1, evaluate: true, bestModelPath: "same", saveModelPath: "./same"},
		{limit: 1, epochs: 1, batch: 1, evaluate: true, bestModelPath: "same", saveWeightsPath: "./same"},
		{limit: 1, epochs: 1, batch: 1, bestModelPath: "best"},
		{limit: 1, epochs: 1, batch: 1, patience: 1},
		{limit: 1, epochs: 1, batch: 1, patience: -1},
		{limit: 1, epochs: 1, batch: 1, minDelta: -1},
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

func TestTrainer_CPUParityAndConvergence(t *testing.T) {
	checkTrainerDevice(t, tensor.CPU(), func() (uint64, uint64) { return 0, 0 })
}

func checkTrainerDevice(t *testing.T, device backend.Device, memory func() (uint64, uint64)) {
	t.Helper()

	for _, task := range []string{"classification", "regression"} {
		t.Run(task, func(t *testing.T) {
			outputs := 1
			if task == "classification" {
				outputs = 2
			}

			makeModel := func() *layer.Sequential {
				dense, err := layer.NewDense(1, outputs, true, rand.New(rand.NewPCG(1, 2)))
				if err != nil {
					t.Fatal(err)
				}

				model, err := layer.NewSequential(dense)
				if err != nil {
					t.Fatal(err)
				}

				return model
			}

			cpu, gpu := makeModel(), makeModel()
			defer func() {
				for _, model := range []*layer.Sequential{cpu, gpu} {
					for _, p := range model.Parameters() {
						p.ZeroGrad()
						_ = p.Value().Data().Close()
					}
				}
			}()

			if err := nn.ToDevice(gpu.Parameters(), device); err != nil {
				t.Fatal(err)
			}

			input, err := tensor.From([]float32{-2, -1, 1, 2}, 4, 1)
			if err != nil {
				t.Fatal(err)
			}

			defer func() { _ = input.Close() }()

			var baseline uint64

			callback := func(report trainer.EpochReport) error {
				if err := device.Synchronize(); err != nil {
					return err
				}

				live, _ := memory()
				if report.Epoch == 1 {
					baseline = live
				} else if live != baseline {
					t.Fatalf("resident storage grew: baseline=%d current=%d epoch=%d", baseline, live, report.Epoch)
				}

				return nil
			}

			var expected, actual trainer.History

			if task == "classification" {
				dataset, err := data.NewClassificationDataset(input, []int{0, 0, 1, 1})
				if err != nil {
					t.Fatal(err)
				}

				loader, err := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: 3, Shuffle: true, Seed: 42})
				if err != nil {
					t.Fatal(err)
				}

				co, _ := optimizer.NewSGD(cpu.Parameters(), 0.1)
				go_, _ := optimizer.NewSGD(gpu.Parameters(), 0.1)
				cr, _ := trainer.NewClassifier(cpu, co, tensor.CPU(), 2)
				gr, _ := trainer.NewClassifier(gpu, go_, device, 2)

				expected, err = cr.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 20})
				if err != nil {
					t.Fatal(err)
				}

				actual, err = gr.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 20, OnEpochEnd: callback})
				if err != nil {
					t.Fatal(err)
				}

				if actual.Epochs[19].Classification.Accuracy != 1 {
					t.Fatal("GPU classifier did not learn")
				}
			} else {
				targets, err := tensor.From([]float32{-3, -1, 3, 5}, 4, 1)
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = targets.Close() }()

				dataset, err := data.NewRegressionDataset(input, targets)
				if err != nil {
					t.Fatal(err)
				}

				loader, err := data.NewLoader(dataset, data.CollateRegression, data.LoaderOptions{BatchSize: 3})
				if err != nil {
					t.Fatal(err)
				}

				co, _ := optimizer.NewAdam(cpu.Parameters(), 0.05, 0.9, 0.999, 1e-8)
				go_, _ := optimizer.NewAdam(gpu.Parameters(), 0.05, 0.9, 0.999, 1e-8)

				defer func() { _ = co.Close(); _ = go_.Close() }()

				cr, _ := trainer.NewRegressor(cpu, co, tensor.CPU())
				gr, _ := trainer.NewRegressor(gpu, go_, device)

				expected, err = cr.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 100})
				if err != nil {
					t.Fatal(err)
				}

				actual, err = gr.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 100, OnEpochEnd: callback})
				if err != nil {
					t.Fatal(err)
				}

				if actual.Epochs[99].Regression.MSE > 0.001 {
					t.Fatal("GPU regressor did not learn")
				}
			}

			for epoch := range actual.Epochs {
				if math.Abs(actual.Epochs[epoch].Validation.MeanLoss-expected.Epochs[epoch].Validation.MeanLoss) > 0.001 {
					t.Fatalf("validation parity at epoch %d: GPU=%g CPU=%g", epoch, actual.Epochs[epoch].Validation.MeanLoss, expected.Epochs[epoch].Validation.MeanLoss)
				}
			}

			for _, p := range gpu.Parameters() {
				if p.Value().Data().Device().ID() != device.ID() || p.Value().Grad() != nil {
					t.Fatal("parameter moved or gradient retained")
				}
			}

			_, peak := memory()
			t.Logf("%s: stable device bytes=%d peak=%d validation loss=%g", task, baseline, peak, actual.Epochs[len(actual.Epochs)-1].Validation.MeanLoss)
		})
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if live, _ := memory(); live != 0 {
		t.Fatalf("model/optimizer cleanup left %d bytes", live)
	}
}
