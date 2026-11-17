// Command mnist trains and evaluates classifiers using local Dataset/DataLoader APIs.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/data/datasets/mnist"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/trainer"
)

type runOptions struct {
	imagesPath, labelsPath, directory, split, device               string
	modelName                                                      string
	limit, epochs, batch                                           int
	seed                                                           uint64
	shuffle, evaluate                                              bool
	saveModelPath, saveWeightsPath, loadModelPath, loadWeightsPath string
	predict                                                        bool
	bestModelPath                                                  string
	patience                                                       int
	minDelta                                                       float64
}

func main() {
	var options runOptions
	flag.StringVar(&options.imagesPath, "images", "", "optional explicit local image IDX file, plain or .gz")
	flag.StringVar(&options.labelsPath, "labels", "", "optional explicit local label IDX file, plain or .gz")
	flag.StringVar(&options.directory, "data", "archive", "local MNIST directory")
	flag.StringVar(&options.split, "split", "train", "dataset split: train or test")
	flag.StringVar(&options.device, "device", "cpu", "cpu, metal (-tags metal), or webgpu (-tags webgpu)")
	flag.StringVar(&options.modelName, "model", "dense", "dense or cnn; ignored with -load-model")
	flag.IntVar(&options.limit, "limit", 1000, "number of selected examples")
	flag.IntVar(&options.epochs, "epochs", 3, "training epochs")
	flag.IntVar(&options.batch, "batch", 32, "maximum batch size")
	flag.Uint64Var(&options.seed, "seed", 42, "deterministic shuffle seed")
	flag.BoolVar(&options.shuffle, "shuffle", true, "shuffle training samples each epoch")
	flag.BoolVar(&options.evaluate, "evaluate", false, "evaluate the local held-out test split after each epoch")
	flag.StringVar(&options.saveModelPath, "save-model", "archive/mnist-model.goml", "network output path; empty disables saving")
	flag.StringVar(&options.saveWeightsPath, "save-weights", "archive/mnist-weights.goml", "weights output path; empty disables saving")
	flag.StringVar(&options.loadModelPath, "load-model", "", "restore architecture and weights")
	flag.StringVar(&options.loadWeightsPath, "load-weights", "", "restore weights into the selected architecture")
	flag.BoolVar(&options.predict, "predict", false, "evaluate a loaded checkpoint without training or saving")
	flag.StringVar(&options.bestModelPath, "best-model", "", "optional best validation model checkpoint")
	flag.IntVar(&options.patience, "patience", 0, "validation-loss early stopping patience; zero disables")
	flag.Float64Var(&options.minDelta, "min-delta", 0, "strict validation loss improvement for early stopping")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := runContext(ctx, options); err != nil {
		log.Fatal(err)
	}
}

func run(options runOptions) error {
	return runContext(context.Background(), options)
}

func runContext(ctx context.Context, options runOptions) error {
	if options.limit <= 0 || options.batch <= 0 || (!options.predict && options.epochs <= 0) {
		return fmt.Errorf("mnist: limit, batch, and training epochs must be positive")
	}

	if options.loadModelPath != "" && options.loadWeightsPath != "" {
		return fmt.Errorf("mnist: choose either load-model or load-weights")
	}

	if options.predict && options.loadModelPath == "" && options.loadWeightsPath == "" {
		return fmt.Errorf("mnist: predict requires load-model or load-weights")
	}

	if options.patience < 0 || options.minDelta < 0 || math.IsNaN(options.minDelta) || math.IsInf(options.minDelta, 0) {
		return fmt.Errorf("mnist: patience and finite min-delta must be nonnegative")
	}

	if !options.predict {
		if (options.patience > 0 || options.bestModelPath != "") && !options.evaluate {
			return fmt.Errorf("mnist: patience and best-model require held-out evaluation")
		}

		var outputs []string

		for _, path := range []string{options.saveModelPath, options.saveWeightsPath, options.bestModelPath} {
			if path == "" {
				continue
			}

			absolute, err := filepath.Abs(path)
			if err != nil {
				return err
			}

			for _, previous := range outputs {
				if previous == absolute {
					return fmt.Errorf("mnist: final model, weights, and best model need different output paths")
				}
			}

			outputs = append(outputs, absolute)
		}
	}

	var (
		model *layer.Sequential
		err   error
	)
	if options.loadModelPath != "" {
		model, err = checkpoint.Load(options.loadModelPath)
	} else {
		model, err = newModel(options.modelName, rand.New(rand.NewPCG(1, 2)))
	}

	if err != nil {
		return err
	}

	defer func() {
		for _, parameter := range model.Parameters() {
			parameter.ZeroGrad()
			_ = parameter.Value().Data().Close()
		}
	}()

	if options.loadWeightsPath != "" {
		if err := checkpoint.LoadWeights(options.loadWeightsPath, model.Parameters()); err != nil {
			return err
		}
	}

	device, err := openDevice(options.device)
	if err != nil {
		return err
	}
	defer func() { _ = device.Close() }()

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		return err
	}

	dataset, err := selectedDataset(ctx, options)
	if err != nil {
		return err
	}

	loader, err := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: options.batch, Shuffle: !options.predict && options.shuffle, Seed: options.seed})
	if err != nil {
		return err
	}

	if options.predict {
		correct, meanLoss, err := predict(ctx, model, loader, device)
		if err != nil {
			return err
		}

		fmt.Printf("examples=%d accuracy=%.2f%% mean_cross_entropy=%.4f\n", dataset.Len(), 100*float64(correct)/float64(dataset.Len()), meanLoss)

		return nil
	}

	var (
		evaluation      trainer.BatchSource[*data.ClassificationBatch]
		evaluationCount int
	)

	if options.evaluate {
		testOptions := options
		testOptions.imagesPath, testOptions.labelsPath, testOptions.split = "", "", "test"
		testOptions.limit = min(options.limit, 10000)

		test, err := selectedDataset(ctx, testOptions)
		if err != nil {
			return err
		}

		evaluationCount = test.Len()

		evaluation, err = data.NewLoader(test, data.CollateClassification, data.LoaderOptions{BatchSize: options.batch})
		if err != nil {
			return err
		}
	}

	opt, err := optimizer.NewAdam(model.Parameters(), 0.001, 0.9, 0.999, 1e-8)
	if err != nil {
		return err
	}
	defer func() { _ = opt.Close() }()

	runner, err := trainer.NewClassifier(model, opt, device, 10)
	if err != nil {
		return err
	}

	fitOptions := trainer.FitOptions{Epochs: options.epochs, BestCheckpointPath: options.bestModelPath, OnEpochEnd: func(report trainer.EpochReport) error {
		fmt.Printf("epoch=%d mean_cross_entropy=%.4f\n", report.Epoch, report.Train.MeanLoss)

		if report.Validation != nil {
			fmt.Printf("test_examples=%d test_accuracy=%.2f%% test_cross_entropy=%.4f\n", evaluationCount, 100*report.Classification.Accuracy, report.Validation.MeanLoss)
		}

		return nil
	}}
	if options.patience != 0 {
		fitOptions.EarlyStopping = &trainer.EarlyStopping{Patience: options.patience, MinDelta: options.minDelta}
	}

	if _, err := runner.Fit(ctx, loader, evaluation, fitOptions); err != nil {
		return err
	}

	if options.saveModelPath != "" {
		if err := checkpoint.Save(options.saveModelPath, model); err != nil {
			return err
		}

		fmt.Printf("saved_model=%s\n", options.saveModelPath)
	}

	if options.saveWeightsPath != "" {
		if err := checkpoint.SaveWeights(options.saveWeightsPath, model.Parameters()); err != nil {
			return err
		}

		fmt.Printf("saved_weights=%s\n", options.saveWeightsPath)
	}

	return nil
}

func selectedDataset(ctx context.Context, options runOptions) (data.Dataset[data.ClassificationSample], error) {
	var (
		dataset *mnist.Dataset
		err     error
	)

	if options.imagesPath != "" || options.labelsPath != "" {
		if options.imagesPath == "" || options.labelsPath == "" {
			return nil, fmt.Errorf("mnist: provide both images and labels")
		}

		dataset, err = mnist.OpenFiles(ctx, options.imagesPath, options.labelsPath)
	} else {
		split := mnist.Train
		switch options.split {
		case "", "train":
		case "test":
			split = mnist.Test
		default:
			return nil, fmt.Errorf("mnist: split must be train or test")
		}

		directory := options.directory
		if directory == "" {
			directory = "archive"
		}

		dataset, err = mnist.OpenContext(ctx, directory, split)
	}

	if err != nil {
		return nil, err
	}

	if options.limit > dataset.Len() {
		return nil, fmt.Errorf("mnist: limit %d exceeds %d available examples", options.limit, dataset.Len())
	}

	indices := make([]int, options.limit)
	for i := range indices {
		indices[i] = i
	}

	return data.Subset[data.ClassificationSample](dataset, indices)
}
