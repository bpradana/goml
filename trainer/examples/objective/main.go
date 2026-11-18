package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

func main() {
	path := flag.String("best-weights", "", "optional best configured-objective weights path")

	flag.Parse()

	if err := run(*path); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(checkpointPath string) error {
	input, err := tensor.From([]float32{-1.1, -1.1, -1.1, 1.1, 1.1, -1.1, 1.1, 1.1}, 4, 2)
	if err != nil {
		return err
	}
	defer func() { _ = input.Close() }()

	target, err := tensor.From([]float32{-1, -1, -1, 1, 1, -1, 1, 1}, 4, 2)
	if err != nil {
		return err
	}
	defer func() { _ = target.Close() }()

	dataset, err := data.NewRegressionDataset(input, target)
	if err != nil {
		return err
	}

	loader, err := data.NewLoader(dataset, data.CollateRegression, data.LoaderOptions{BatchSize: 3, Shuffle: true, Seed: 42})
	if err != nil {
		return err
	}

	dense, err := layer.NewDense(2, 2, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		return err
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		return err
	}

	defer func() {
		for _, parameter := range model.Parameters() {
			parameter.ZeroGrad()
			_ = parameter.Value().Data().Close()
		}
	}()

	opt, err := optimizer.NewAdam(model.Parameters(), 0.05, 0.9, 0.999, 1e-8)
	if err != nil {
		return err
	}
	defer func() { _ = opt.Close() }()

	runner, err := trainer.NewRegressorWithConfig(model, opt, tensor.CPU(), trainer.RegressorConfig{Loss: reconstructionLoss{}})
	if err != nil {
		return err
	}

	before, err := runner.Evaluate(context.Background(), loader)
	if err != nil {
		return err
	}

	options := trainer.FitOptions{Epochs: 100}
	if checkpointPath != "" {
		options.BestCheckpointPath = checkpointPath
		// Application policy selects weights-only serialization for this model.
		options.SaveBest = func(path string) error {
			return checkpoint.SaveWeights(path, model.Parameters())
		}
	}

	history, err := runner.Fit(context.Background(), loader, loader, options)
	if err != nil {
		return err
	}

	last := history.Epochs[len(history.Epochs)-1]
	fmt.Printf("objective=%g reconstruction_mse=%g\n", last.Validation.MeanLoss, last.Regression.MSE)

	if last.Validation.MeanLoss >= before.MeanLoss*0.1 {
		return fmt.Errorf("denoising objective did not improve")
	}

	return nil
}
