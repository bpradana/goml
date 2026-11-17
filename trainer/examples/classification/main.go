package main

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	input, err := tensor.From([]float32{-2, -1, 1, 2}, 4, 1)
	if err != nil {
		return err
	}
	defer func() { _ = input.Close() }()

	dataset, err := data.NewClassificationDataset(input, []int{0, 0, 1, 1})
	if err != nil {
		return err
	}

	loader, err := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: 3, Shuffle: true, Seed: 42})
	if err != nil {
		return err
	}

	dense, err := layer.NewDense(1, 2, true, rand.New(rand.NewPCG(1, 2)))
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

	opt, err := optimizer.NewSGD(model.Parameters(), 0.1)
	if err != nil {
		return err
	}

	runner, err := trainer.NewClassifier(model, opt, tensor.CPU(), 2)
	if err != nil {
		return err
	}

	history, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 20, OnEpochEnd: func(report trainer.EpochReport) error {
		if report.Epoch%5 == 0 {
			fmt.Printf("epoch=%d loss=%.4f validation=%.4f accuracy=%.2f%%\n", report.Epoch, report.Train.MeanLoss, report.Validation.MeanLoss, 100*report.Classification.Accuracy)
		}

		return nil
	}})
	if err != nil {
		return err
	}

	if history.Epochs[len(history.Epochs)-1].Classification.Accuracy != 1 {
		return fmt.Errorf("classifier did not converge")
	}

	return nil
}
