package main

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/metrics/examples/custom/meanbias"
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
	input, err := tensor.From([]float32{-1, 0, 1, 2}, 4, 1)
	if err != nil {
		return err
	}
	defer func() { _ = input.Close() }()

	target, err := tensor.From([]float32{-1, 1, 3, 5}, 4, 1)
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

	dense, err := layer.NewDense(1, 1, true, rand.New(rand.NewPCG(1, 2)))
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

	bias := meanbias.New()

	runner, err := trainer.NewRegressor(model, opt, tensor.CPU(), bias)
	if err != nil {
		return err
	}

	history, err := runner.Fit(context.Background(), loader, loader, trainer.FitOptions{Epochs: 100})
	if err != nil {
		return err
	}

	last := history.Epochs[len(history.Epochs)-1]

	custom, err := bias.Result()
	if err != nil {
		return err
	}

	fmt.Printf("custom_mean_bias=%g observations=%d\n", custom.MeanBias, custom.Count)
	fmt.Printf("epochs=%d loss=%g mse=%g r2=%g\n", len(history.Epochs), last.Train.MeanLoss, last.Regression.MSE, last.Regression.R2)

	if last.Regression.MSE > 0.001 {
		return fmt.Errorf("regression did not converge")
	}

	return nil
}
