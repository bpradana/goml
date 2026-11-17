package main

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

type points [][2]float32

func (p points) Len() int { return len(p) }

func (p points) Get(ctx context.Context, index int) ([2]float32, error) {
	if err := ctx.Err(); err != nil {
		return [2]float32{}, err
	}

	if index < 0 || index >= len(p) {
		return [2]float32{}, fmt.Errorf("index out of range")
	}

	return p[index], nil
}

func collate(samples [][2]float32) (*tensor.Tensor, error) {
	values := make([]float32, 0, len(samples)*2)
	for _, sample := range samples {
		values = append(values, sample[:]...)
	}

	return tensor.From(values, len(samples), 2)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	loader, err := data.NewLoader(points{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, collate, data.LoaderOptions{BatchSize: 4})
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
		for _, p := range model.Parameters() {
			p.ZeroGrad()
			_ = p.Value().Data().Close()
		}
	}()

	opt, err := optimizer.NewSGD(model.Parameters(), 0.1)
	if err != nil {
		return err
	}

	step := func(training bool) trainer.StepFunc[*tensor.Tensor] {
		return func(ctx context.Context, batch *tensor.Tensor) (result trainer.StepResult, err error) {
			defer func() { err = errors.Join(err, batch.Close()) }()

			if training {
				opt.ZeroGrad()
				defer opt.ZeroGrad()
			}

			if err := ctx.Err(); err != nil {
				return result, err
			}

			scope, err := tensor.NewScope(tensor.CPU())
			if err != nil {
				return result, err
			}
			defer func() { err = errors.Join(err, scope.Close()) }()

			input, err := scope.To(batch)
			if err != nil {
				return result, err
			}

			value, err := nn.NewValue(input, false)
			if !training {
				value, err = nn.Inference(input)
			}

			if err != nil {
				return result, err
			}

			prediction, err := model.Forward(value)
			if err != nil {
				return result, err
			}

			objective, err := loss.NewMSE().Forward(prediction, value)
			if err != nil {
				return result, err
			}
			// A reconstruction task with explicit output regularization uses
			// an ordinary tensor batch, not a Trainer-specific task hierarchy.
			penalty, err := nn.Mul(prediction, prediction)
			if err != nil {
				return result, err
			}

			penalty, err = nn.Mean(penalty)
			if err != nil {
				return result, err
			}

			penalty, err = nn.Scale(penalty, 0.001)
			if err != nil {
				return result, err
			}

			objective, err = nn.Add(objective, penalty)
			if err != nil {
				return result, err
			}

			values, err := objective.Data().HostData()
			if err != nil {
				return result, err
			}

			if training {
				if err := objective.Backward(); err != nil {
					return result, err
				}

				if err := opt.Step(); err != nil {
					return result, err
				}
			}

			return trainer.StepResult{MeanLoss: float64(values[0]), Samples: uint64(batch.Shape()[0])}, nil
		}
	}

	history, err := trainer.Fit(context.Background(), loader, loader, step(true), step(false), trainer.FitOptions{Epochs: 100})
	if err != nil {
		return err
	}

	last := history.Epochs[len(history.Epochs)-1]
	fmt.Printf("regularized_reconstruction_loss=%g\n", last.Validation.MeanLoss)

	if last.Validation.MeanLoss >= history.Epochs[0].Validation.MeanLoss*0.1 {
		return fmt.Errorf("reconstruction did not improve")
	}

	return nil
}
