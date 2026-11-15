package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	rng := rand.New(rand.NewPCG(1, 2))

	dense, err := layer.NewDense(1, 1, true, rng)
	if err != nil {
		return err
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		return err
	}

	x, err := tensor.From([]float32{-1, 0, 1, 2}, 4, 1)
	if err != nil {
		return err
	}

	y, err := tensor.From([]float32{-1, 1, 3, 5}, 4, 1)
	if err != nil {
		return err
	}

	input, err := nn.NewValue(x, false)
	if err != nil {
		return err
	}

	target, err := nn.NewValue(y, false)
	if err != nil {
		return err
	}

	opt, err := optimizer.NewSGD(model.Parameters(), 0.05)
	if err != nil {
		return err
	}

	var criterion loss.Loss[*nn.Value] = loss.NewMSE()

	for iteration := 0; iteration < 1000; iteration++ {
		prediction, err := model.Forward(input)
		if err != nil {
			return err
		}

		objective, err := criterion.Forward(prediction, target)
		if err != nil {
			return err
		}

		if err := objective.Backward(); err != nil {
			return err
		}

		if err := opt.Step(); err != nil {
			return err
		}

		opt.ZeroGrad()
	}

	prediction, err := model.Predict(x)
	if err != nil {
		return err
	}

	for i, got := range prediction.Data() {
		if math.Abs(float64(got-y.Data()[i])) > 0.001 {
			return fmt.Errorf("linear: training did not converge at example %d: got %g", i, got)
		}
	}

	fmt.Println(prediction)

	return nil
}
