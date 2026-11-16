package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
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

	hidden, err := layer.NewDense(2, 8, true, rng)
	if err != nil {
		return err
	}

	output, err := layer.NewDense(8, 1, true, rng)
	if err != nil {
		return err
	}

	model, err := layer.NewSequential(hidden, activation.NewTanh(), output, activation.NewSigmoid())
	if err != nil {
		return err
	}

	x, err := tensor.From([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	if err != nil {
		return err
	}

	y, err := tensor.From([]float32{0, 1, 1, 0}, 4, 1)
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

	opt, err := optimizer.NewAdam(model.Parameters(), 0.03, 0.9, 0.999, 1e-8)
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

	values, err := prediction.HostData()
	if err != nil {
		return err
	}

	targets, err := y.HostData()
	if err != nil {
		return err
	}

	for i, got := range values {
		if math.Abs(float64(got-targets[i])) > 0.1 {
			return fmt.Errorf("xor: training did not converge at example %d: got %g", i, got)
		}
	}

	fmt.Println(prediction)

	return nil
}
