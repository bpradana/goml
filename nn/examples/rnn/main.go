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

// Train a tanh RNN to sum three numbers using only the final hidden state.
func run() error {
	rng := rand.New(rand.NewPCG(1, 2))

	recurrent, err := layer.NewRNN(1, 4, rng)
	if err != nil {
		return err
	}

	readout, err := layer.NewDense(4, 1, true, rng)
	if err != nil {
		return err
	}

	model, err := layer.NewSequential(recurrent, layer.NewLastStep(), readout)
	if err != nil {
		return err
	}

	data, err := tensor.From([]float32{0.1, 0.2, 0.3, -0.3, 0.1, -0.2, 0.2, -0.2, 0.2, -0.1, 0.1, -0.1}, 4, 3, 1)
	if err != nil {
		return err
	}

	targetData, err := tensor.From([]float32{0.6, -0.4, 0.2, -0.1}, 4, 1)
	if err != nil {
		return err
	}

	input, err := nn.NewValue(data, false)
	if err != nil {
		return err
	}

	target, err := nn.NewValue(targetData, false)
	if err != nil {
		return err
	}

	opt, err := optimizer.NewAdam(model.Parameters(), 0.02, 0.9, 0.999, 1e-8)
	if err != nil {
		return err
	}

	criterion := loss.NewMSE()

	var initialLoss float32

	for iteration := range 300 {
		prediction, err := model.Forward(input)
		if err != nil {
			return err
		}

		objective, err := criterion.Forward(prediction, target)
		if err != nil {
			return err
		}

		if iteration == 0 {
			lossValue1, err := objective.Data().At()
			if err != nil {
				return err
			}

			initialLoss = lossValue1
		}

		if err := objective.Backward(); err != nil {
			return err
		}

		if err := opt.Step(); err != nil {
			return err
		}

		opt.ZeroGrad()
	}

	predictions, err := model.Predict(data)
	if err != nil {
		return err
	}

	values, err := predictions.HostData()
	if err != nil {
		return err
	}

	targets, err := targetData.HostData()
	if err != nil {
		return err
	}

	for sample, prediction := range values {
		if math.IsNaN(float64(prediction)) || math.Abs(float64(prediction-targets[sample])) > 0.03 {
			return fmt.Errorf("rnn: sequence %d did not converge: got %g", sample, prediction)
		}
	}

	fmt.Printf("rnn: initial loss %.6f, predictions %v\n", initialLoss, values)

	return nil
}
