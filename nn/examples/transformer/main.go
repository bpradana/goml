package main

import (
	"fmt"
	"log"
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

// Train a small encoder to recover the first token from the final position.
func run() error {
	rng := rand.New(rand.NewPCG(1, 2))

	encoder, err := layer.NewTransformerEncoder(4, 2, 8, rng)
	if err != nil {
		return err
	}

	readout, err := layer.NewDense(4, 2, true, rng)
	if err != nil {
		return err
	}

	model, err := layer.NewSequential(layer.NewPositionalEncoding(), encoder, layer.NewLastStep(), readout)
	if err != nil {
		return err
	}

	sequences := make([]float32, 8*3*4)
	labels := make([]int, 8)

	for sample := range 8 {
		labels[sample] = sample >> 2
		for position := range 3 {
			bit := (sample >> (2 - position)) & 1
			sequences[(sample*3+position)*4] = float32(bit)
			sequences[(sample*3+position)*4+1] = float32(1 - bit)
		}
	}

	data, err := tensor.From(sequences, 8, 3, 4)
	if err != nil {
		return err
	}

	input, err := nn.NewValue(data, false)
	if err != nil {
		return err
	}

	opt, err := optimizer.NewAdam(model.Parameters(), 0.02, 0.9, 0.999, 1e-8)
	if err != nil {
		return err
	}

	criterion := loss.NewCrossEntropy()

	var initialLoss float32

	for iteration := range 250 {
		prediction, err := model.Forward(input)
		if err != nil {
			return err
		}

		objective, err := criterion.Forward(prediction, labels)
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

	prediction, err := nn.NewValue(predictions, false)
	if err != nil {
		return err
	}

	objective, err := criterion.Forward(prediction, labels)
	if err != nil {
		return err
	}

	lossValue2, err := objective.Data().At()
	if err != nil {
		return err
	}

	finalLoss := lossValue2

	values, err := predictions.HostData()
	if err != nil {
		return err
	}

	correct := 0

	for sample, label := range labels {
		class := 0
		if values[sample*2+1] > values[sample*2] {
			class = 1
		}

		if class == label {
			correct++
		}
	}

	if !(finalLoss < 0.03) || correct != len(labels) {
		return fmt.Errorf("transformer: training did not converge: loss %g, correct %d/%d", finalLoss, correct, len(labels))
	}

	fmt.Printf("transformer: loss %.6f -> %.6f, correct %d/%d\n", initialLoss, finalLoss, correct, len(labels))

	return nil
}
