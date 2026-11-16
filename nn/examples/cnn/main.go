package main

import (
	"fmt"
	"log"
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

// Train a convolutional classifier on four synthetic horizontal and vertical lines.
func run() error {
	rng := rand.New(rand.NewPCG(1, 2))

	convolution, err := layer.NewConv2D(1, 3, 2, 1, 0, true, rng)
	if err != nil {
		return err
	}

	readout, err := layer.NewDense(27, 2, true, rng)
	if err != nil {
		return err
	}

	model, err := layer.NewSequential(convolution, activation.NewTanh(), layer.NewFlatten(), readout)
	if err != nil {
		return err
	}

	images := make([]float32, 4*4*4)

	for sample := range 4 {
		for pixel := range 4 {
			if sample < 2 {
				images[sample*16+(sample+1)*4+pixel] = 1
			} else {
				images[sample*16+pixel*4+sample-1] = 1
			}
		}
	}

	data, err := tensor.From(images, 4, 1, 4, 4)
	if err != nil {
		return err
	}

	labels := []int{0, 0, 1, 1}

	input, err := nn.NewValue(data, false)
	if err != nil {
		return err
	}

	opt, err := optimizer.NewAdam(model.Parameters(), 0.03, 0.9, 0.999, 1e-8)
	if err != nil {
		return err
	}

	criterion := loss.NewCrossEntropy()

	var initialLoss float32

	for iteration := range 200 {
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
		return fmt.Errorf("cnn: training did not converge: loss %g, correct %d/%d", finalLoss, correct, len(labels))
	}

	fmt.Printf("cnn: loss %.6f -> %.6f, correct %d/%d\n", initialLoss, finalLoss, correct, len(labels))

	return nil
}
