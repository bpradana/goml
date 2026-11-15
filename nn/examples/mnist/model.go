package main

import (
	"fmt"
	"math/rand/v2"

	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/layer"
)

// Both models accept normalized NCHW images and return [batch,10] logits.
func newModel(name string, rng *rand.Rand) (*layer.Sequential, error) {
	switch name {
	case "dense":
		hidden, err := layer.NewDense(784, 32, true, rng)
		if err != nil {
			return nil, err
		}

		output, err := layer.NewDense(32, 10, true, rng)
		if err != nil {
			return nil, err
		}

		return layer.NewSequential(layer.NewFlatten(), hidden, activation.NewReLU(), output)

	case "cnn":
		first, err := layer.NewConv2D(1, 8, 3, 2, 1, true, rng)
		if err != nil {
			return nil, err
		}

		second, err := layer.NewConv2D(8, 16, 3, 2, 1, true, rng)
		if err != nil {
			return nil, err
		}

		output, err := layer.NewDense(16*7*7, 10, true, rng)
		if err != nil {
			return nil, err
		}

		return layer.NewSequential(
			first,
			activation.NewReLU(),
			second,
			activation.NewReLU(),
			layer.NewFlatten(),
			output,
		)

	default:
		return nil, fmt.Errorf("mnist: unknown model %q: want dense or cnn", name)
	}
}
