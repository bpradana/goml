package main

import (
	"math/rand/v2"

	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/layer"
)

func newModel(seed uint64) (*layer.Sequential, error) {
	rng := rand.New(rand.NewPCG(seed, 0))
	stages := make([]layer.Layer, 0, 13)

	inChannels := 3
	for _, channels := range []int{16, 32, 64} {
		convolution, err := layer.NewConv2D(inChannels, channels, 3, 1, 1, true, rng)
		if err != nil {
			return nil, err
		}

		pool, err := layer.NewMaxPool2D(2, 2, 0)
		if err != nil {
			return nil, err
		}

		stages = append(stages, convolution, activation.NewReLU(), pool)
		inChannels = channels
	}

	hidden, err := layer.NewDense(64*4*4, 64, true, rng)
	if err != nil {
		return nil, err
	}

	output, err := layer.NewDense(64, 10, true, rng)
	if err != nil {
		return nil, err
	}

	stages = append(stages, layer.NewFlatten(), hidden, activation.NewReLU(), output)

	return layer.NewSequential(stages...)
}

func closeModel(model *layer.Sequential) {
	for _, parameter := range model.Parameters() {
		parameter.ZeroGrad()
		_ = parameter.Value().Data().Close()
	}
}
