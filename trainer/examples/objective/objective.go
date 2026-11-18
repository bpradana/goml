package main

import (
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/loss"
)

// reconstructionLoss averages reconstruction errors and output energy over
// every sample/output. The second term penalizes unnecessarily large outputs.
// Tensor/nn operations preserve the caller's scope, device, and autograd graph.
type reconstructionLoss struct{}

func (reconstructionLoss) Forward(prediction, target *nn.Value) (*nn.Value, error) {
	reconstruction, err := loss.NewMSE().Forward(prediction, target)
	if err != nil {
		return nil, err
	}

	energy, err := nn.Mul(prediction, prediction)
	if err != nil {
		return nil, err
	}

	energy, err = nn.Mean(energy)
	if err != nil {
		return nil, err
	}

	penalty, err := nn.Scale(energy, 0.001)
	if err != nil {
		return nil, err
	}

	return nn.Add(reconstruction, penalty)
}

var _ loss.Loss[*nn.Value] = reconstructionLoss{}
