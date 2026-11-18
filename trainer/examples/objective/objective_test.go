package main

import (
	"math"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestReconstructionLoss_MeanReductionAndBothGradientTerms(t *testing.T) {
	scope, err := tensor.NewScope(tensor.CPU())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = scope.Close() }()

	p, err := tensor.From([]float32{1, 3}, 1, 2)
	if err != nil {
		t.Fatal(err)
	}

	target, err := tensor.From([]float32{0, 1}, 1, 2)
	if err != nil {
		t.Fatal(err)
	}

	defer func(owned *tensor.Tensor) { _ = owned.Close() }(p)
	defer func(owned *tensor.Tensor) { _ = owned.Close() }(target)

	p, err = scope.To(p)
	if err != nil {
		t.Fatal(err)
	}

	target, err = scope.To(target)
	if err != nil {
		t.Fatal(err)
	}

	prediction, err := nn.NewValue(p, true)
	if err != nil {
		t.Fatal(err)
	}
	defer prediction.ZeroGrad()

	expected, err := nn.NewValue(target, false)
	if err != nil {
		t.Fatal(err)
	}

	objective, err := (reconstructionLoss{}).Forward(prediction, expected)
	if err != nil {
		t.Fatal(err)
	}

	values, err := objective.Data().HostData()
	if err != nil || math.Abs(float64(values[0])-2.505) > 1e-6 {
		t.Fatal(values, err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	gradient, err := prediction.Grad().HostData()
	if err != nil || math.Abs(float64(gradient[0])-1.001) > 1e-6 || math.Abs(float64(gradient[1])-2.003) > 1e-6 {
		t.Fatal(gradient, err)
	}
}
