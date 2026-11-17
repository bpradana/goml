package main

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/bpradana/goml/data"

	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
)

func TestPredict_WeightsPartialBatchLossAndUsesInference(t *testing.T) {
	dense, err := layer.NewDense(784, 10, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	zeros, err := tensor.Zeros(784, 10)
	if err != nil {
		t.Fatal(err)
	}

	if err := dense.Parameters()[0].Update(zeros); err != nil {
		t.Fatal(err)
	}

	bias, err := tensor.From([]float32{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, 10)
	if err != nil {
		t.Fatal(err)
	}

	if err := dense.Parameters()[1].Update(bias); err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(layer.NewFlatten(), dense)
	if err != nil {
		t.Fatal(err)
	}

	inputs, err := tensor.Zeros(3, 1, 28, 28)
	if err != nil {
		t.Fatal(err)
	}

	dataset, err := data.NewClassificationDataset(inputs, []int{2, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	loader, err := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: 2})
	if err != nil {
		t.Fatal(err)
	}

	correct, meanLoss, err := predict(context.Background(), model, loader, tensor.CPU())
	if err != nil {
		t.Fatal(err)
	}

	wantLoss := math.Log(math.E+9) - 2.0/3
	if correct != 2 || math.Abs(float64(meanLoss)-wantLoss) > 1e-6 {
		t.Fatalf("got correct=%d loss=%g, want correct=2 loss=%g", correct, meanLoss, wantLoss)
	}

	for _, parameter := range model.Parameters() {
		if parameter.Value().Grad() != nil {
			t.Fatal("inference retained a parameter gradient")
		}
	}
}
