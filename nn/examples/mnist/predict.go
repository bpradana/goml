package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func predict(ctx context.Context, model *layer.Sequential, loader *data.Loader[data.ClassificationSample, *data.ClassificationBatch], device backend.Device) (int, float32, error) {
	correct, count := 0, 0

	var total float32

	for batch, err := range loader.Batches(ctx, 0) {
		if err != nil {
			return 0, 0, err
		}

		matches, meanLoss, predictErr := predictBatch(model, batch, device)

		closeErr := batch.Close()
		if err := errors.Join(predictErr, closeErr); err != nil {
			return 0, 0, err
		}

		correct += matches
		count += len(batch.Labels)
		total += meanLoss * float32(len(batch.Labels))
	}

	if count == 0 {
		return 0, 0, fmt.Errorf("mnist: cannot evaluate an empty dataset")
	}

	return correct, total / float32(count), nil
}

func predictBatch(model *layer.Sequential, batch *data.ClassificationBatch, device backend.Device) (int, float32, error) {
	scope, err := tensor.NewScope(device)
	if err != nil {
		return 0, 0, err
	}
	defer func() { _ = scope.Close() }()

	images, err := scope.To(batch.Inputs)
	if err != nil {
		return 0, 0, err
	}

	logits, err := model.Predict(images)
	if err != nil {
		return 0, 0, err
	}

	value, err := nn.NewValue(logits, false)
	if err != nil {
		return 0, 0, err
	}

	objective, err := loss.NewCrossEntropy().Forward(value, batch.Labels)
	if err != nil {
		return 0, 0, err
	}

	lossValues, err := objective.Data().HostData()
	if err != nil {
		return 0, 0, err
	}

	scores, err := logits.HostData()
	if err != nil {
		return 0, 0, err
	}

	correct, classes := 0, logits.Shape()[1]

	for row, label := range batch.Labels {
		best := 0
		for class := 1; class < classes; class++ {
			if scores[row*classes+class] > scores[row*classes+best] {
				best = class
			}
		}

		if best == label {
			correct++
		}
	}

	return correct, lossValues[0], nil
}
