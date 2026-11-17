package main

import (
	"context"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/trainer"
)

func predict(ctx context.Context, model *layer.Sequential, loader *data.Loader[data.ClassificationSample, *data.ClassificationBatch], device backend.Device) (int, float32, error) {
	report, err := trainer.EvaluateClassification(ctx, model, device, 10, loader)
	if err != nil {
		return 0, 0, err
	}

	var correct uint64
	for class, row := range report.Classification.Confusion {
		correct += row[class]
	}

	return int(correct), float32(report.MeanLoss), nil
}
