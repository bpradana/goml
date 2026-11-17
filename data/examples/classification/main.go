package main

import (
	"context"
	"fmt"
	"os"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/tensor"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	inputs, err := tensor.From([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	if err != nil {
		return err
	}
	defer func() { _ = inputs.Close() }()

	dataset, err := data.NewClassificationDataset(inputs, []int{0, 1, 1, 0})
	if err != nil {
		return err
	}

	loader, err := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: 3, Shuffle: true, Seed: 42})
	if err != nil {
		return err
	}

	for batch, err := range loader.Batches(context.Background(), 0) {
		if err != nil {
			return err
		}

		fmt.Printf("shape=%v labels=%v\n", batch.Inputs.Shape(), batch.Labels)

		if err := batch.Close(); err != nil {
			return err
		}
	}

	return nil
}
