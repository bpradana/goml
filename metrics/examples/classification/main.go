package main

import (
	"fmt"
	"os"

	"github.com/bpradana/goml/metrics"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	metric, err := metrics.NewClassification(3)
	if err != nil {
		return err
	}

	if err := metric.Update([]int{0, 0, 0}, []int{0, 0, 0}); err != nil {
		return err
	}

	if err := metric.UpdateLogits([]float32{0, 2, 1, 0, 3, 1, 0, 0, 4}, []int{0, 1, 1}); err != nil {
		return err
	}

	result, err := metric.Result()
	if err != nil {
		return err
	}

	fmt.Printf("accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f confusion=%v\n", result.Accuracy, result.F1Macro, result.F1Weighted, result.Confusion)

	return nil
}
