package main

import (
	"fmt"
	"os"

	"github.com/bpradana/goml/metrics"
	"github.com/bpradana/goml/metrics/examples/custom/meanbias"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	var metric metrics.Metric[float32, float32, meanbias.Result] = meanbias.New()
	if err := metric.Update([]float32{2, 4}, []float32{1, 3}); err != nil {
		return err
	}

	if err := metric.Update([]float32{2.5}, []float32{3}); err != nil {
		return err
	}

	result, err := metric.Result()
	if err != nil {
		return err
	}

	fmt.Printf("observations=%d mean_bias=%g\n", result.Count, result.MeanBias)

	return nil
}
