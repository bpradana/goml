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
	metric := metrics.NewRegression()
	if err := metric.Update([]float32{2.5, 0}, []float32{3, -0.5}); err != nil {
		return err
	}

	if err := metric.Update([]float32{2, 8}, []float32{2, 7}); err != nil {
		return err
	}

	result, err := metric.Result()
	if err != nil {
		return err
	}

	fmt.Printf("mae=%.4f mse=%.4f rmse=%.4f r2=%g valid=%t\n", result.MAE, result.MSE, result.RMSE, result.R2, result.R2Valid)

	return nil
}
