package main

import (
	"encoding/json"
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

	// JSON-safe undefined R² keeps the numerical NaN/validity contract in Go.
	metric.Reset()

	if err := metric.Update([]float32{2}, []float32{1}); err != nil {
		return err
	}

	undefined, err := metric.Result()
	if err != nil {
		return err
	}

	encoded, err := json.Marshal(undefined)
	if err != nil {
		return err
	}

	fmt.Println(string(encoded))

	return nil
}
