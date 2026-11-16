package main

import (
	"fmt"
	"os"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/tensor"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	device, err := metal.Open()
	if err != nil {
		return err
	}
	defer func() { _ = device.Close() }()

	scope, err := tensor.NewScope(device)
	if err != nil {
		return err
	}
	defer func() { _ = scope.Close() }()

	left, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		return err
	}

	right, err := tensor.From([]float32{1, 0, 0, 1, 1, 1}, 3, 2)
	if err != nil {
		return err
	}

	a, err := scope.To(left)
	if err != nil {
		return err
	}

	b, err := scope.To(right)
	if err != nil {
		return err
	}

	result, err := tensor.MatMul(a, b)
	if err != nil {
		return err
	}

	values, err := result.HostData()
	if err != nil {
		return err
	}

	fmt.Printf("%s: shape=%v values=%v\n", device.Name(), result.Shape(), values)

	return nil
}
