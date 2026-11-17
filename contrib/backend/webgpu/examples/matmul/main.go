package main

import (
	"fmt"
	"os"

	"github.com/bpradana/goml/contrib/backend/webgpu"
	"github.com/bpradana/goml/tensor"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	device, err := webgpu.Open()
	if err != nil {
		return err
	}
	defer func() { _ = device.Close() }()

	left, err := tensor.FromOn(device, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		return err
	}
	defer func() { _ = left.Close() }()

	right, err := tensor.FromOn(device, []float32{1, 0, 0, 1, 1, 1}, 3, 2)
	if err != nil {
		return err
	}
	defer func() { _ = right.Close() }()

	result, err := tensor.MatMul(left, right)
	if err != nil {
		return err
	}
	defer func() { _ = result.Close() }()

	values, err := result.HostData()
	if err != nil {
		return err
	}

	fmt.Printf("%s: shape=%v values=%v\n", device.Name(), result.Shape(), values)

	return nil
}
