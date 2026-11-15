package main

import (
	"fmt"
	"log"

	"github.com/bpradana/goml/tensor"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	a, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		return err
	}

	transposed, err := a.Transpose()
	if err != nil {
		return err
	}

	product, err := tensor.MatMul(a, transposed)
	if err != nil {
		return err
	}

	fmt.Println(a)
	fmt.Println(transposed)
	fmt.Println(product)

	return nil
}
