//go:build metal

package main

import (
	"fmt"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func openDevice(name string) (backend.Device, error) {
	switch name {
	case "", "cpu":
		return tensor.CPU(), nil
	case "metal":
		return metal.Open()
	default:
		return nil, fmt.Errorf("mnist: unknown device %q", name)
	}
}
