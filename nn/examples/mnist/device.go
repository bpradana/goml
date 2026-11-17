package main

import (
	"fmt"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func openDevice(name string) (backend.Device, error) {
	switch name {
	case "", "cpu":
		return tensor.CPU(), nil
	case "metal":
		return openMetalDevice()
	case "webgpu":
		return openWebGPUDevice()
	default:
		return nil, fmt.Errorf("mnist: unknown device %q", name)
	}
}
