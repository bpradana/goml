//go:build !webgpu

package main

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

func openWebGPUDevice() (backend.Device, error) {
	return nil, fmt.Errorf("mnist: %w: build with -tags webgpu for WebGPU selection", backend.ErrUnsupported)
}
