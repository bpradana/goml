//go:build !metal

package main

import (
	"fmt"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func openDevice(name string) (backend.Device, error) {
	if name == "" || name == "cpu" {
		return tensor.CPU(), nil
	}

	return nil, fmt.Errorf("mnist: unsupported device %q; Metal requires -tags metal and supported macOS hardware", name)
}
