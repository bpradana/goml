//go:build !metal

package main

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

func openMetalDevice() (backend.Device, error) {
	return nil, fmt.Errorf("mnist: %w: build with -tags metal for Metal selection", backend.ErrUnsupported)
}
