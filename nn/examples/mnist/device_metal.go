//go:build metal

package main

import (
	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/tensor/backend"
)

func openMetalDevice() (backend.Device, error) {
	return metal.Open()
}
