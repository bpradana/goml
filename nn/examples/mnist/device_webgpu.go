//go:build webgpu

package main

import (
	"github.com/bpradana/goml/contrib/backend/webgpu"
	"github.com/bpradana/goml/tensor/backend"
)

func openWebGPUDevice() (backend.Device, error) { return webgpu.Open() }
