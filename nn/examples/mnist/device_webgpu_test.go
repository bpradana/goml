//go:build webgpu && cgo

package main

import (
	"testing"

	"github.com/bpradana/goml/contrib/backend/webgpu"
)

func TestDevice_WebGPUSelectedExplicitly(t *testing.T) {
	device, err := openDevice("webgpu")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	if device.ID() == "" {
		t.Fatal("missing context identity")
	}
}

func TestTrainer_WebGPUParityConvergenceAndStableMemory(t *testing.T) {
	device, err := webgpu.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	checkTrainerDevice(t, device, func() (uint64, uint64) { stats := device.Stats(); return stats.LiveBytes, stats.PeakBytes })
}
