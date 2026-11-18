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

func BenchmarkLossReadback_Webgpu(b *testing.B) {
	device, err := webgpu.Open()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	benchmarkLossReadback(b, device, func() uint64 { return device.Stats().LiveBytes })
}
