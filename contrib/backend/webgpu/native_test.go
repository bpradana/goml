//go:build cgo && !android && !ios && (darwin || linux || windows) && (amd64 || arm64)

package webgpu

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/oliverbestmann/webgpu/wgpu"
)

func TestNative_InvalidShaderReturnsErrorAndContextRemainsUsable(t *testing.T) {
	device := openDevice(t)

	module, err := device.native.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSLSource: &wgpu.ShaderSourceWGSL{Code: "not valid WGSL"}})
	if err == nil {
		module.Release()
		t.Fatal("invalid WGSL accepted")
	}

	if err := device.Synchronize(); err != nil {
		t.Fatalf("shader validation poisoned the context: %v", err)
	}
}

func TestNative_CallbackFailureIsIsolatedToItsContext(t *testing.T) {
	first, second := openDevice(t), openDevice(t)

	buffer, err := first.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}

	if err := first.Upload(buffer, []float32{3}); err != nil {
		t.Fatal(err)
	}

	want := errors.New("injected device-loss callback")
	first.native.failure.set(want)

	if _, err := first.Allocate(1); !errors.Is(err, want) {
		t.Fatalf("lost context accepted allocation: %v", err)
	}

	if err := second.Synchronize(); err != nil {
		t.Fatalf("independent context inherited failure: %v", err)
	}

	// Close still cleans resources after reporting the sticky native error.
	if err := first.Close(); !errors.Is(err, want) {
		t.Fatalf("Close did not report native error: %v", err)
	}

	if first.native != nil || first.Stats().LiveBuffers != 0 {
		t.Fatal("failed-context shutdown retained native resources")
	}
}

func TestNative_DispatchCrossesWorkgroupDimensionBoundary(t *testing.T) {
	device := openDevice(t)

	count := uint64(device.native.limits.MaxComputeWorkgroupsPerDimension)*64 + 13
	if count*4 > device.native.limits.MaxStorageBufferBindingSize {
		t.Skip("storage limit is reached before a two-dimensional dispatch is needed")
	}

	buffer, err := device.Allocate(int(count))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = buffer.Close() }()

	view := backend.View{Buffer: buffer, Shape: []int{int(count)}, Strides: []int{1}}
	if err := device.Fill(view, 3); err != nil {
		t.Fatal(err)
	}

	values := make([]float32, int(count))
	if err := device.Download(buffer, values); err != nil {
		t.Fatal(err)
	}

	for i, value := range values {
		if value != 3 {
			t.Fatalf("dispatch element %d: got %g, want 3", i, value)
		}
	}
}
