package webgpu

import (
	"errors"
	"os"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/backendtest"
)

func openDevice(t testing.TB) *Device {
	t.Helper()

	device, err := Open()
	if err != nil {
		if errors.Is(err, backend.ErrUnsupported) && os.Getenv("GOML_WEBGPU_REQUIRED") != "1" {
			t.Skipf("native WebGPU unavailable: %v", err)
		}

		t.Fatal(err)
	}

	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Error(err)
		}
	})

	return device
}

func TestDevice_ConformanceAndShutdown(t *testing.T) {
	device := openDevice(t)
	t.Logf("native adapter: %s", device.Name())
	backendtest.RunDevice(t, device)
	backendtest.RunShutdown(t, device)
}

func TestOpenDevice_RejectsInvalidIndex(t *testing.T) {
	if device, err := OpenDevice(-2); err == nil || device != nil {
		t.Fatalf("OpenDevice(-2) = %v, %v", device, err)
	}
}

func TestDevice_IndependentContextsRejectForeignAllocations(t *testing.T) {
	first, second := openDevice(t), openDevice(t)
	if first.ID() == second.ID() {
		t.Fatal("independent contexts share an identity")
	}

	buffer, err := first.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = buffer.Close() }()

	if err := second.Upload(buffer, []float32{1}); !errors.Is(err, backend.ErrDeviceMismatch) {
		t.Fatalf("foreign upload: %v", err)
	}
}

func TestDevice_ZeroValueRejectsExecution(t *testing.T) {
	var device Device
	if _, err := device.Allocate(1); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("uninitialized allocation: %v", err)
	}

	if err := device.Synchronize(); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("uninitialized synchronization: %v", err)
	}

	if err := device.Close(); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkDevice_ContextStartup(b *testing.B) {
	b.ReportAllocs()

	for b.Loop() {
		device, err := Open()
		if err != nil {
			b.Fatal(err)
		}

		if err := device.Close(); err != nil {
			b.Fatal(err)
		}
	}
}
