package cpu_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"

	"github.com/bpradana/goml/tensor/backend"
)

func TestDevice_TransfersCopyAndCloseIsIdempotent(t *testing.T) {
	device := cpu.Device{}

	buffer, err := device.Allocate(3)
	if err != nil {
		t.Fatal(err)
	}

	values := []float32{1, 2, 3}
	if err := device.Upload(buffer, values); err != nil {
		t.Fatal(err)
	}

	values[0] = 99

	got := make([]float32, 3)
	if err := device.Download(buffer, got); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got, []float32{1, 2, 3}) {
		t.Fatalf("got %v", got)
	}

	if err := device.Close(); err != nil {
		t.Fatal(err)
	}

	if err := buffer.Valid(); err != nil {
		t.Fatal("shared CPU Close invalidated allocation", err)
	}

	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := device.Download(buffer, got); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("got %v", err)
	}
}

func TestDevice_RejectsInvalidLengthsAndAllocation(t *testing.T) {
	device := cpu.Device{}
	if _, err := device.Allocate(-1); !errors.Is(err, backend.ErrAllocation) {
		t.Fatalf("got %v", err)
	}

	buffer, err := device.Allocate(2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = buffer.Close() }()

	if err := device.Upload(buffer, []float32{1}); !errors.Is(err, backend.ErrInvalidView) {
		t.Fatalf("got %v", err)
	}

	if err := device.Download(buffer, make([]float32, 1)); !errors.Is(err, backend.ErrInvalidView) {
		t.Fatalf("got %v", err)
	}

	if err := device.Upload(nil, nil); !errors.Is(err, backend.ErrDeviceMismatch) {
		t.Fatalf("got %v", err)
	}
}

func TestDevice_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunDevice(t, device)
}
