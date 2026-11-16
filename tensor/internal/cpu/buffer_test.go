package cpu_test

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

func TestBuffer_GCManagedStorageRetainsExplicitCloseSemantics(t *testing.T) {
	buffer, err := (cpu.Device{}).Allocate(1)
	if err != nil {
		t.Fatal(err)
	}

	managed, ok := buffer.(backend.GCManagedBuffer)
	if !ok || !managed.GCManaged() {
		t.Fatal("CPU memory should not require a native cleanup queue")
	}

	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := buffer.Valid(); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("got %v", err)
	}
}

func TestBuffer_OwnPreservesPrivateMemoryAndCloseReleasesIt(t *testing.T) {
	buffer := cpu.Own([]float32{2, 3})

	values := make([]float32, 2)
	if err := (cpu.Device{}).Download(buffer, values); err != nil {
		t.Fatal(err)
	}

	if values[0] != 2 || values[1] != 3 {
		t.Fatal(values)
	}

	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := (cpu.Device{}).Download(buffer, values); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("got %v", err)
	}
}
