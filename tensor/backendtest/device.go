package backendtest

import (
	"errors"
	"sync"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunDevice checks allocation, exact transfers, buffer identity, closed-resource
// errors, foreign-buffer rejection, and concurrent use of independent buffers.
func RunDevice(t *testing.T, device backend.Device) {
	t.Helper()
	t.Run("Device/Identity", func(t *testing.T) {
		if device.ID() == "" || device.Name() == "" {
			t.Fatal("device identity must be nonempty")
		}
	})
	t.Run("Device/TransfersCopy", func(t *testing.T) {
		source := []float32{1, -2, 3, 4}

		value := view(t, device, source, 4)
		if value.Buffer.DeviceID() != device.ID() || value.Buffer.Len() != 4 {
			t.Fatal("allocation identity or length mismatch")
		}

		source[0] = 99

		compare(t, device, value, []float32{1, -2, 3, 4}, 0)

		if err := device.Upload(value.Buffer, []float32{1}); !errors.Is(err, backend.ErrInvalidView) {
			t.Fatalf("upload length: %v", err)
		}

		if err := device.Download(value.Buffer, make([]float32, 1)); !errors.Is(err, backend.ErrInvalidView) {
			t.Fatalf("download length: %v", err)
		}

		compare(t, device, value, []float32{1, -2, 3, 4}, 0)
	})
	t.Run("Device/EmptyAllocation", func(t *testing.T) {
		value := view(t, device, []float32{}, 0)
		compare(t, device, value, nil, 0)
	})
	t.Run("Device/InvalidAllocation", func(t *testing.T) {
		if _, err := device.Allocate(-1); !errors.Is(err, backend.ErrAllocation) {
			t.Fatalf("got %v", err)
		}
	})
	t.Run("Device/IdempotentRelease", func(t *testing.T) {
		value := view(t, device, []float32{3}, 1)
		if err := value.Buffer.Close(); err != nil {
			t.Fatal(err)
		}

		if err := value.Buffer.Close(); err != nil {
			t.Fatal(err)
		}

		if err := value.Buffer.Valid(); !errors.Is(err, backend.ErrClosed) {
			t.Fatalf("valid after release: %v", err)
		}

		if err := device.Download(value.Buffer, make([]float32, 1)); !errors.Is(err, backend.ErrClosed) {
			t.Fatalf("download after release: %v", err)
		}

		if err := device.Upload(value.Buffer, []float32{1}); !errors.Is(err, backend.ErrClosed) {
			t.Fatalf("upload after release: %v", err)
		}
	})
	t.Run("Device/ForeignBuffer", func(t *testing.T) {
		foreign := foreignBuffer{}
		if err := device.Upload(foreign, []float32{1}); !errors.Is(err, backend.ErrDeviceMismatch) {
			t.Fatalf("foreign upload: %v", err)
		}

		if err := device.Download(foreign, make([]float32, 1)); !errors.Is(err, backend.ErrDeviceMismatch) {
			t.Fatalf("foreign download: %v", err)
		}
	})
	t.Run("Device/ConcurrentIndependentAllocations", func(t *testing.T) {
		var workers sync.WaitGroup
		for worker := range 8 {
			workers.Go(func() {
				buffer, err := device.Allocate(2)
				if err != nil {
					t.Error(err)
					return
				}
				defer func() { _ = buffer.Close() }()

				values := []float32{float32(worker), float32(-worker)}
				if err := device.Upload(buffer, values); err != nil {
					t.Error(err)
					return
				}

				var actual [2]float32
				if err := device.Download(buffer, actual[:]); err != nil {
					t.Error(err)
					return
				}

				for i, want := range values {
					if actual[i] != want {
						t.Errorf("worker %d element %d: got %g, want %g", worker, i, actual[i], want)
					}
				}
			})
		}

		workers.Wait()

		if err := device.Synchronize(); err != nil {
			t.Fatal(err)
		}
	})
}

type foreignBuffer []float32

func (foreignBuffer) Len() int { return 1 }

func (foreignBuffer) DeviceID() string { return "backendtest:foreign" }

func (foreignBuffer) Valid() error { return nil }

func (foreignBuffer) Close() error { return nil }

// RunShutdown verifies deterministic shutdown of an owned context. It closes
// device and must run after its other suites. A shared CPU device has no shutdown.
func RunShutdown(t *testing.T, device backend.Device) {
	t.Helper()

	buffer, err := device.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}

	if err := device.Upload(buffer, []float32{7}); err != nil {
		t.Fatal(err)
	}

	if err := device.Close(); err != nil {
		t.Fatal(err)
	}

	if err := device.Close(); err != nil {
		t.Fatal(err)
	}

	if err := buffer.Valid(); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("buffer survives owned shutdown: %v", err)
	}

	if _, err := device.Allocate(1); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("allocation after shutdown: %v", err)
	}

	if err := device.Synchronize(); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("synchronize after shutdown: %v", err)
	}

	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}
}
