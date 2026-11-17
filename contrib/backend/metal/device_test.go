//go:build darwin && cgo

package metal_test

import (
	"math"
	"sync"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/tensor"
)

func openDevice(t *testing.T) *metal.Device {
	t.Helper()

	device, err := metal.Open()
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Error(err)
		}
	})

	return device
}

func TestDevice_ConcurrentScopesReleaseAllAllocations(t *testing.T) {
	device := openDevice(t)
	failures := make(chan error, 8)

	var workers sync.WaitGroup

	for range 8 {
		workers.Go(func() {
			scope, err := tensor.NewScope(device)
			if err != nil {
				failures <- err
				return
			}
			defer func() { _ = scope.Close() }()

			input, err := scope.To(tensor.Scalar(2))
			if err != nil {
				failures <- err
				return
			}

			for range 20 {
				output, err := tensor.Scale(input, 3)
				if err != nil {
					failures <- err
					return
				}

				if _, err := output.HostData(); err != nil {
					failures <- err
					return
				}
			}
		})
	}

	workers.Wait()
	close(failures)

	for err := range failures {
		t.Error(err)
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBytes != 0 || stats.LiveBuffers != 0 {
		t.Fatalf("concurrent scopes retained allocations: %+v", stats)
	}
}

func TestDevice_CloseWaitsForPendingWorkAndIsIdempotent(t *testing.T) {
	device := openDevice(t)

	input, err := tensor.Scalar(2).To(device)
	if err != nil {
		t.Fatal(err)
	}

	output, err := tensor.Scale(input, 5)
	if err != nil {
		t.Fatal(err)
	}

	if err := device.Close(); err != nil {
		t.Fatal(err)
	}

	if !input.IsClosed() || !output.IsClosed() {
		t.Fatal("device shutdown left tensor storage valid")
	}

	if err := device.Close(); err != nil {
		t.Fatal("repeated Close failed", err)
	}

	if err := device.Synchronize(); err == nil {
		t.Fatal("synchronization accepted a closed device")
	}
}

func assertValues(t *testing.T, got *tensor.Tensor, expected []float32) {
	t.Helper()

	values, err := got.HostData()
	if err != nil {
		t.Fatal(err)
	}

	if len(values) != len(expected) {
		t.Fatalf("length: got %d, expected %d", len(values), len(expected))
	}

	for index, value := range values {
		difference := math.Abs(float64(value - expected[index]))
		if math.IsNaN(difference) || difference > 1e-4+1e-4*math.Abs(float64(expected[index])) {
			t.Fatalf("element %d: got %g, expected %g", index, value, expected[index])
		}
	}
}

func TestDevice_UploadViewsAndExplicitDownload(t *testing.T) {
	device := openDevice(t)

	source, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	gpu, err := source.To(device)
	if err != nil {
		t.Fatal(err)
	}

	transposed, err := gpu.Transpose()
	if err != nil {
		t.Fatal(err)
	}

	if err := gpu.Close(); err != nil {
		t.Fatal(err)
	}

	assertValues(t, transposed, []float32{1, 4, 2, 5, 3, 6})

	if err := transposed.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := transposed.HostData(); err == nil {
		t.Fatal("closed view remained readable")
	}
}

func TestDevice_ScopeRetainsPendingCommandsAndKeptResults(t *testing.T) {
	device := openDevice(t)

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	input, err := scope.To(tensor.Scalar(7))
	if err != nil {
		t.Fatal(err)
	}

	output, err := tensor.Scale(input, 3)
	if err != nil {
		t.Fatal(err)
	}

	output.Keep()

	if err := scope.Close(); err != nil {
		t.Fatal(err)
	}

	assertValues(t, output, []float32{21})

	if _, err := tensor.Scale(input, 2); err == nil {
		t.Fatal("closed scoped input accepted")
	}

	if err := output.Close(); err != nil {
		t.Fatal(err)
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBytes != 0 || stats.LiveBuffers != 0 {
		t.Fatalf("native allocations retained after synchronization: %+v", stats)
	}
}

func TestDevice_Conformance(t *testing.T) {
	device := openDevice(t)
	backendtest.RunDevice(t, device)
	backendtest.RunShutdown(t, device)
}

func BenchmarkDeviceLifecycle(b *testing.B) {
	b.Run("context_startup", func(b *testing.B) {
		b.ReportAllocs()

		for b.Loop() {
			device, err := metal.Open()
			if err != nil {
				b.Fatal(err)
			}

			if err := device.Close(); err != nil {
				b.Fatal(err)
			}
		}
	})

	device, err := metal.Open()
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = device.Close() }()

	b.Run("allocate_release_4096", func(b *testing.B) {
		b.ReportAllocs()

		for b.Loop() {
			buffer, err := device.Allocate(4096)
			if err != nil {
				b.Fatal(err)
			}

			if err := buffer.Close(); err != nil {
				b.Fatal(err)
			}
		}
	})

	buffer, err := device.Allocate(4096)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = buffer.Close() }()

	values := make([]float32, 4096)

	b.Run("upload_4096", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(values)) * 4)

		for b.Loop() {
			if err := device.Upload(buffer, values); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("download_4096", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(values)) * 4)

		for b.Loop() {
			if err := device.Download(buffer, values); err != nil {
				b.Fatal(err)
			}
		}
	})
}
