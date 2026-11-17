package webgpu

import (
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestBuffer_TensorViewsAndScopes(t *testing.T) {
	backendtest.RunTensor(t, openDevice(t))
}

func TestBuffer_PendingCopyRetainsSourceUntilCompletion(t *testing.T) {
	device := openDevice(t)

	input, err := tensor.FromOn(device, []float32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = input.Close() }()

	output, err := input.MakeContiguous()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = output.Close() }()

	if err := input.Close(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBuffers != 2 {
		t.Fatalf("queued copy lost physical ownership: %+v", stats)
	}

	values := backendtest.HostData(t, output)
	for i, value := range values {
		if value != float32(i+1) {
			t.Fatalf("element %d: %g", i, value)
		}
	}

	if err := output.Close(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBytes != 0 || stats.LiveBuffers != 0 {
		t.Fatalf("completed copies retained storage: %+v", stats)
	}
}

func BenchmarkBuffer_TransferRoundTrip4096(b *testing.B) {
	device := openDevice(b)

	buffer, err := device.Allocate(4096)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = buffer.Close() }()

	values, output := make([]float32, 4096), make([]float32, 4096)

	b.ReportAllocs()
	b.SetBytes(int64(len(values)) * 8)

	for b.Loop() {
		if err := device.Upload(buffer, values); err != nil {
			b.Fatal(err)
		}

		if err := device.Download(buffer, output); err != nil {
			b.Fatal(err)
		}
	}
}
