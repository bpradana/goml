package cpu_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"

	"github.com/bpradana/goml/tensor/backend"
)

func TestBinary_BroadcastAndInvalidOutput(t *testing.T) {
	device := cpu.Device{}
	a := testView(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := testView(t, []float32{10, 20, 30}, 3)

	out := testView(t, nil, 2, 3)
	if err := device.Binary(backend.Add, out, a, b); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, 6)
	if err := device.Download(out.Buffer, got); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got, []float32{11, 22, 33, 14, 25, 36}) {
		t.Fatalf("got %v", got)
	}

	if err := device.Binary(backend.Add, b, a, b); !errors.Is(err, backend.ErrInvalidView) {
		t.Fatalf("got %v", err)
	}

	if err := device.Binary(255, out, a, b); !errors.Is(err, backend.ErrUnsupported) {
		t.Fatalf("got %v", err)
	}
}

func TestBinary_EmptyViewWithUnusedOffset(t *testing.T) {
	device := cpu.Device{}
	empty := testView(t, nil, 0, 3)
	empty.Offset = 99

	out := testView(t, nil, 0, 3)
	if err := device.Binary(backend.Add, out, empty, empty); err != nil {
		t.Fatal(err)
	}

	if err := device.Scalar(backend.Scale, out, empty, 2); err != nil {
		t.Fatal(err)
	}
}

func TestArithmetic_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunArithmetic(t, device)
}
