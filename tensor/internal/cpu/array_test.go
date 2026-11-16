package cpu_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/tensor/internal/cpu"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/shape"
)

func testView(t *testing.T, values []float32, dimensions ...int) backend.View {
	t.Helper()

	size, strides, err := shape.Layout(dimensions)
	if err != nil {
		t.Fatal(err)
	}

	buffer, err := (cpu.Device{}).Allocate(size)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = buffer.Close() })

	if values != nil {
		if err := (cpu.Device{}).Upload(buffer, values); err != nil {
			t.Fatal(err)
		}
	}

	return backend.View{Buffer: buffer, Shape: dimensions, Strides: strides}
}

func TestArray_ReadStridedLogicalValues(t *testing.T) {
	view := testView(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	view.Shape, view.Strides = []int{3, 2}, []int{1, 3}

	values := make([]float32, 6)
	if err := (cpu.Device{}).Read(view, values); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(values, []float32{1, 4, 2, 5, 3, 6}) {
		t.Fatalf("got %v", values)
	}
}

func TestHostAccess_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunHostAccess(t, device)
}

func TestArray_RejectsOverlappingWritesBeforeLayoutRestrictions(t *testing.T) {
	input := testView(t, []float32{1, 2, 3, 4}, 2, 2)
	output := testView(t, nil, 2, 2)

	output.Strides = []int{0, 1}
	if err := (cpu.Device{}).Scalar(backend.Scale, output, input, 2); !errors.Is(err, backend.ErrInvalidView) {
		t.Fatalf("overlapping destination: %v", err)
	}
}
