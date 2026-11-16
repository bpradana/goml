package cpu_test

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

func TestSubScaled_FusedImmutableUpdate(t *testing.T) {
	device := cpu.Device{}
	values := testView(t, []float32{1, 2}, 2)
	gradient := testView(t, []float32{2, 4}, 2)

	output := testView(t, nil, 2)
	if err := device.SubScaled(output, values, gradient, 0.25); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, 2)
	if err := device.Download(output.Buffer, got); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got, []float32{0.5, 1}) {
		t.Fatalf("got %v", got)
	}

	if err := device.Download(values.Buffer, got); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got, []float32{1, 2}) {
		t.Fatal("update mutated input")
	}
}

func TestAdam_ValidatesBeforeUpdatingMoments(t *testing.T) {
	device := cpu.Device{}
	values, gradient, output := testView(t, []float32{1}, 1), testView(t, []float32{2}, 1), testView(t, nil, 1)

	first, second := []float32{3}, []float32{4}
	if err := device.Adam(output, values, gradient, first, second, backend.AdamOptions{Rate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8}); err == nil {
		t.Fatal("step zero accepted")
	}

	if first[0] != 3 || second[0] != 4 {
		t.Fatal("invalid update changed moments")
	}
}

func TestSubScaled_Conformance(t *testing.T) { backendtest.RunSubScaled(t, cpu.Device{}) }

func TestAdam_Conformance(t *testing.T) { backendtest.RunHostAdam(t, cpu.Device{}) }
