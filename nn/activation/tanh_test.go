package activation

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestTanh_GradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{-2, -0.5, 0.5, 2}, NewTanh().Forward)
}

func TestTanh_ExtremeInputsSaturate(t *testing.T) {
	input := testValue(t, []float32{-1000, 0, 1000}, []int{3}, false)
	stage := NewTanh()

	output, err := stage.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, output.Data()), []float32{-1, 0, 1}) || len(stage.Parameters()) != 0 {
		t.Fatal(output.Data())
	}
}

func TestTanh_ForwardRejectsNilInput(t *testing.T) {
	if _, err := NewTanh().Forward(nil); err == nil {
		t.Fatal("expected invalid input error")
	}
}
