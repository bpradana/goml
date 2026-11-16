package activation

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestReLU_GradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{-2, -0.5, 0.5, 2}, NewReLU().Forward)
}

func TestReLU_ZeroDerivativeAtZero(t *testing.T) {
	input := testValue(t, []float32{0}, nil, true)

	output, err := NewReLU().Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if err := output.Backward(); err != nil {
		t.Fatal(err)
	}

	if backendtest.HostData(t, input.Grad())[0] != 0 {
		t.Fatal("ReLU derivative at zero must be zero")
	}
}

func TestReLU_ForwardPreservesStridedInputAndShape(t *testing.T) {
	data, err := tensor.From([]float32{-3, -2, -1, 0, 1, 2}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	view, err := data.Transpose()
	if err != nil {
		t.Fatal(err)
	}

	input, err := nn.NewValue(view, true)
	if err != nil {
		t.Fatal(err)
	}

	output, err := NewReLU().Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Shape(), []int{3, 2}) ||
		!slices.Equal(backendtest.HostData(t, output.Data()), []float32{0, 0, 0, 1, 0, 2}) {
		t.Fatal(output.Data())
	}

	if !slices.Equal(backendtest.HostData(t, data), []float32{-3, -2, -1, 0, 1, 2}) {
		t.Fatal("activation changed its input storage")
	}
}

func TestReLU_ForwardPropagatesSpecialValues(t *testing.T) {
	input := testValue(t, []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.Copysign(0, -1))}, []int{4}, false)

	output, err := NewReLU().Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	values := backendtest.HostData(t, output.Data())
	if !math.IsNaN(float64(values[0])) || !math.IsInf(float64(values[1]), 1) ||
		values[2] != 0 || values[3] != 0 || math.Signbit(float64(values[3])) {
		t.Fatal(values)
	}
}

func TestReLU_ForwardRejectsNilInput(t *testing.T) {
	if _, err := NewReLU().Forward(nil); err == nil {
		t.Fatal("expected invalid input error")
	}
}
