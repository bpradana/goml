package optimizer

import (
	"math"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func testParameter(t *testing.T, values []float32) *nn.Parameter {
	t.Helper()

	data, err := tensor.From(values, len(values))
	if err != nil {
		t.Fatal(err)
	}

	parameter, err := nn.NewParameter(data)
	if err != nil {
		t.Fatal(err)
	}

	return parameter
}

func accumulateGradient(t *testing.T, parameter *nn.Parameter, values []float32) {
	t.Helper()

	data, err := tensor.From(values, len(values))
	if err != nil {
		t.Fatal(err)
	}

	gradient, err := nn.NewValue(data, false)
	if err != nil {
		t.Fatal(err)
	}

	product, err := nn.Mul(parameter.Value(), gradient)
	if err != nil {
		t.Fatal(err)
	}

	loss, err := nn.Sum(product)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
}

func TestOptimizerParameters_RejectsInvalidLearningRates(t *testing.T) {
	for _, rate := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := NewSGD(nil, rate); err == nil {
			t.Fatalf("accepted invalid rate %v", rate)
		}
	}
}

func TestOptimizerParameters_RejectsInvalidParameters(t *testing.T) {
	for _, parameter := range []*nn.Parameter{nil, {}} {
		if _, err := NewSGD([]*nn.Parameter{parameter}, 0.1); err == nil {
			t.Fatal("expected invalid parameter error")
		}
	}
}
