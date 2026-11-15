package optimizer

import (
	"math"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestSGD_StepMatchesReferenceAndPreservesIdentity(t *testing.T) {
	parameter := testParameter(t, []float32{1, -2})
	leaf := parameter.Value()
	oldData := leaf.Data()

	accumulateGradient(t, parameter, []float32{2, -4})

	opt, err := NewSGD([]*nn.Parameter{parameter, parameter}, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}

	for i, want := range []float32{0.8, -1.6} {
		if math.Abs(float64(leaf.Data().Data()[i]-want)) > 1e-6 {
			t.Fatalf("element %d: got %v, want %v", i, leaf.Data(), want)
		}
	}

	if parameter.Value() != leaf || oldData.Data()[0] != 1 {
		t.Fatal("step changed identity or mutated previous tensor")
	}

	if leaf.Grad() == nil {
		t.Fatal("step must retain gradients")
	}

	opt.ZeroGrad()

	if leaf.Grad() != nil {
		t.Fatal("ZeroGrad retained a gradient")
	}
}

func TestSGD_StepSkipsMissingGradients(t *testing.T) {
	parameter := testParameter(t, []float32{1})

	opt, err := NewSGD([]*nn.Parameter{parameter}, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}

	if parameter.Value().Data().Data()[0] != 1 {
		t.Fatal("parameter changed without a gradient")
	}
}
