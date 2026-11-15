package layer

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestLSTM_ReferenceGatesAndExplicitState(t *testing.T) {
	recurrent, err := NewLSTM(1, 1, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	setParameter(t, recurrent.Parameters()[0], []float32{0.2, -0.1, 0.4, 0.3})
	setParameter(t, recurrent.Parameters()[1], []float32{0.1, 0.2, -0.1, 0.3})
	setParameter(t, recurrent.Parameters()[2], []float32{0.3, 0.2, -0.2, 0.1})
	input := layerValue(t, []float32{0.5, -0.2}, []int{1, 2, 1}, false)
	hidden := layerValue(t, []float32{0.2}, []int{1, 1}, true)
	cell := layerValue(t, []float32{-0.1}, []int{1, 1}, true)

	output, lastHidden, lastCell, err := recurrent.ForwardState(input, hidden, cell)
	if err != nil {
		t.Fatal(err)
	}

	h, c := float64(0.2), float64(-0.1)

	sigmoid := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	for step, x := range []float64{0.5, -0.2} {
		i := sigmoid(x*0.2 + h*0.3 + 0.1)
		f := sigmoid(x*-0.1 + h*0.2 + 0.2)
		candidate := math.Tanh(x*0.4 + h*-0.2 - 0.1)
		o := sigmoid(x*0.3 + h*0.1 + 0.3)
		c = f*c + i*candidate

		h = o * math.Tanh(c)
		if math.Abs(float64(output.Data().Data()[step])-h) > 1e-6 {
			t.Fatalf("step %d: got %v, want %g", step, output.Data(), h)
		}
	}

	if math.Abs(float64(lastCell.Data().Data()[0])-c) > 1e-6 || math.Abs(float64(lastHidden.Data().Data()[0])-h) > 1e-6 {
		t.Fatal("incorrect final states")
	}

	objective, err := nn.Sum(lastHidden)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	for _, state := range []*nn.Value{hidden, cell} {
		original := state.Data().Data()[0]
		results := [2]float32{}

		for i, delta := range []float32{-1e-3, 1e-3} {
			h0, c0 := hidden, cell

			perturbed := layerValue(t, []float32{original + delta}, []int{1, 1}, false)
			if state == hidden {
				h0 = perturbed
			} else {
				c0 = perturbed
			}

			_, result, _, err := recurrent.ForwardState(input, h0, c0)
			if err != nil {
				t.Fatal(err)
			}

			results[i] = result.Data().Data()[0]
		}

		assertDerivative(t, "initial state", 0, state.Grad().Data()[0], float64((results[1]-results[0])/0.002))
	}
}

func TestLSTM_BatchedSequenceAndPackedParametersMatchFiniteDifferences(t *testing.T) {
	recurrent, err := NewLSTM(2, 2, rand.New(rand.NewPCG(7, 8)))
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(recurrent.Parameters()[1].Value().Data().Data(), []float32{0, 0, 1, 1, 0, 0, 0, 0}) {
		t.Fatal("forget gate bias was not initialized to one")
	}

	checkLayerGradients(t, recurrent, []float32{0.1, 0.3, -0.2, 0.4, 0.2, -0.1, -0.3, 0.2}, []int{2, 2, 2})
}

func TestLSTM_RejectsInvalidConfigurationAndStates(t *testing.T) {
	for _, hidden := range []int{0, -1, int(^uint(0) >> 1)} {
		if _, err := NewLSTM(1, hidden, rand.New(rand.NewPCG(1, 2))); err == nil {
			t.Fatalf("accepted hidden size %d", hidden)
		}
	}

	recurrent, err := NewLSTM(1, 1, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	input := layerValue(t, []float32{1}, []int{1, 1, 1}, false)

	badState := layerValue(t, []float32{1, 2}, []int{1, 2}, false)
	if _, _, _, err := recurrent.ForwardState(input, nil, badState); err == nil {
		t.Fatal("accepted mismatched cell state")
	}

	if _, err := NewLSTM(1, 1, nil); err == nil {
		t.Fatal("accepted nil generator")
	}

	if _, err := (&LSTM{}).Forward(input); err == nil {
		t.Fatal("accepted uninitialized layer")
	}
}
